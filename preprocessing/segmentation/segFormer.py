"""
SegFormer 语义分割模型
从 arch/FaultyYawLanding/perception/segmentation/segFormer.py 迁移
适配 Orin 真机部署
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

# 定义不同变体的配置
MIT_CONFIGS = {
    'b0': {'embed_dims': [32, 64, 160, 256], 'depths': [2, 2, 2, 2]},
    'b1': {'embed_dims': [64, 128, 320, 512], 'depths': [2, 2, 2, 2]},
    'b2': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 6, 3]},
    'b3': {'embed_dims': [64, 128, 320, 512], 'depths': [3, 4, 18, 3]},
}


class MixVisionTransformer(nn.Module):
    def __init__(self, variant='b0', pretrained=True, weight_path=None):
        super().__init__()
        cfg = MIT_CONFIGS[variant]
        embed_dims = cfg['embed_dims']
        depths = cfg['depths']
        num_heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        mlp_ratios = [4, 4, 4, 4]

        self.variant = variant
        self.patch_embed1 = OverlapPatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = OverlapPatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = OverlapPatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = OverlapPatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        self.block1 = nn.ModuleList([Block(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0]) for _ in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.block2 = nn.ModuleList([Block(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.block3 = nn.ModuleList([Block(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2]) for _ in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.block4 = nn.ModuleList([Block(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3]) for _ in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

        if pretrained and weight_path and os.path.exists(weight_path):
            self._load_weights(weight_path)

    def _load_weights(self, weight_path):
        try:
            print(f"[SegFormer] Loading backbone weights from {weight_path} ...")
            state_dict = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # 去除 'backbone.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v
            model_keys = set(self.state_dict().keys())
            load_keys = set(new_state_dict.keys())
            common_keys = model_keys.intersection(load_keys)
            print(f"[SegFormer] Loaded {len(common_keys)} / {len(model_keys)} keys.")
            self.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print(f"[SegFormer] Warning: Failed to load weights: {e}")

    def forward(self, x):
        outs = []
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        outs.append(self.norm1(x).reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2))

        x, H, W = self.patch_embed2(outs[-1])
        for blk in self.block2: x = blk(x, H, W)
        outs.append(self.norm2(x).reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2))

        x, H, W = self.patch_embed3(outs[-1])
        for blk in self.block3: x = blk(x, H, W)
        outs.append(self.norm3(x).reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2))

        x, H, W = self.patch_embed4(outs[-1])
        for blk in self.block4: x = blk(x, H, W)
        outs.append(self.norm4(x).reshape(x.shape[0], H, W, -1).permute(0, 3, 1, 2))
        return outs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c, out_c, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(out_c)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.kv(self.norm(x_)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        if H is None or W is None:
            import math
            side = int(math.sqrt(N))
            H = W = side
        feat = self.fc1(x)
        feat = self.dwconv(feat.transpose(1, 2).view(B, -1, H, W)).flatten(2).transpose(1, 2)
        return self.fc2(self.act(feat))


class SegFormer(nn.Module):
    """SegFormer 语义分割模型"""
    def __init__(self, num_classes=10, variant='b0', pretrained=True, weight_path=None):
        super().__init__()
        self.backbone = MixVisionTransformer(variant=variant, pretrained=pretrained, weight_path=weight_path)

        embed_dims = MIT_CONFIGS[variant]['embed_dims']
        self.decoder = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in embed_dims])
        self.fusion = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        features = self.backbone(x)
        outs = []
        for i, feat in enumerate(features):
            feat = self.decoder[i](feat)
            feat = F.interpolate(feat, size=features[0].shape[2:], mode='bilinear', align_corners=False)
            outs.append(feat)
        out = self.fusion(torch.cat(outs, dim=1))
        out = self.classifier(out)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
