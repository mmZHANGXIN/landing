"""
GIS 卫星影像下载模块
从 arch/FaultyYawLanding/gis/gis_hd.py 迁移适配

功能: 从 Bing Maps 下载指定 GPS 坐标周围的高清卫星图, 保存为 GeoTIFF + PNG。
用于离线预处理阶段的 GIS 影像获取。
"""

import requests
import numpy as np
import math
import io
import os

try:
    import rasterio
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[GISFetcher] rasterio not installed, GeoTIFF output disabled.")

from PIL import Image


class GISFetcher:
    """
    Bing Maps 卫星瓦片下载器。

    用法:
        fetcher = GISFetcher()
        fetcher.download(
            center_lat=22.7295, center_lon=113.9066,
            zoom=18, target_w=350, target_h=350,
            output_dir="./gis_data"
        )
    """

    # Bing 瓦片服务器
    TILE_SERVERS = ['t0', 't1', 't2', 't3']
    TILE_SIZE = 256

    def __init__(self):
        pass

    # ---- 坐标转换工具 ----
    @staticmethod
    def _deg2tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(math.radians(lat)) +
                 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        return x, y

    @staticmethod
    def _tile2deg(x, y, zoom):
        n = 2 ** zoom
        lon = x / n * 360 - 180
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return lat, lon

    @staticmethod
    def _quadkey(x, y, zoom):
        qk = []
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if x & mask: digit += 1
            if y & mask: digit += 2
            qk.append(str(digit))
        return ''.join(qk)

    def _download_tile(self, x, y, zoom):
        """下载单张瓦片"""
        s = self.TILE_SERVERS[(x + y) % 4]
        url = f"https://{s}.tiles.virtualearth.net/tiles/a{self._quadkey(x, y, zoom)}.jpeg?g=1"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert('RGB')
        except Exception:
            pass
        return None

    def download(self, center_lat: float, center_lon: float,
                 zoom: int = 18, target_w: int = 350, target_h: int = 350,
                 output_dir: str = "./gis_data",
                 save_geotiff: bool = True,
                 save_png: bool = True) -> dict:
        """
        下载指定 GPS 坐标周围卫星图。

        参数:
            center_lat, center_lon: 目标中心 GPS 坐标
            zoom: 瓦片级别 (18≈0.6m/px, 19≈0.3m/px)
            target_w, target_h: 输出图像像素尺寸
            output_dir: 输出目录
            save_geotiff: 是否保存 GeoTIFF
            save_png: 是否保存 PNG 预览

        返回:
            dict: {
                'img': np.ndarray (H, W, 3),
                'tif_path': str,
                'png_path': str,
                'center_lat': float,
                'center_lon': float,
                'resolution_m': float,
                'bounds': tuple (lon_left, lat_bot, lon_right, lat_top),
            }
        """
        os.makedirs(output_dir, exist_ok=True)

        cx, cy = self._deg2tile(center_lat, center_lon, zoom)

        # 计算需要下载的瓦片范围
        tiles_x = math.ceil(target_w / self.TILE_SIZE) + 2
        tiles_y = math.ceil(target_h / self.TILE_SIZE) + 2
        x_start = cx - tiles_x // 2
        y_start = cy - tiles_y // 2

        print(f"[GISFetcher] Center tile: ({cx}, {cy}), "
              f"downloading {tiles_x}x{tiles_y} tiles...")

        # 拼接瓦片
        canvas_w = tiles_x * self.TILE_SIZE
        canvas_h = tiles_y * self.TILE_SIZE
        canvas = Image.new('RGB', (canvas_w, canvas_h), (128, 128, 128))

        for dy in range(tiles_y):
            for dx in range(tiles_x):
                tx, ty = x_start + dx, y_start + dy
                tile_img = self._download_tile(tx, ty, zoom)
                if tile_img:
                    canvas.paste(tile_img, (dx * self.TILE_SIZE, dy * self.TILE_SIZE))

        # 裁剪到目标尺寸（以中心点为中心）
        cx_px = (cx - x_start) * self.TILE_SIZE + self.TILE_SIZE // 2
        cy_px = (cy - y_start) * self.TILE_SIZE + self.TILE_SIZE // 2
        left = cx_px - target_w // 2
        top = cy_px - target_h // 2
        right = left + target_w
        bottom = top + target_h
        cropped = canvas.crop((left, top, right, bottom))

        img_array = np.array(cropped)  # (H, W, 3) RGB

        # 计算地理范围
        lat_top, lon_left = self._tile2deg(
            x_start + left / self.TILE_SIZE,
            y_start + top / self.TILE_SIZE, zoom)
        lat_bot, lon_right = self._tile2deg(
            x_start + right / self.TILE_SIZE,
            y_start + bottom / self.TILE_SIZE, zoom)

        # 计算分辨率 (米/像素)
        res_deg = (lon_right - lon_left) / target_w
        res_m = res_deg * 111320 * math.cos(math.radians(center_lat))

        result = {
            'img': img_array,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'resolution_m': res_m,
            'bounds': (lon_left, lat_bot, lon_right, lat_top),
            'tif_path': None,
            'png_path': None,
        }

        # 保存 PNG
        if save_png:
            png_path = os.path.join(output_dir, "landing_area.png")
            Image.fromarray(img_array).save(png_path)
            result['png_path'] = png_path
            print(f"[GISFetcher] PNG saved: {png_path}")

        # 保存 GeoTIFF
        if save_geotiff and HAS_RASTERIO:
            tif_path = os.path.join(output_dir, "landing_area.tif")
            transform = from_bounds(lon_left, lat_bot, lon_right, lat_top,
                                    target_w, target_h)
            img_tif = img_array.transpose(2, 0, 1)  # (3, H, W)
            with rasterio.open(
                tif_path, 'w', driver='GTiff',
                height=target_h, width=target_w, count=3,
                dtype='uint8', crs='EPSG:4326',
                transform=transform) as dst:
                dst.write(img_tif)
            result['tif_path'] = tif_path
            print(f"[GISFetcher] GeoTIFF saved: {tif_path}")

        print(f"[GISFetcher] Done. Resolution: {res_m:.2f} m/px, "
              f"Coverage: {target_w * res_m:.1f}m x {target_h * res_m:.1f}m")
        return result


# ---- 快捷函数 ----
def fetch_gis_image(lat: float, lon: float, zoom: int = 18,
                    size: int = 512, output_dir: str = "./gis_data") -> dict:
    """一键下载 GIS 卫星影像"""
    fetcher = GISFetcher()
    return fetcher.download(
        center_lat=lat, center_lon=lon,
        zoom=zoom, target_w=size, target_h=size,
        output_dir=output_dir
    )


if __name__ == "__main__":
    # 示例: 下载深圳宝安某区域
    result = fetch_gis_image(22.7295, 113.9066, zoom=18, size=512)
    print(f"Resolution: {result['resolution_m']:.2f} m/px")
