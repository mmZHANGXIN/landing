"""
离线预处理模块
飞行前一次性执行: GIS 卫星图下载 → 语义分割 → 九宫格风险评估 → 全局安全着陆点
"""

from .gis_fetcher import GISFetcher, fetch_gis_image
from .global_safety_prior import GlobalSafetyPrior
