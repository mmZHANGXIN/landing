"""
离线预处理模块
飞行前一次性执行: GIS 卫星图下载 → 语义分割 → 九宫格风险评估 → 全局安全着陆点
"""


def __getattr__(name):
    if name == "GISFetcher" or name == "fetch_gis_image":
        from .gis_fetcher import GISFetcher, fetch_gis_image
        return GISFetcher if name == "GISFetcher" else fetch_gis_image
    if name == "GlobalSafetyPrior":
        from .global_safety_prior import GlobalSafetyPrior
        return GlobalSafetyPrior
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GISFetcher", "fetch_gis_image", "GlobalSafetyPrior"]
