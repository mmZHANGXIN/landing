def __getattr__(name):
    if name == "MAVSDKController":
        from .mavsdk_controller import MAVSDKController
        return MAVSDKController
    if name == "MAVROSController":
        from .mavros_controller import MAVROSController
        return MAVROSController
    if name == "PoseSourceManager":
        from .pose_source_manager import PoseSourceManager
        return PoseSourceManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MAVSDKController", "MAVROSController", "PoseSourceManager"]
