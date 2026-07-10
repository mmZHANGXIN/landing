# Orin Environment Check

- generated_at: `2026-06-07T14:59:50`
- python: `3.9.6`
- platform: `macOS-26.4-arm64-arm-64bit`
- required_failures: `0`
- failed_keys: `none`

| Key | Required | Status | Detail | Fix |
| --- | --- | --- | --- | --- |
| jetpack | False | WARN | /etc/nv_tegra_release missing | Run on Jetson Orin with JetPack 5.x. |
| nvcc | False | WARN | nvcc not found | Install CUDA toolkit from JetPack. |
| ros2_cli | False | WARN | ros2 not found | Install/source ROS2 Galactic. |
| ros_setup | False | WARN | no /opt/ros/{galactic,humble}/setup.bash | Install/source ROS2 Galactic for FAST-LIO. |
| rclpy | False | WARN | import failed: No module named 'rclpy' | Source ROS2 setup.bash before running Python. |
| torch_cuda | False | WARN | import failed: No module named 'torch' | Install Jetson PyTorch with CUDA support. |
| opencv | False | WARN | import failed: No module named 'cv2' | Install opencv-python or system OpenCV. |
| numpy | False | WARN | import failed: No module named 'numpy' | Install numpy compatible with Jetson PyTorch. |
| yaml | False | WARN | import failed: No module named 'yaml' | Install PyYAML. |
| stable_baselines3 | False | WARN | import failed: No module named 'stable_baselines3' | Install stable-baselines3. |
| mavsdk | False | WARN | import failed: No module named 'mavsdk' | Install mavsdk. |
