import unittest

import numpy as np

from perception.training_camera_projection import (
    TrainingCameraModel,
    project_training_camera,
    sample_nearest_points_by_camera_rays,
)


class TrainingCameraProjectionTest(unittest.TestCase):
    def setUp(self):
        self.camera = TrainingCameraModel()
        self.bounds = {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0}
        self.semantic = np.ones((64, 64), dtype=np.uint8)

    def project_one(self, point):
        depth, valid, semantic, semantic_valid = project_training_camera(
            np.asarray([point], np.float32), self.semantic, self.bounds, self.camera
        )
        row, col = np.argwhere(valid)[0]
        return row, col, depth[row, col], semantic[row, col], semantic_valid[row, col]

    def test_scaled_intrinsics_and_fov(self):
        self.assertAlmostEqual(self.camera.fx, 77.4468, places=3)
        self.assertAlmostEqual(self.camera.fy, 121.3333, places=3)
        self.assertAlmostEqual(self.camera.horizontal_fov_deg, 79.2, delta=0.2)
        self.assertAlmostEqual(self.camera.vertical_fov_deg, 55.7, delta=0.2)

    def test_center_and_directions(self):
        center = self.project_one([0.0, 0.0, 2.0])
        front = self.project_one([0.3, 0.0, 2.0])
        rear = self.project_one([-0.3, 0.0, 2.0])
        left = self.project_one([0.0, 0.3, 2.0])
        right = self.project_one([0.0, -0.3, 2.0])
        self.assertLess(front[0], center[0])
        self.assertGreater(rear[0], center[0])
        self.assertLess(left[1], center[1])
        self.assertGreater(right[1], center[1])

    def test_ground_extents_match_training_fov(self):
        half_x, half_y = self.camera.ground_half_extents(2.0)
        self.assertAlmostEqual(half_x / 2.0, 0.527, delta=0.003)
        self.assertAlmostEqual(half_y / 2.0, 0.826, delta=0.003)

    def test_shared_z_buffer_keeps_nearest_semantic(self):
        sem = np.full((64, 64), 9, dtype=np.uint8)
        sem[31:33, 31:33] = 1
        points = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        depth, valid, labels, _ = project_training_camera(
            points, sem, self.bounds, self.camera
        )
        row, col = np.argwhere(valid)[0]
        self.assertAlmostEqual(depth[row, col], 1.0)
        self.assertEqual(labels[row, col], 1)

    def test_ray_sampling_keeps_nearest_and_bounds_output(self):
        points = np.array([
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 1.0],  # same centre ray, nearer
            [0.0, 100.0, 2.0],  # outside the camera frustum
        ], dtype=np.float32)
        sampled, stats = sample_nearest_points_by_camera_rays(
            points, self.camera, ray_width=64, ray_height=64,
        )
        self.assertEqual(len(sampled), 1)
        self.assertAlmostEqual(float(sampled[0, 2]), 1.0)
        self.assertEqual(stats["frustum_points"], 2)
        self.assertLessEqual(stats["ray_points"], 64 * 64)

    def test_ray_sampling_never_invents_empty_rays(self):
        sampled, stats = sample_nearest_points_by_camera_rays(
            np.empty((0, 3), dtype=np.float32), self.camera,
            ray_width=64, ray_height=64,
        )
        self.assertEqual(sampled.shape, (0, 3))
        self.assertEqual(stats["ray_points"], 0)


if __name__ == "__main__":
    unittest.main()
