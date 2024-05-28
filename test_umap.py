import numpy as np
import unittest

from umap_run import computeUMAP


class TestComputeUMAP(unittest.TestCase):

    def test_1d_input(self):
        data = np.random.rand(100, 50)
        result = computeUMAP(data, n_components=2)
        self.assertEqual(result.shape, (100, 2), "Output shape should be (N, 2)")

    def test_2d_input(self):
        data = np.random.rand(100, 10, 10)
        result = computeUMAP(data, n_components=2)
        self.assertEqual(result.shape, (100, 2), "Output shape should be (N, 2)")

    def test_n_components_3(self):
        data = np.random.rand(100, 50)
        result = computeUMAP(data, n_components=3)
        self.assertEqual(result.shape, (100, 3), "Output shape should be (N, 3) when n_components=3")

    def test_min_dist(self):
        data = np.random.rand(100, 50)
        result1 = computeUMAP(data, min_dist=0.1)
        result2 = computeUMAP(data, min_dist=0.5)
        self.assertNotEqual(np.sum(result1), np.sum(result2), "UMAP results should differ for different min_dist values")

    def test_random_state(self):
        data = np.random.rand(100, 50)
        result1 = computeUMAP(data, random_state=42)
        result2 = computeUMAP(data, random_state=42)
        np.testing.assert_array_almost_equal(result1, result2, err_msg="Results should be the same for the same random_state")

    def test_different_random_state(self):
        data = np.random.rand(100, 50)
        result1 = computeUMAP(data, random_state=42)
        result2 = computeUMAP(data, random_state=24)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_almost_equal(result1, result2, err_msg="Results should differ for different random_states")


if __name__ == '__main__':
    unittest.main()
