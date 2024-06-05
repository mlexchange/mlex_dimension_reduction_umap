import numpy as np
import pytest

from src.dim_reduction import compute_umap


def test_1d_input():
    data = np.random.rand(100, 50)
    result = compute_umap(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"


def test_2d_input():
    data = np.random.rand(100, 10, 10)
    result = compute_umap(data, n_components=2)
    assert result.shape == (100, 2), "Output shape should be (N, 2)"


def test_n_components_3():
    data = np.random.rand(100, 50)
    result = compute_umap(data, n_components=3)
    assert result.shape == (100, 3), "Output shape should be (N, 3) when n_components=3"


def test_min_dist():
    data = np.random.rand(100, 50)
    result1 = compute_umap(data, min_dist=0.1)
    result2 = compute_umap(data, min_dist=0.5)
    assert np.sum(result1) != np.sum(
        result2
    ), "UMAP results should differ for different min_dist values"


def test_random_state():
    data = np.random.rand(100, 50)
    result1 = compute_umap(data, random_state=42)
    result2 = compute_umap(data, random_state=42)
    np.testing.assert_array_almost_equal(
        result1, result2, err_msg="Results should be the same for the same random_state"
    )


def test_different_random_state():
    data = np.random.rand(100, 50)
    result1 = compute_umap(data, random_state=42)
    result2 = compute_umap(data, random_state=24)
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(
            result1,
            result2,
            err_msg="Results should differ for different random_states",
        )
