from numba import cuda
from cu_utils.transform import cu_mean_transform, cu_min_transform, cu_max_transform
import numpy as np

SIZE = 128
np.random.seed(0)


def test_cu_mean_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cu_mean_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.mean(arr))
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_min_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cu_min_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.min(arr))
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_max_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cu_max_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.max(arr))
    np.testing.assert_equal(res.shape[0], SIZE)
