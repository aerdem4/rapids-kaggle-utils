from numba import cuda
import cu_utils.transform as cutr
import numpy as np

SIZE = 128
np.random.seed(0)


def test_cu_mean_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cutr.cu_mean_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.mean(arr))
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_min_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cutr.cu_min_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.min(arr))
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_max_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)

    cuda.jit(cutr.cu_max_transform)(arr, res)

    np.testing.assert_almost_equal(res, np.max(arr))
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_shift_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)
    shift = 3
    null_val = -1

    cuda.jit(cutr.get_cu_shift_transform(shift_by=shift, null_val=null_val))(arr, res)

    np.testing.assert_equal(res[shift:], arr[:-shift])
    np.testing.assert_equal(res[:shift], null_val)
    np.testing.assert_equal(res.shape[0], SIZE)

    res = np.zeros(SIZE)
    shift = -4

    cuda.jit(cutr.get_cu_shift_transform(shift_by=shift, null_val=null_val))(arr, res)

    np.testing.assert_equal(res[:shift], arr[-shift:])
    np.testing.assert_equal(res[shift:], null_val)
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_rolling_mean_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)
    window = 3
    null_val = -1

    cuda.jit(cutr.get_cu_rolling_mean_transform(window=window, null_val=null_val))(arr, res)

    expected = arr.copy()
    for i in range(1, window):
        expected += np.roll(arr, i)
    expected /= window

    np.testing.assert_equal(res[window-1:], expected[window-1:])
    np.testing.assert_equal(res[:window-1], null_val)
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_rolling_max_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)
    window = 5
    null_val = -1

    cuda.jit(cutr.get_cu_rolling_max_transform(window=window, null_val=null_val))(arr, res)

    expected = arr.copy()
    for i in range(1, window):
        expected = np.maximum(np.roll(arr, i), expected)

    np.testing.assert_equal(res[window-1:], expected[window-1:])
    np.testing.assert_equal(res[:window-1], null_val)
    np.testing.assert_equal(res.shape[0], SIZE)


def test_cu_rolling_min_transform():
    arr = np.random.rand(SIZE)
    res = np.zeros(SIZE)
    window = 5
    null_val = -1

    cuda.jit(cutr.get_cu_rolling_min_transform(window=window, null_val=null_val))(arr, res)

    expected = arr.copy()
    for i in range(1, window):
        expected = np.minimum(np.roll(arr, i), expected)

    np.testing.assert_equal(res[window-1:], expected[window-1:])
    np.testing.assert_equal(res[:window-1], null_val)
    np.testing.assert_equal(res.shape[0], SIZE)
