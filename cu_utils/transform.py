from numba import cuda, float32
import math


def cu_mean_transform(x, y_out):
    res = cuda.shared.array(1, dtype=float32)
    res[0] = 0
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        cuda.atomic.add(res, 0, x[i])
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        y_out[i] = res[0] / len(x)


def cu_max_transform(x, y_out):
    res = cuda.shared.array(1, dtype=float32)
    res[0] = -math.inf
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        cuda.atomic.max(res, 0, x[i])
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        y_out[i] = res[0]


def cu_min_transform(x, y_out):
    res = cuda.shared.array(1, dtype=float32)
    res[0] = math.inf
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        cuda.atomic.min(res, 0, x[i])
    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
        y_out[i] = res[0]


def get_cu_shift_transform(shift_by, null_val=-1):
    def cu_shift_transform(x, y_out):
        for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
            y_out[i] = null_val
            if 0 <= i-shift_by < len(x):
                y_out[i] = x[i-shift_by]
    return cu_shift_transform


def get_cu_rolling_mean_transform(window, null_val=-1):
    def cu_rolling_mean_transform(x, y_out):
        for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
            y_out[i] = 0
            if i >= window-1:
                for j in range(cuda.threadIdx.y, window, cuda.blockDim.y):
                    cuda.atomic.add(y_out, i, x[i-j])
                y_out[i] /= window
            else:
                y_out[i] = null_val
    return cu_rolling_mean_transform


def get_cu_rolling_max_transform(window, null_val=-1):
    def cu_rolling_max_transform(x, y_out):
        for i in range(cuda.threadIdx.x, len(x), cuda.blockDim.x):
            y_out[i] = -math.inf
            if i >= window-1:
                for j in range(cuda.threadIdx.y, window, cuda.blockDim.y):
                    cuda.atomic.max(y_out, i, x[i-j])
            else:
                y_out[i] = null_val
    return cu_rolling_max_transform
