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
