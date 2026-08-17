#include <cuda_runtime.h>
#include <stddef.h>

// Bootstrap multi-vector primitives for topology-guided shared-linearization groups.
// The production block GMRES/Arnoldi implementation may delegate SpMM to cuSPARSE;
// this source establishes the device data contract and fused column reductions.
extern "C" __global__ void weaveode_axpy_multivector_kernel(
    double alpha,
    const double *x,
    double *y,
    size_t rows,
    size_t columns) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * columns;
    if (index < total) {
        y[index] = fma(alpha, x[index], y[index]);
    }
}

extern "C" cudaError_t weaveode_launch_axpy_multivector(
    double alpha,
    const double *x,
    double *y,
    size_t rows,
    size_t columns,
    cudaStream_t stream) {
    if (x == nullptr || y == nullptr || rows == 0 || columns == 0) {
        return cudaErrorInvalidValue;
    }
    const size_t total = rows * columns;
    constexpr unsigned int block = 256;
    const unsigned int grid = (unsigned int)((total + block - 1) / block);
    weaveode_axpy_multivector_kernel<<<grid, block, 0, stream>>>(alpha, x, y, rows, columns);
    return cudaGetLastError();
}
