// Optional CUDA C++ fused scorer for scalar polynomial BDF1 candidates.
// This source is intentionally isolated from the CPU build. It was reviewed but
// cannot be compiled or benchmarked in the current CPU-only environment.

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

extern "C" __global__ void weaveode_poly_bdf1_score_kernel(
    const double *candidates,
    size_t count,
    double previous,
    double h,
    const double *coefficients,
    size_t coefficient_count,
    double atol,
    double rtol,
    double *scores) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const double y = candidates[index];
    double rhs = 0.0;
    for (size_t k = coefficient_count; k-- > 0;) {
        rhs = fma(rhs, y, coefficients[k]);
    }
    const double residual = y - previous - h * rhs;
    const double scale = atol + rtol * fmax(fabs(previous), fabs(y));
    scores[index] = (isfinite(residual) && isfinite(scale) && scale > 0.0)
        ? fabs(residual) / scale
        : INFINITY;
}

extern "C" cudaError_t weaveode_launch_poly_bdf1_scores(
    const double *candidates,
    size_t count,
    double previous,
    double h,
    const double *coefficients,
    size_t coefficient_count,
    double atol,
    double rtol,
    double *scores,
    cudaStream_t stream) {
    if (count == 0 || candidates == nullptr || coefficients == nullptr || scores == nullptr ||
        !isfinite(previous) || !isfinite(h) || !(h > 0.0) ||
        !isfinite(atol) || atol < 0.0 || !isfinite(rtol) || rtol < 0.0 ||
        (atol == 0.0 && rtol == 0.0)) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>((count + block_size - 1) / block_size);
    weaveode_poly_bdf1_score_kernel<<<grid_size, block_size, 0, stream>>>(
        candidates, count, previous, h, coefficients, coefficient_count, atol, rtol, scores);
    return cudaGetLastError();
}
