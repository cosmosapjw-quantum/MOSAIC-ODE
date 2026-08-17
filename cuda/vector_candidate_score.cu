#include "candidate_bundle.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

extern "C" __global__ void weaveode_vector_bdf1_score_from_rhs_kernel(
    CandidateBundleDevice bundle,
    const double *rhs_values,
    const double *previous,
    double h,
    const double *atol,
    double rtol) {
    const size_t candidate = blockIdx.x;
    if (candidate >= bundle.count || bundle.active_mask[candidate] == 0) {
        return;
    }
    double sum = 0.0;
    for (size_t j = threadIdx.x; j < bundle.dimension; j += blockDim.x) {
        const size_t index = candidate * bundle.dimension + j;
        const double y = bundle.state[index];
        const double r = y - previous[j] - h * rhs_values[index];
        bundle.residual[index] = r;
        const double scale = atol[j] + rtol * fmax(fabs(previous[j]), fabs(y));
        if (!(scale > 0.0) || !isfinite(scale) || !isfinite(r)) {
            sum = INFINITY;
        } else if (isfinite(sum)) {
            const double z = r / scale;
            sum += z * z;
        }
    }
    extern __shared__ double scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        bundle.score[candidate] = sqrt(scratch[0] / (double)bundle.dimension);
        if (bundle.work_units != nullptr) {
            bundle.work_units[candidate] += bundle.dimension;
        }
    }
}

extern "C" cudaError_t weaveode_launch_vector_bdf1_scores_from_rhs(
    CandidateBundleDevice bundle,
    const double *rhs_values,
    const double *previous,
    double h,
    const double *atol,
    double rtol,
    cudaStream_t stream) {
    if (bundle.count == 0 || bundle.dimension == 0 || bundle.state == nullptr ||
        bundle.residual == nullptr || bundle.score == nullptr || bundle.active_mask == nullptr ||
        rhs_values == nullptr || previous == nullptr || atol == nullptr ||
        !isfinite(h) || !(h > 0.0) || !isfinite(rtol) || rtol < 0.0) {
        return cudaErrorInvalidValue;
    }
    unsigned int threads = 1;
    while (threads < bundle.dimension && threads < 256) {
        threads <<= 1;
    }
    weaveode_vector_bdf1_score_from_rhs_kernel<<<
        (unsigned int)bundle.count, threads, threads * sizeof(double), stream>>>(
        bundle, rhs_values, previous, h, atol, rtol);
    return cudaGetLastError();
}
