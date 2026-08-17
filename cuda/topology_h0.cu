#include "candidate_bundle.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __global__ void weaveode_init_h0_labels(CandidateBundleDevice bundle) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bundle.count) {
        bundle.component_id[i] = bundle.active_mask[i] ? (int64_t)i : -1;
    }
}

// Label-propagation H0 primitive. This is intentionally simple and deterministic;
// a CUB/CCCL edge-sort + union-find implementation is the performance follow-up.
extern "C" __global__ void weaveode_radius_h0_relax(
    CandidateBundleDevice bundle,
    double radius,
    int *changed) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bundle.count || bundle.active_mask[i] == 0) {
        return;
    }
    int64_t best = bundle.component_id[i];
    const double radius_sq = radius * radius;
    for (size_t j = 0; j < bundle.count; ++j) {
        if (j == i || bundle.active_mask[j] == 0) {
            continue;
        }
        double distance_sq = 0.0;
        for (size_t k = 0; k < bundle.dimension; ++k) {
            const double delta = bundle.state[i * bundle.dimension + k] -
                                 bundle.state[j * bundle.dimension + k];
            distance_sq += delta * delta;
            if (distance_sq > radius_sq) {
                break;
            }
        }
        if (distance_sq <= radius_sq) {
            const int64_t candidate_label = bundle.component_id[j];
            if (candidate_label >= 0 && candidate_label < best) {
                best = candidate_label;
            }
        }
    }
    if (best < bundle.component_id[i]) {
        atomicMin((unsigned long long *)&bundle.component_id[i], (unsigned long long)best);
        atomicExch(changed, 1);
    }
}

extern "C" cudaError_t weaveode_launch_radius_h0(
    CandidateBundleDevice bundle,
    double radius,
    int *device_changed,
    unsigned int max_passes,
    cudaStream_t stream) {
    if (!isfinite(radius) || radius < 0.0 || device_changed == nullptr ||
        bundle.count == 0 || bundle.dimension == 0 || bundle.state == nullptr ||
        bundle.component_id == nullptr || bundle.active_mask == nullptr) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned int block = 128;
    const unsigned int grid = (unsigned int)((bundle.count + block - 1) / block);
    weaveode_init_h0_labels<<<grid, block, 0, stream>>>(bundle);
    for (unsigned int pass = 0; pass < max_passes; ++pass) {
        cudaMemsetAsync(device_changed, 0, sizeof(int), stream);
        weaveode_radius_h0_relax<<<grid, block, 0, stream>>>(bundle, radius, device_changed);
    }
    return cudaGetLastError();
}
