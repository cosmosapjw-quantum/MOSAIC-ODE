#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

// Apply a bounded low-rank online proposal: delta = bound * scale * tanh(U(V^T x)).
extern "C" __global__ void weaveode_low_rank_proposal_kernel(
    const double *features,
    size_t feature_dimension,
    const double *encoder,  // [rank, feature_dimension]
    const double *decoder,  // [state_dimension, rank]
    size_t state_dimension,
    size_t rank,
    const double *scale,
    double bound,
    double *proposal) {
    extern __shared__ double hidden[];
    if (threadIdx.x < rank) {
        double value = 0.0;
        for (size_t j = 0; j < feature_dimension; ++j) {
            value += encoder[threadIdx.x * feature_dimension + j] * features[j];
        }
        hidden[threadIdx.x] = tanh(value);
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < state_dimension; i += blockDim.x) {
        double raw = 0.0;
        for (size_t k = 0; k < rank; ++k) {
            raw += decoder[i * rank + k] * hidden[k];
        }
        proposal[i] = bound * scale[i] * tanh(raw);
    }
}

extern "C" cudaError_t weaveode_launch_low_rank_proposal(
    const double *features,
    size_t feature_dimension,
    const double *encoder,
    const double *decoder,
    size_t state_dimension,
    size_t rank,
    const double *scale,
    double bound,
    double *proposal,
    cudaStream_t stream) {
    if (features == nullptr || encoder == nullptr || decoder == nullptr || scale == nullptr ||
        proposal == nullptr || feature_dimension == 0 || state_dimension == 0 || rank == 0 ||
        rank > 256 || !isfinite(bound) || !(bound > 0.0)) {
        return cudaErrorInvalidValue;
    }
    unsigned int threads = 64;
    if (rank > threads) threads = 128;
    if (rank > threads) threads = 256;
    weaveode_low_rank_proposal_kernel<<<1, threads, rank * sizeof(double), stream>>>(
        features, feature_dimension, encoder, decoder, state_dimension, rank, scale, bound, proposal);
    return cudaGetLastError();
}
