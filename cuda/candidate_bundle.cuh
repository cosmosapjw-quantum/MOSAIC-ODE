#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// Non-owning device view. Allocation and lifetime are controlled by the host-side
// device arena so speculative epochs can reuse storage without cudaMalloc/cudaFree.
struct CandidateBundleDevice {
    double *state;          // [count, dimension]
    double *residual;       // [count, dimension]
    double *score;          // [count]
    int64_t *branch_id;     // [count]
    int64_t *component_id;  // [count]
    int64_t *path_id;       // [count]
    uint8_t *active_mask;   // [count]
    uint8_t *solver_lane;   // [count]
    uint8_t *precision_mode;// [count]
    uint64_t *work_units;   // [count]
    size_t count;
    size_t dimension;
};
