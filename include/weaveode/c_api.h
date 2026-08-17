#ifndef WEAVEODE_C_API_H
#define WEAVEODE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef WEAVEODE_BUILD_SHARED
#    define WEAVEODE_API __declspec(dllexport)
#  else
#    define WEAVEODE_API __declspec(dllimport)
#  endif
#else
#  define WEAVEODE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum weaveode_status {
    WEAVEODE_OK = 0,
    WEAVEODE_INVALID_ARGUMENT = 1,
    WEAVEODE_NUMERICAL_FAILURE = 2,
    WEAVEODE_INTERNAL_ERROR = 3
} weaveode_status;

#define WEAVEODE_ABI_VERSION 1u

typedef weaveode_status (*weaveode_rhs_callback)(
    void *ctx, double t, const double *y, size_t dimension, double *out);

typedef struct weaveode_system_vtable {
    uint32_t abi_version;
    weaveode_rhs_callback rhs;
    void (*destroy)(void *ctx);
} weaveode_system_vtable;

WEAVEODE_API weaveode_status weaveode_score_bdf1_candidates(
    const weaveode_system_vtable *system, void *ctx, const double *candidates,
    size_t n_candidates, size_t dimension, double t, const double *previous,
    double h, const double *atol, double rtol, double *scores);

WEAVEODE_API weaveode_status weaveode_wrms_scores(
    const double *residuals, size_t n_candidates, size_t dimension,
    const double *scales, double *scores);

WEAVEODE_API weaveode_status weaveode_poly_bdf1_scores(
    const double *candidates, size_t n_candidates, double y_prev, double h,
    const double *coeffs, size_t n_coeffs, double atol, double rtol, double *scores);

WEAVEODE_API weaveode_status weaveode_vector_bdf1_scores_from_rhs(
    const double *candidates, const double *rhs_values, size_t n_candidates,
    size_t dimension, const double *previous, double h, const double *atol,
    double rtol, double *scores);

WEAVEODE_API weaveode_status weaveode_radius_components(
    const double *points, size_t n_points, size_t dimension, double radius,
    int64_t *labels);

#ifdef __cplusplus
}
#endif

#endif
