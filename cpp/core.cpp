#include "weaveode/c_api.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace {

bool finite_array(const double *values, std::size_t count) {
    if (values == nullptr) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

class DisjointSet {
public:
    explicit DisjointSet(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }

    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }

    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }
        if (rank_[a] < rank_[b]) {
            std::swap(a, b);
        }
        parent_[b] = a;
        if (rank_[a] == rank_[b]) {
            ++rank_[a];
        }
    }

private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned char> rank_;
};

}  // namespace


extern "C" weaveode_status weaveode_score_bdf1_candidates(
    const weaveode_system_vtable *system,
    void *ctx,
    const double *candidates,
    std::size_t n_candidates,
    std::size_t dimension,
    double t,
    const double *previous,
    double h,
    const double *atol,
    double rtol,
    double *scores) {
    if (system == nullptr || system->abi_version != WEAVEODE_ABI_VERSION || system->rhs == nullptr ||
        n_candidates == 0 || dimension == 0 || scores == nullptr ||
        !finite_array(candidates, n_candidates * dimension) || !finite_array(previous, dimension) ||
        !finite_array(atol, dimension) || !std::isfinite(t) || !std::isfinite(h) || !(h > 0.0) ||
        !std::isfinite(rtol) || rtol < 0.0) {
        return WEAVEODE_INVALID_ARGUMENT;
    }
    std::vector<double> rhs_values(n_candidates * dimension, 0.0);
    for (std::size_t i = 0; i < n_candidates; ++i) {
        const auto status = system->rhs(
            ctx, t + h, candidates + i * dimension, dimension,
            rhs_values.data() + i * dimension);
        if (status != WEAVEODE_OK) {
            return status;
        }
    }
    return weaveode_vector_bdf1_scores_from_rhs(
        candidates, rhs_values.data(), n_candidates, dimension, previous, h, atol, rtol, scores);
}

extern "C" weaveode_status weaveode_wrms_scores(
    const double *residuals,
    std::size_t n_candidates,
    std::size_t dimension,
    const double *scales,
    double *scores) {
    if (n_candidates == 0 || dimension == 0 || scores == nullptr ||
        !finite_array(residuals, n_candidates * dimension) ||
        !finite_array(scales, dimension)) {
        return WEAVEODE_INVALID_ARGUMENT;
    }
    for (std::size_t j = 0; j < dimension; ++j) {
        if (!(scales[j] > 0.0)) {
            return WEAVEODE_INVALID_ARGUMENT;
        }
    }

    for (std::size_t i = 0; i < n_candidates; ++i) {
        long double sum = 0.0L;
        for (std::size_t j = 0; j < dimension; ++j) {
            const long double value = static_cast<long double>(residuals[i * dimension + j]) /
                                      static_cast<long double>(scales[j]);
            sum += value * value;
        }
        scores[i] = std::sqrt(static_cast<double>(sum / static_cast<long double>(dimension)));
    }
    return WEAVEODE_OK;
}

extern "C" weaveode_status weaveode_poly_bdf1_scores(
    const double *candidates,
    std::size_t n_candidates,
    double y_prev,
    double h,
    const double *coeffs,
    std::size_t n_coeffs,
    double atol,
    double rtol,
    double *scores) {
    if (n_candidates == 0 || n_coeffs == 0 || scores == nullptr ||
        !finite_array(candidates, n_candidates) || !finite_array(coeffs, n_coeffs) ||
        !std::isfinite(y_prev) || !std::isfinite(h) || !(h > 0.0) ||
        !std::isfinite(atol) || !std::isfinite(rtol) || atol < 0.0 || rtol < 0.0 ||
        (atol == 0.0 && rtol == 0.0)) {
        return WEAVEODE_INVALID_ARGUMENT;
    }

    for (std::size_t i = 0; i < n_candidates; ++i) {
        const double y = candidates[i];
        double f = 0.0;
        for (std::size_t k = n_coeffs; k-- > 0;) {
            f = std::fma(f, y, coeffs[k]);
        }
        const double residual = y - y_prev - h * f;
        const double scale = atol + rtol * std::max(std::abs(y_prev), std::abs(y));
        if (!(scale > 0.0) || !std::isfinite(residual)) {
            return WEAVEODE_NUMERICAL_FAILURE;
        }
        scores[i] = std::abs(residual) / scale;
    }
    return WEAVEODE_OK;
}


extern "C" weaveode_status weaveode_vector_bdf1_scores_from_rhs(
    const double *candidates,
    const double *rhs_values,
    std::size_t n_candidates,
    std::size_t dimension,
    const double *previous,
    double h,
    const double *atol,
    double rtol,
    double *scores) {
    if (n_candidates == 0 || dimension == 0 || scores == nullptr ||
        !finite_array(candidates, n_candidates * dimension) ||
        !finite_array(rhs_values, n_candidates * dimension) ||
        !finite_array(previous, dimension) || !finite_array(atol, dimension) ||
        !std::isfinite(h) || !(h > 0.0) || !std::isfinite(rtol) || rtol < 0.0) {
        return WEAVEODE_INVALID_ARGUMENT;
    }
    for (std::size_t j = 0; j < dimension; ++j) {
        if (atol[j] < 0.0 || (atol[j] == 0.0 && rtol == 0.0)) {
            return WEAVEODE_INVALID_ARGUMENT;
        }
    }

    for (std::size_t i = 0; i < n_candidates; ++i) {
        long double sum = 0.0L;
        for (std::size_t j = 0; j < dimension; ++j) {
            const std::size_t index = i * dimension + j;
            const double y = candidates[index];
            const double residual = y - previous[j] - h * rhs_values[index];
            const double scale = atol[j] + rtol * std::max(std::abs(previous[j]), std::abs(y));
            if (!(scale > 0.0) || !std::isfinite(residual)) {
                return WEAVEODE_NUMERICAL_FAILURE;
            }
            const long double z = static_cast<long double>(residual) /
                                  static_cast<long double>(scale);
            sum += z * z;
        }
        scores[i] = std::sqrt(static_cast<double>(sum / static_cast<long double>(dimension)));
    }
    return WEAVEODE_OK;
}

extern "C" weaveode_status weaveode_radius_components(
    const double *points,
    std::size_t n_points,
    std::size_t dimension,
    double radius,
    std::int64_t *labels) {
    if (n_points == 0 || dimension == 0 || labels == nullptr ||
        !finite_array(points, n_points * dimension) || !std::isfinite(radius) || radius < 0.0) {
        return WEAVEODE_INVALID_ARGUMENT;
    }

    const long double radius_sq = static_cast<long double>(radius) * radius;
    DisjointSet sets(n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        for (std::size_t j = i + 1; j < n_points; ++j) {
            long double distance_sq = 0.0L;
            for (std::size_t k = 0; k < dimension; ++k) {
                const long double delta = static_cast<long double>(points[i * dimension + k]) -
                                          static_cast<long double>(points[j * dimension + k]);
                distance_sq += delta * delta;
                if (distance_sq > radius_sq) {
                    break;
                }
            }
            if (distance_sq <= radius_sq) {
                sets.unite(i, j);
            }
        }
    }

    std::vector<std::size_t> roots;
    roots.reserve(n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        const std::size_t root = sets.find(i);
        auto position = std::find(roots.begin(), roots.end(), root);
        if (position == roots.end()) {
            roots.push_back(root);
            labels[i] = static_cast<std::int64_t>(roots.size() - 1);
        } else {
            labels[i] = static_cast<std::int64_t>(std::distance(roots.begin(), position));
        }
    }
    return WEAVEODE_OK;
}
