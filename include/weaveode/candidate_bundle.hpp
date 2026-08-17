#ifndef WEAVEODE_CANDIDATE_BUNDLE_HPP
#define WEAVEODE_CANDIDATE_BUNDLE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace weaveode {

class CandidateBundle {
public:
    CandidateBundle(std::size_t count, std::size_t dimension)
        : count_(count), dimension_(dimension), states_(count * dimension, 0.0),
          residuals_(count * dimension, 0.0), scores_(count, 0.0), branch_id_(count, -1),
          component_id_(count, -1), path_id_(count, -1), active_mask_(count, 1),
          solver_lane_(count, 0), precision_mode_(count, 0), work_units_(count, 0) {
        if (count == 0 || dimension == 0) {
            throw std::invalid_argument("candidate bundle dimensions must be positive");
        }
    }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }
    [[nodiscard]] std::vector<double> &states() noexcept { return states_; }
    [[nodiscard]] const std::vector<double> &states() const noexcept { return states_; }
    [[nodiscard]] std::vector<double> &residuals() noexcept { return residuals_; }
    [[nodiscard]] const std::vector<double> &residuals() const noexcept { return residuals_; }
    [[nodiscard]] std::vector<double> &scores() noexcept { return scores_; }
    [[nodiscard]] const std::vector<double> &scores() const noexcept { return scores_; }
    [[nodiscard]] std::vector<std::int64_t> &branch_id() noexcept { return branch_id_; }
    [[nodiscard]] std::vector<std::int64_t> &component_id() noexcept { return component_id_; }
    [[nodiscard]] std::vector<std::int64_t> &path_id() noexcept { return path_id_; }
    [[nodiscard]] std::vector<std::uint8_t> &active_mask() noexcept { return active_mask_; }
    [[nodiscard]] std::vector<std::uint8_t> &solver_lane() noexcept { return solver_lane_; }
    [[nodiscard]] std::vector<std::uint8_t> &precision_mode() noexcept { return precision_mode_; }
    [[nodiscard]] std::vector<std::uint64_t> &work_units() noexcept { return work_units_; }
private:
    std::size_t count_; std::size_t dimension_;
    std::vector<double> states_, residuals_, scores_;
    std::vector<std::int64_t> branch_id_, component_id_, path_id_;
    std::vector<std::uint8_t> active_mask_, solver_lane_, precision_mode_;
    std::vector<std::uint64_t> work_units_;
};

}  // namespace weaveode
#endif
