#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

double linear_quantile(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double position = static_cast<double>(sorted.size() - 1) * q;
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

}  // namespace

extern "C" int stage3c_bootstrap_timing(
    const double* values,
    const std::uint8_t* available,
    const std::int64_t* indices,
    std::size_t n_bootstrap,
    std::size_t n_realizations,
    double* output) {
    if (values == nullptr || available == nullptr || indices == nullptr ||
        output == nullptr || n_bootstrap == 0 || n_realizations == 0) {
        return 1;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> sample;
    std::vector<double> absolute;
    std::vector<double> deviations;
    sample.reserve(n_realizations);
    absolute.reserve(n_realizations);
    deviations.reserve(n_realizations);

    for (std::size_t replicate = 0; replicate < n_bootstrap; ++replicate) {
        sample.clear();
        const std::size_t base = replicate * n_realizations;
        for (std::size_t draw = 0; draw < n_realizations; ++draw) {
            const std::int64_t id = indices[base + draw];
            if (id < 0 || static_cast<std::size_t>(id) >= n_realizations) {
                return 2;
            }
            if (available[id] != 0) {
                sample.push_back(values[id]);
            }
        }

        double* row = output + replicate * 6;
        if (sample.empty()) {
            std::fill(row, row + 6, nan);
            continue;
        }

        std::sort(sample.begin(), sample.end());
        const double median = linear_quantile(sample, 0.5);

        absolute.clear();
        deviations.clear();
        for (double value : sample) {
            absolute.push_back(std::abs(value));
            deviations.push_back(std::abs(value - median));
        }
        std::sort(absolute.begin(), absolute.end());
        std::sort(deviations.begin(), deviations.end());

        row[0] = median;
        row[1] = linear_quantile(absolute, 0.5);
        row[2] = linear_quantile(deviations, 0.5);
        row[3] = linear_quantile(sample, 0.75) - linear_quantile(sample, 0.25);
        row[4] = linear_quantile(sample, 0.10);
        row[5] = linear_quantile(sample, 0.90);
    }
    return 0;
}
