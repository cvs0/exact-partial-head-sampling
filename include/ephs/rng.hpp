#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>

namespace ephs {

// Deterministic RNG for Gumbel(0, 1) draws and uniform indices.
// uniform_open() lies in (0, 1), so logarithms stay finite.
class Rng {
public:
  explicit Rng(std::uint64_t seed) : eng_(seed) {}

  double uniform_open() {
    for (;;) {
      double u = open_(eng_);
      if (u > 0.0 && u < 1.0) {
        return u;
      }
    }
  }

  // Unit exponential, mean 1.
  double exponential() { return -std::log(uniform_open()); }

  // Standard Gumbel(0, 1), CDF exp(-exp(-x)).
  double gumbel() { return -std::log(exponential()); }

  // Unbiased integer in [0, n).
  int uniform_index(int n) {
    if (n <= 0) {
      throw std::invalid_argument("uniform_index requires n > 0");
    }
    std::uniform_int_distribution<int> dist(0, n - 1);
    return dist(eng_);
  }

private:
  std::mt19937_64 eng_;
  std::uniform_real_distribution<double> open_{0.0, 1.0};
};

// Gumbel conditioned on being strictly below `ceiling`.
// If E ~ Exp(1) then -log(exp(-ceiling) + E) has the law of Gumbel(0, 1) | G < ceiling.
inline double gumbel_truncated_below(double ceiling, Rng& rng) {
  const double e = rng.exponential();
  const double log_e = std::log(e);
  const double neg_c = -ceiling;
  const double m = std::max(neg_c, log_e);
  const double lse = m + std::log(std::exp(neg_c - m) + std::exp(log_e - m));
  double g = -lse;
  if (!(g < ceiling)) {
    g = std::nextafter(ceiling, -std::numeric_limits<double>::infinity());
  }
  return g;
}

} // namespace ephs
