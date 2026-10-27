#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace ephs::test {

inline double log_gamma(double z) {
  // Lanczos, g=7, n=9. Accurate to well under 1e-12 on z > 0.5, which covers chi-square df/2.
  static const double c[9] = {0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
                              771.32342877765313,   -176.61502916214059,   12.507343278686905,
                              -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
  if (z < 0.5) {
    return std::log(3.14159265358979323846 / std::sin(3.14159265358979323846 * z)) -
           log_gamma(1.0 - z);
  }
  z -= 1.0;
  double x = c[0];
  for (int i = 1; i < 9; ++i) {
    x += c[i] / (z + static_cast<double>(i));
  }
  const double t = z + 7.5;
  return 0.5 * std::log(2.0 * 3.14159265358979323846) + (z + 0.5) * std::log(t) - t + std::log(x);
}

// Regularized upper incomplete gamma Q(s, x) = Gamma(s, x) / Gamma(s).
inline double gamma_q(double s, double x) {
  if (x < 0.0 || s <= 0.0) {
    return 1.0;
  }
  if (x == 0.0) {
    return 1.0;
  }
  if (x < s + 1.0) {
    // Series for P, then Q = 1 - P.
    double term = 1.0 / s;
    double sum = term;
    for (int n = 1; n < 400; ++n) {
      term *= x / (s + static_cast<double>(n));
      sum += term;
      if (std::fabs(term) < std::fabs(sum) * 1e-15) {
        break;
      }
    }
    const double p = sum * std::exp(-x + s * std::log(x) - log_gamma(s));
    return std::clamp(1.0 - p, 0.0, 1.0);
  }
  // Continued fraction for Q.
  double b = x + 1.0 - s;
  double c = 1e30;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= 400; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - s);
    b += 2.0;
    d = an * d + b;
    if (std::fabs(d) < 1e-30) {
      d = 1e-30;
    }
    c = b + an / c;
    if (std::fabs(c) < 1e-30) {
      c = 1e-30;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-12) {
      break;
    }
  }
  const double q = std::exp(-x + s * std::log(x) - log_gamma(s)) * h;
  return std::clamp(q, 0.0, 1.0);
}

// Survival function P(ChiSquare(df) > x).
inline double chi2_sf(double x, int df) {
  if (df < 1) {
    return 1.0;
  }
  if (x <= 0.0) {
    return 1.0;
  }
  return gamma_q(0.5 * static_cast<double>(df), 0.5 * x);
}

// Asymptotic survival function of the Kolmogorov-Smirnov statistic D_n.
inline double ks_sf(double d, int n) {
  if (n < 1 || !(d > 0.0)) {
    return 1.0;
  }
  const double z = std::sqrt(static_cast<double>(n)) * d;
  double sum = 0.0;
  for (int k = 1; k <= 200; ++k) {
    const double term = std::exp(-2.0 * static_cast<double>(k) * static_cast<double>(k) * z * z);
    sum += (k % 2 == 1) ? term : -term;
    if (term < 1e-16) {
      break;
    }
  }
  return std::clamp(2.0 * sum, 0.0, 1.0);
}

inline double ks_uniform_statistic(const std::vector<double>& samples) {
  std::vector<double> x = samples;
  std::sort(x.begin(), x.end());
  const double n = static_cast<double>(x.size());
  double d = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double emp_hi = static_cast<double>(i + 1) / n;
    const double emp_lo = static_cast<double>(i) / n;
    d = std::max(d, std::fabs(emp_hi - x[i]));
    d = std::max(d, std::fabs(x[i] - emp_lo));
  }
  return d;
}

} // namespace ephs::test
