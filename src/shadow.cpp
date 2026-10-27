#include "ephs/shadow.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ephs {
namespace {

float round_up(double x) {
  if (!(x > 0.0)) {
    return 0.f;
  }
  float f = static_cast<float>(x);
  if (static_cast<double>(f) < x) {
    f = std::nextafter(f, std::numeric_limits<float>::infinity());
  }
  return f;
}

std::int8_t quantize_symmetric(float value, float scale) {
  float r = std::round(value / scale);
  if (r > 127.f) {
    r = 127.f;
  } else if (r < -127.f) {
    r = -127.f;
  }
  return static_cast<std::int8_t>(r);
}

} // namespace

void hidden_group_norms(const float* h, std::int32_t hidden, std::int32_t groups, float* out) {
  if (groups < 1 || hidden % groups != 0) {
    throw std::invalid_argument("hidden must be divisible by groups");
  }
  const int width = hidden / groups;
  for (int g = 0; g < groups; ++g) {
    const float* x = h + static_cast<std::size_t>(g) * static_cast<std::size_t>(width);
    double ss = 0.0;
    for (int j = 0; j < width; ++j) {
      const double v = x[j];
      ss += v * v;
    }
    out[g] = round_up(std::sqrt(ss));
  }
}

Shadow build_shadow(const Head& head, std::int32_t groups) {
  if (head.vocab < 1 || head.hidden < 1) {
    throw std::invalid_argument("head is empty");
  }
  if (static_cast<std::size_t>(head.vocab) * static_cast<std::size_t>(head.hidden) !=
      head.w.size()) {
    throw std::invalid_argument("head weight size does not match vocab * hidden");
  }
  if (groups < 1 || head.hidden % groups != 0) {
    throw std::invalid_argument("hidden must be divisible by groups");
  }
  Shadow shadow;
  shadow.vocab = head.vocab;
  shadow.hidden = head.hidden;
  shadow.groups = groups;
  const std::size_t rows = static_cast<std::size_t>(head.vocab);
  const std::size_t hidden = static_cast<std::size_t>(head.hidden);
  const int width = head.hidden / groups;
  shadow.codes.resize(rows * hidden);
  shadow.scale.resize(rows);
  shadow.group_norm.resize(rows * static_cast<std::size_t>(groups));

  for (std::int32_t v = 0; v < head.vocab; ++v) {
    const float* row = head.row(v);
    float max_abs = 0.f;
    for (int j = 0; j < head.hidden; ++j) {
      max_abs = std::max(max_abs, std::fabs(row[j]));
    }
    float scale = max_abs / 127.f;
    if (!(scale > 0.f)) {
      scale = 1e-12f;
    }
    shadow.scale[static_cast<std::size_t>(v)] = scale;
    std::int8_t* codes = shadow.codes.data() + static_cast<std::size_t>(v) * hidden;
    for (int j = 0; j < head.hidden; ++j) {
      codes[j] = quantize_symmetric(row[j], scale);
    }
    float* norms =
        shadow.group_norm.data() + static_cast<std::size_t>(v) * static_cast<std::size_t>(groups);
    for (int g = 0; g < groups; ++g) {
      double ss = 0.0;
      const int begin = g * width;
      for (int j = 0; j < width; ++j) {
        const double recon = static_cast<double>(codes[begin + j]) * static_cast<double>(scale);
        const double err = static_cast<double>(row[begin + j]) - recon;
        ss += err * err;
      }
      norms[g] = round_up(std::sqrt(ss));
    }
  }
  return shadow;
}

} // namespace ephs
