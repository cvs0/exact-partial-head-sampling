#include "ephs/gemv.hpp"

namespace ephs {

float dot_f32_scalar(const float* a, const float* b, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(acc);
}

float dot_i8_f32_scalar(const std::int8_t* q, const float* h, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += static_cast<double>(q[i]) * static_cast<double>(h[i]);
  }
  return static_cast<float>(acc);
}

} // namespace ephs
