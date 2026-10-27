#include "ephs/gemv.hpp"

#include <cmath>
#include <immintrin.h>
#include <stdexcept>

namespace ephs {
namespace {

Backend g_backend = Backend::Auto;

float dot_f32_dispatch(const float* a, const float* b, int n) {
  if (active_backend() == Backend::Avx2) {
    return dot_f32_avx2(a, b, n);
  }
  return dot_f32_scalar(a, b, n);
}

float dot_i8_dispatch(const std::int8_t* q, const float* h, int n) {
  if (active_backend() == Backend::Avx2) {
    return dot_i8_f32_avx2(q, h, n);
  }
  return dot_i8_f32_scalar(q, h, n);
}

} // namespace

bool cpu_supports_avx2() {
#if defined(__GNUC__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

void set_backend(Backend backend) {
  if (backend == Backend::Avx2 && !cpu_supports_avx2()) {
    throw std::runtime_error("AVX2 backend requested but this CPU does not advertise AVX2");
  }
  g_backend = backend;
}

Backend active_backend() {
  if (g_backend == Backend::Auto) {
    return cpu_supports_avx2() ? Backend::Avx2 : Backend::Scalar;
  }
  return g_backend;
}

float dot_f32(const float* a, const float* b, int n) {
  return dot_f32_dispatch(a, b, n);
}

float dot_i8_f32(const std::int8_t* q, const float* h, int n) {
  return dot_i8_dispatch(q, h, n);
}

void compute_upper(const std::int8_t* codes, const float* scale, const float* group_norm,
                   const float* h, const float* h_group_norm, int vocab, int hidden, int groups,
                   float safety, float* upper) {
  const bool avx = active_backend() == Backend::Avx2;
  for (int v = 0; v < vocab; ++v) {
    const std::int8_t* row = codes + static_cast<std::size_t>(v) * static_cast<std::size_t>(hidden);
    if (avx && v + 1 < vocab) {
      const std::int8_t* next =
          codes + static_cast<std::size_t>(v + 1) * static_cast<std::size_t>(hidden);
      _mm_prefetch(reinterpret_cast<const char*>(next), _MM_HINT_T0);
    }
    const float dot = avx ? dot_i8_f32_avx2(row, h, hidden) : dot_i8_f32_scalar(row, h, hidden);
    const float* delta =
        group_norm + static_cast<std::size_t>(v) * static_cast<std::size_t>(groups);
    float corr = 0.f;
    for (int g = 0; g < groups; ++g) {
      corr += delta[g] * h_group_norm[g];
    }
    upper[v] = scale[v] * dot + corr + safety;
  }
}

void dense_logits(const float* w, const float* h, int vocab, int hidden, float* logits) {
  const bool avx = active_backend() == Backend::Avx2;
  for (int v = 0; v < vocab; ++v) {
    const float* row = w + static_cast<std::size_t>(v) * static_cast<std::size_t>(hidden);
    if (avx && v + 1 < vocab) {
      const float* next = w + static_cast<std::size_t>(v + 1) * static_cast<std::size_t>(hidden);
      _mm_prefetch(reinterpret_cast<const char*>(next), _MM_HINT_T0);
    }
    logits[v] = avx ? dot_f32_avx2(row, h, hidden) : dot_f32_scalar(row, h, hidden);
  }
}

} // namespace ephs
