#include "ephs/gemv.hpp"

#include <cstddef>
#include <immintrin.h>

namespace ephs {
namespace {

float hsum256(const __m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(lo, lo);
  lo = _mm_add_ss(lo, sh);
  return _mm_cvtss_f32(lo);
}

} // namespace

float dot_f32_avx2(const float* a, const float* b, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  float sum = hsum256(acc);
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

float dot_i8_f32_avx2(const std::int8_t* q, const float* h, int n) {
  __m256 acc = _mm256_setzero_ps();
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i));
    const __m128i hi_bytes = _mm_srli_si128(bytes, 8);
    const __m256 flo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
    const __m256 fhi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi_bytes));
    acc = _mm256_fmadd_ps(flo, _mm256_loadu_ps(h + i), acc);
    acc = _mm256_fmadd_ps(fhi, _mm256_loadu_ps(h + i + 8), acc);
  }
  float sum = hsum256(acc);
  for (; i < n; ++i) {
    sum += static_cast<float>(q[i]) * h[i];
  }
  return sum;
}

} // namespace ephs
