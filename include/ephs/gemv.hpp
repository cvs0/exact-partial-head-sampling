#pragma once

#include "ephs/types.hpp"

#include <cstdint>

namespace ephs {

bool cpu_supports_avx2();
void set_backend(Backend backend);
Backend active_backend();

float dot_f32(const float* a, const float* b, int n);
float dot_i8_f32(const std::int8_t* q, const float* h, int n);

// Dispatched implementations, exposed so tests can compare them directly.
float dot_f32_scalar(const float* a, const float* b, int n);
float dot_f32_avx2(const float* a, const float* b, int n);
float dot_i8_f32_scalar(const std::int8_t* q, const float* h, int n);
float dot_i8_f32_avx2(const std::int8_t* q, const float* h, int n);

// U[v] = scale[v] * <q_v, h> + <delta_v, rho> + safety
// rho is the per-group L2 norm of h, length `groups`.
// delta layout is [vocab, groups].
void compute_upper(const std::int8_t* codes, const float* scale, const float* group_norm,
                   const float* h, const float* h_group_norm, int vocab, int hidden, int groups,
                   float safety, float* upper);

void dense_logits(const float* w, const float* h, int vocab, int hidden, float* logits);

} // namespace ephs
