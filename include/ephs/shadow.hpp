#pragma once

#include "ephs/types.hpp"

namespace ephs {

// Per-row symmetric int8 shadow of W plus grouped residual L2 norms.
// U_i = scale_i * <q_i, h> + sum_g ||e_i^{(g)}||_2 ||h^{(g)}||_2
// dominates W_i · h in real arithmetic. Stored norms are rounded up.
Shadow build_shadow(const Head& head, std::int32_t groups);

// rho_g = ||h^{(g)}||_2, rounded up, length `groups`.
void hidden_group_norms(const float* h, std::int32_t hidden, std::int32_t groups, float* out);

} // namespace ephs
