#pragma once

#include "ephs/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ephs {

// Little-endian dump:
//   magic[8] = "EPHS0001"
//   u32 vocab, u32 hidden, u32 n_states
//   f32 W[vocab * hidden] row major
//   f32 H[n_states * hidden]
Head load_head_dump(const std::string& path, std::vector<float>& states, std::int32_t& n_states);

void save_head_dump(const std::string& path, const Head& head, const float* states,
                    std::int32_t n_states);

} // namespace ephs
