#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ephs {

enum class Backend { Auto, Scalar, Avx2 };

enum class CertMode {
  // Block ceiling is max_i (U_i / T + G_i). Requires a per-token noise draw.
  Aligned,
  // Block ceiling is max_i(U_i) / T + max_i(G_i). The max noise is the only
  // noise the stopping rule needs; individual noises are drawn when the block opens.
  Split,
};

struct SampleOptions {
  float temperature = 1.f;
  // 0 selects the full vocabulary. Positive k certifies the exact top-k logit
  // set and then samples from that set. Top-p is not implemented.
  std::int32_t top_k = 0;
  std::int32_t block_size = 1;
  CertMode mode = CertMode::Aligned;
  // Added to every logit upper bound to absorb fp32 rounding in the int8 dot.
  float safety = 1e-3f;
};

struct SampleStats {
  std::int32_t token = -1;
  std::uint64_t shadow_bytes = 0;
  std::uint64_t row_bytes = 0;
  std::int32_t rows_read = 0;
  std::int32_t blocks_opened = 0;

  std::uint64_t head_bytes() const { return shadow_bytes + row_bytes; }
};

struct Head {
  std::int32_t vocab = 0;
  std::int32_t hidden = 0;
  // Row-major [vocab, hidden], fp32.
  std::vector<float> w;

  const float* row(std::int32_t i) const { return w.data() + static_cast<std::size_t>(i) * hidden; }
};

struct Shadow {
  std::int32_t vocab = 0;
  std::int32_t hidden = 0;
  std::int32_t groups = 1;
  // int8 codes, row-major [vocab, hidden]. Symmetric per-row scale.
  std::vector<std::int8_t> codes;
  std::vector<float> scale;
  // L2 norm of the quantization residual inside each contiguous hidden group.
  // Layout [vocab, groups]. Norms are rounded upward.
  std::vector<float> group_norm;

  std::uint64_t bytes() const {
    return static_cast<std::uint64_t>(codes.size()) * sizeof(std::int8_t) +
           static_cast<std::uint64_t>(scale.size()) * sizeof(float) +
           static_cast<std::uint64_t>(group_norm.size()) * sizeof(float);
  }
};

struct Workspace {
  std::vector<float> upper;
  std::vector<float> gumbel;
  std::vector<float> block_key;
  std::vector<float> h_gnorm;
  std::vector<std::int32_t> order;
  std::vector<float> scratch;
};

inline void validate_options(const SampleOptions& opt, std::int32_t vocab) {
  if (!(opt.temperature > 0.f) || !std::isfinite(opt.temperature)) {
    throw std::invalid_argument("temperature must be finite and positive");
  }
  if (opt.block_size < 1) {
    throw std::invalid_argument("block_size must be >= 1");
  }
  if (opt.top_k < 0 || opt.top_k > vocab) {
    throw std::invalid_argument("top_k must lie in [0, vocab]");
  }
  if (!(opt.safety >= 0.f) || !std::isfinite(opt.safety)) {
    throw std::invalid_argument("safety must be finite and non-negative");
  }
}

} // namespace ephs
