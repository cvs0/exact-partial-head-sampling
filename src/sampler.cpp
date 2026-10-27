#include "ephs/sampler.hpp"

#include "ephs/gemv.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace ephs {
namespace {

struct Cand {
  float logit = 0.f;
  std::int32_t index = 0;
};

bool better_logit(Cand a, Cand b) {
  if (a.logit != b.logit) {
    return a.logit > b.logit;
  }
  return a.index < b.index;
}

bool better_score(float score, std::int32_t index, float best, std::int32_t best_index) {
  if (best_index < 0) {
    return true;
  }
  if (score != best) {
    return score > best;
  }
  return index < best_index;
}

int block_begin(int block, int block_size) {
  return block * block_size;
}

int block_end(int block, int block_size, int vocab) {
  return std::min(vocab, (block + 1) * block_size);
}

int block_count(int vocab, int block_size) {
  return (vocab + block_size - 1) / block_size;
}

} // namespace

Sampler::Sampler(const Head& head, Shadow shadow) : head_(head), shadow_(std::move(shadow)) {
  if (shadow_.vocab != head_.vocab || shadow_.hidden != head_.hidden) {
    throw std::invalid_argument("shadow shape does not match head");
  }
  dense_bytes_ = static_cast<std::uint64_t>(head_.vocab) *
                 static_cast<std::uint64_t>(head_.hidden) * sizeof(float);
}

void Sampler::upper_bounds(const float* h, float safety, float* upper, Workspace& ws) const {
  ws.h_gnorm.resize(static_cast<std::size_t>(shadow_.groups));
  hidden_group_norms(h, head_.hidden, shadow_.groups, ws.h_gnorm.data());
  compute_upper(shadow_.codes.data(), shadow_.scale.data(), shadow_.group_norm.data(), h,
                ws.h_gnorm.data(), head_.vocab, head_.hidden, shadow_.groups, safety, upper);
}

void Sampler::logits(const float* h, float* out) const {
  dense_logits(head_.w.data(), h, head_.vocab, head_.hidden, out);
}

SampleStats Sampler::sample(const float* h, const SampleOptions& opt, Rng& rng,
                            Workspace& ws) const {
  validate_options(opt, head_.vocab);
  if (opt.top_k == 0 && opt.mode == CertMode::Split) {
    return sample_split_online(h, opt, rng, ws);
  }
  ws.gumbel.resize(static_cast<std::size_t>(head_.vocab));
  for (std::int32_t i = 0; i < head_.vocab; ++i) {
    ws.gumbel[static_cast<std::size_t>(i)] = static_cast<float>(rng.gumbel());
  }
  return sample_with_gumbels(h, ws.gumbel.data(), opt, ws);
}

SampleStats Sampler::sample_dense(const float* h, const SampleOptions& opt, Rng& rng,
                                  Workspace& ws) const {
  validate_options(opt, head_.vocab);
  ws.gumbel.resize(static_cast<std::size_t>(head_.vocab));
  for (std::int32_t i = 0; i < head_.vocab; ++i) {
    ws.gumbel[static_cast<std::size_t>(i)] = static_cast<float>(rng.gumbel());
  }
  return sample_dense_with_gumbels(h, ws.gumbel.data(), opt, ws);
}

SampleStats Sampler::sample_with_gumbels(const float* h, const float* gumbels,
                                         const SampleOptions& opt, Workspace& ws) const {
  validate_options(opt, head_.vocab);
  ws.upper.resize(static_cast<std::size_t>(head_.vocab));
  upper_bounds(h, opt.safety, ws.upper.data(), ws);
  if (opt.top_k > 0) {
    return sample_topk(h, gumbels, opt, ws);
  }
  return sample_full(h, gumbels, opt, ws);
}

SampleStats Sampler::sample_full(const float* h, const float* gumbels, const SampleOptions& opt,
                                 Workspace& ws) const {
  const int vocab = head_.vocab;
  const int hidden = head_.hidden;
  const int nb = block_count(vocab, opt.block_size);
  const float inv_t = 1.f / opt.temperature;
  ws.block_key.resize(static_cast<std::size_t>(nb));
  for (int b = 0; b < nb; ++b) {
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    if (opt.mode == CertMode::Aligned) {
      float key = -std::numeric_limits<float>::infinity();
      for (int i = s; i < e; ++i) {
        key = std::max(key, ws.upper[static_cast<std::size_t>(i)] * inv_t + gumbels[i]);
      }
      ws.block_key[static_cast<std::size_t>(b)] = key;
    } else {
      float max_u = -std::numeric_limits<float>::infinity();
      float max_g = -std::numeric_limits<float>::infinity();
      for (int i = s; i < e; ++i) {
        max_u = std::max(max_u, ws.upper[static_cast<std::size_t>(i)]);
        max_g = std::max(max_g, gumbels[i]);
      }
      ws.block_key[static_cast<std::size_t>(b)] = max_u * inv_t + max_g;
    }
  }
  ws.order.resize(static_cast<std::size_t>(nb));
  std::iota(ws.order.begin(), ws.order.end(), 0);
  std::sort(ws.order.begin(), ws.order.end(), [&](int a, int b) {
    const float ka = ws.block_key[static_cast<std::size_t>(a)];
    const float kb = ws.block_key[static_cast<std::size_t>(b)];
    if (ka != kb) {
      return ka > kb;
    }
    return a < b;
  });

  float best = -std::numeric_limits<float>::infinity();
  std::int32_t best_i = -1;
  std::int32_t rows = 0;
  std::int32_t opened = 0;
  for (int pos = 0; pos < nb; ++pos) {
    const int b = ws.order[static_cast<std::size_t>(pos)];
    const float key = ws.block_key[static_cast<std::size_t>(b)];
    if (best_i >= 0 && !(key >= best)) {
      break;
    }
    ++opened;
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    for (int i = s; i < e; ++i) {
      const float z = dot_f32(head_.row(i), h, hidden);
      const float score = z * inv_t + gumbels[i];
      if (better_score(score, i, best, best_i)) {
        best = score;
        best_i = i;
      }
    }
    rows += e - s;
  }

  SampleStats stats;
  stats.token = best_i;
  stats.shadow_bytes = shadow_.bytes();
  stats.rows_read = rows;
  stats.row_bytes =
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(hidden) * sizeof(float);
  stats.blocks_opened = opened;
  return stats;
}

SampleStats Sampler::sample_split_online(const float* h, const SampleOptions& opt, Rng& rng,
                                         Workspace& ws) const {
  const int vocab = head_.vocab;
  const int hidden = head_.hidden;
  const int nb = block_count(vocab, opt.block_size);
  const float inv_t = 1.f / opt.temperature;
  ws.upper.resize(static_cast<std::size_t>(vocab));
  upper_bounds(h, opt.safety, ws.upper.data(), ws);
  ws.block_key.resize(static_cast<std::size_t>(nb));
  ws.scratch.resize(static_cast<std::size_t>(nb));
  for (int b = 0; b < nb; ++b) {
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    float max_u = -std::numeric_limits<float>::infinity();
    for (int i = s; i < e; ++i) {
      max_u = std::max(max_u, ws.upper[static_cast<std::size_t>(i)]);
    }
    const double block_max = std::log(static_cast<double>(e - s)) + rng.gumbel();
    ws.scratch[static_cast<std::size_t>(b)] = static_cast<float>(block_max);
    ws.block_key[static_cast<std::size_t>(b)] = max_u * inv_t + static_cast<float>(block_max);
  }
  ws.order.resize(static_cast<std::size_t>(nb));
  std::iota(ws.order.begin(), ws.order.end(), 0);
  std::sort(ws.order.begin(), ws.order.end(), [&](int a, int b) {
    const float ka = ws.block_key[static_cast<std::size_t>(a)];
    const float kb = ws.block_key[static_cast<std::size_t>(b)];
    if (ka != kb) {
      return ka > kb;
    }
    return a < b;
  });

  float best = -std::numeric_limits<float>::infinity();
  std::int32_t best_i = -1;
  std::int32_t rows = 0;
  std::int32_t opened = 0;
  for (int pos = 0; pos < nb; ++pos) {
    const int b = ws.order[static_cast<std::size_t>(pos)];
    const float key = ws.block_key[static_cast<std::size_t>(b)];
    if (best_i >= 0 && !(key >= best)) {
      break;
    }
    ++opened;
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    const int width = e - s;
    const float block_max = ws.scratch[static_cast<std::size_t>(b)];
    const int arg = rng.uniform_index(width);
    for (int t = 0; t < width; ++t) {
      const int i = s + t;
      float g = block_max;
      if (t != arg) {
        const double gt = gumbel_truncated_below(static_cast<double>(block_max), rng);
        g = static_cast<float>(gt);
        if (!(g < block_max)) {
          g = std::nextafter(block_max, -std::numeric_limits<float>::infinity());
        }
      }
      const float z = dot_f32(head_.row(i), h, hidden);
      const float score = z * inv_t + g;
      if (better_score(score, i, best, best_i)) {
        best = score;
        best_i = i;
      }
    }
    rows += width;
  }

  SampleStats stats;
  stats.token = best_i;
  stats.shadow_bytes = shadow_.bytes();
  stats.rows_read = rows;
  stats.row_bytes =
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(hidden) * sizeof(float);
  stats.blocks_opened = opened;
  return stats;
}

SampleStats Sampler::sample_topk(const float* h, const float* gumbels, const SampleOptions& opt,
                                 Workspace& ws) const {
  const int vocab = head_.vocab;
  const int hidden = head_.hidden;
  const int k = opt.top_k;
  const int nb = block_count(vocab, opt.block_size);
  const float inv_t = 1.f / opt.temperature;
  ws.block_key.resize(static_cast<std::size_t>(nb));
  for (int b = 0; b < nb; ++b) {
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    float max_u = -std::numeric_limits<float>::infinity();
    for (int i = s; i < e; ++i) {
      max_u = std::max(max_u, ws.upper[static_cast<std::size_t>(i)]);
    }
    ws.block_key[static_cast<std::size_t>(b)] = max_u;
  }
  ws.order.resize(static_cast<std::size_t>(nb));
  std::iota(ws.order.begin(), ws.order.end(), 0);
  std::sort(ws.order.begin(), ws.order.end(), [&](int a, int b) {
    const float ka = ws.block_key[static_cast<std::size_t>(a)];
    const float kb = ws.block_key[static_cast<std::size_t>(b)];
    if (ka != kb) {
      return ka > kb;
    }
    return a < b;
  });

  std::vector<Cand> top;
  top.reserve(static_cast<std::size_t>(k));
  std::int32_t rows = 0;
  std::int32_t opened = 0;
  for (int pos = 0; pos < nb; ++pos) {
    const int b = ws.order[static_cast<std::size_t>(pos)];
    if (static_cast<int>(top.size()) == k) {
      const Cand worst = *std::min_element(top.begin(), top.end(),
                                           [](Cand a, Cand c) { return better_logit(c, a); });
      if (ws.block_key[static_cast<std::size_t>(b)] < worst.logit) {
        break;
      }
    }
    ++opened;
    const int s = block_begin(b, opt.block_size);
    const int e = block_end(b, opt.block_size, vocab);
    for (int i = s; i < e; ++i) {
      Cand cand;
      cand.logit = dot_f32(head_.row(i), h, hidden);
      cand.index = i;
      if (static_cast<int>(top.size()) < k) {
        top.push_back(cand);
      } else {
        auto worst_it = std::min_element(top.begin(), top.end(),
                                         [](Cand a, Cand c) { return better_logit(c, a); });
        if (better_logit(cand, *worst_it)) {
          *worst_it = cand;
        }
      }
    }
    rows += e - s;
  }

  float best = -std::numeric_limits<float>::infinity();
  std::int32_t best_i = -1;
  for (const Cand& cand : top) {
    const float score = cand.logit * inv_t + gumbels[cand.index];
    if (better_score(score, cand.index, best, best_i)) {
      best = score;
      best_i = cand.index;
    }
  }

  SampleStats stats;
  stats.token = best_i;
  stats.shadow_bytes = shadow_.bytes();
  stats.rows_read = rows;
  stats.row_bytes =
      static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(hidden) * sizeof(float);
  stats.blocks_opened = opened;
  return stats;
}

SampleStats Sampler::sample_dense_with_gumbels(const float* h, const float* gumbels,
                                               const SampleOptions& opt, Workspace& ws) const {
  validate_options(opt, head_.vocab);
  const int vocab = head_.vocab;
  const float inv_t = 1.f / opt.temperature;
  ws.scratch.resize(static_cast<std::size_t>(vocab));
  logits(h, ws.scratch.data());

  std::vector<char> keep(static_cast<std::size_t>(vocab), 1);
  if (opt.top_k > 0) {
    std::vector<Cand> top;
    top.reserve(static_cast<std::size_t>(opt.top_k));
    for (int i = 0; i < vocab; ++i) {
      Cand cand{ws.scratch[static_cast<std::size_t>(i)], i};
      if (static_cast<int>(top.size()) < opt.top_k) {
        top.push_back(cand);
      } else {
        auto worst_it = std::min_element(top.begin(), top.end(),
                                         [](Cand a, Cand c) { return better_logit(c, a); });
        if (better_logit(cand, *worst_it)) {
          *worst_it = cand;
        }
      }
    }
    std::fill(keep.begin(), keep.end(), 0);
    for (const Cand& cand : top) {
      keep[static_cast<std::size_t>(cand.index)] = 1;
    }
  }

  float best = -std::numeric_limits<float>::infinity();
  std::int32_t best_i = -1;
  for (int i = 0; i < vocab; ++i) {
    if (!keep[static_cast<std::size_t>(i)]) {
      continue;
    }
    const float score = ws.scratch[static_cast<std::size_t>(i)] * inv_t + gumbels[i];
    if (better_score(score, i, best, best_i)) {
      best = score;
      best_i = i;
    }
  }
  SampleStats stats;
  stats.token = best_i;
  stats.shadow_bytes = 0;
  stats.rows_read = vocab;
  stats.row_bytes = dense_bytes_;
  stats.blocks_opened = block_count(vocab, opt.block_size);
  return stats;
}

} // namespace ephs
