#pragma once

#include "ephs/rng.hpp"
#include "ephs/shadow.hpp"
#include "ephs/types.hpp"

namespace ephs {

class Sampler {
public:
  Sampler(const Head& head, Shadow shadow);

  const Head& head() const { return head_; }
  const Shadow& shadow() const { return shadow_; }

  // Full-vocabulary or top-k sample. Split mode with top_k == 0 draws one
  // block-max Gumbel per block and materializes truncated Gumbels only inside
  // opened blocks. Every other setting draws (or consumes) one Gumbel per token.
  SampleStats sample(const float* h, const SampleOptions& opt, Rng& rng, Workspace& ws) const;

  // Both routines consume gumbels[0, vocab). Split uses the per-block maximum
  // of those draws as the block noise, which couples the partial read to a
  // dense Gumbel-max on the same noise.
  SampleStats sample_with_gumbels(const float* h, const float* gumbels, const SampleOptions& opt,
                                  Workspace& ws) const;
  SampleStats sample_dense_with_gumbels(const float* h, const float* gumbels,
                                        const SampleOptions& opt, Workspace& ws) const;

  SampleStats sample_dense(const float* h, const SampleOptions& opt, Rng& rng, Workspace& ws) const;

  void upper_bounds(const float* h, float safety, float* upper, Workspace& ws) const;
  void logits(const float* h, float* out) const;

private:
  SampleStats sample_full(const float* h, const float* gumbels, const SampleOptions& opt,
                          Workspace& ws) const;
  SampleStats sample_topk(const float* h, const float* gumbels, const SampleOptions& opt,
                          Workspace& ws) const;
  SampleStats sample_split_online(const float* h, const SampleOptions& opt, Rng& rng,
                                  Workspace& ws) const;

  const Head& head_;
  Shadow shadow_;
  std::uint64_t dense_bytes_ = 0;
};

} // namespace ephs
