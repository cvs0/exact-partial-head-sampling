#include "ephs/dump.hpp"
#include "ephs/gemv.hpp"
#include "ephs/sampler.hpp"
#include "stats.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const char* expr, const char* file, int line) {
  if (!cond) {
    std::cerr << file << ":" << line << " failed: " << expr << "\n";
    ++g_fails;
  }
}

#define CHECK(cond) check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

ephs::Head make_head(std::mt19937_64& eng, int vocab, int hidden, double stddev) {
  std::normal_distribution<double> nd(0.0, stddev);
  ephs::Head head;
  head.vocab = vocab;
  head.hidden = hidden;
  head.w.resize(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden));
  for (float& x : head.w) {
    x = static_cast<float>(nd(eng));
  }
  return head;
}

std::vector<float> make_h(std::mt19937_64& eng, int hidden, double stddev) {
  std::normal_distribution<double> nd(0.0, stddev);
  std::vector<float> h(static_cast<std::size_t>(hidden));
  for (float& x : h) {
    x = static_cast<float>(nd(eng));
  }
  return h;
}

std::vector<float> draw_gumbels(ephs::Rng& rng, int n) {
  std::vector<float> g(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    g[static_cast<std::size_t>(i)] = static_cast<float>(rng.gumbel());
  }
  return g;
}

void softmax_from_logits(const std::vector<float>& logits, const std::vector<char>& keep,
                         float temperature, std::vector<double>& prob) {
  const int n = static_cast<int>(logits.size());
  prob.assign(static_cast<std::size_t>(n), 0.0);
  double m = -1e300;
  for (int i = 0; i < n; ++i) {
    if (!keep[static_cast<std::size_t>(i)]) {
      continue;
    }
    m = std::max(m, static_cast<double>(logits[static_cast<std::size_t>(i)]) / temperature);
  }
  double z = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!keep[static_cast<std::size_t>(i)]) {
      continue;
    }
    const double e =
        std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)]) / temperature - m);
    prob[static_cast<std::size_t>(i)] = e;
    z += e;
  }
  for (double& p : prob) {
    p /= z;
  }
}

void test_special_functions() {
  const double e1 = ephs::test::gamma_q(1.0, 1.0);
  CHECK(std::fabs(e1 - std::exp(-1.0)) < 1e-10);
  CHECK(std::fabs(ephs::test::chi2_sf(3.841458820694124, 1) - 0.05) < 1e-6);
  CHECK(std::fabs(ephs::test::chi2_sf(5.991464547107979, 2) - 0.05) < 1e-6);
  CHECK(ephs::test::chi2_sf(0.0, 4) == 1.0);
  CHECK(ephs::test::chi2_sf(1000.0, 1) < 1e-12);
}

void test_gumbel_marginal(std::uint64_t seed) {
  ephs::Rng rng(seed);
  const int n = 20000;
  double sum = 0.0;
  double sum2 = 0.0;
  for (int i = 0; i < n; ++i) {
    const double g = rng.gumbel();
    sum += g;
    sum2 += g * g;
  }
  const double mean = sum / n;
  const double var = sum2 / n - mean * mean;
  // E[Gumbel(0,1)] = gamma_Euler, Var = pi^2 / 6.
  CHECK(std::fabs(mean - 0.5772156649015329) < 0.02);
  CHECK(std::fabs(var - 1.6449340668482264) < 0.08);
}

void test_truncated_gumbel(std::uint64_t seed) {
  ephs::Rng rng(seed);
  const double ceiling = 0.7;
  const int n = 15000;
  std::vector<double> pit;
  pit.reserve(static_cast<std::size_t>(n));
  const double fm = std::exp(-std::exp(-ceiling));
  int argmax_hits[5] = {0, 0, 0, 0, 0};
  for (int trial = 0; trial < n; ++trial) {
    const double g = ephs::gumbel_truncated_below(ceiling, rng);
    CHECK(g < ceiling);
    const double u = std::exp(-std::exp(-g)) / fm;
    pit.push_back(u);
  }
  const double d = ephs::test::ks_uniform_statistic(pit);
  const double p = ephs::test::ks_sf(d, n);
  CHECK(p > 1e-6);

  const int trials = 8000;
  for (int trial = 0; trial < trials; ++trial) {
    const double m = std::log(5.0) + rng.gumbel();
    const int arg = rng.uniform_index(5);
    ++argmax_hits[arg];
    for (int i = 0; i < 5; ++i) {
      if (i == arg) {
        continue;
      }
      const double g = ephs::gumbel_truncated_below(m, rng);
      CHECK(g < m);
    }
  }
  double chi = 0.0;
  const double expected = trials / 5.0;
  for (int hits : argmax_hits) {
    const double diff = hits - expected;
    chi += diff * diff / expected;
  }
  CHECK(ephs::test::chi2_sf(chi, 4) > 1e-6);
}

void test_dot_backends() {
  if (!ephs::cpu_supports_avx2()) {
    return;
  }
  std::mt19937_64 eng(11);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int n = 768;
  std::vector<float> a(static_cast<std::size_t>(n));
  std::vector<float> b(static_cast<std::size_t>(n));
  std::vector<std::int8_t> q(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    a[static_cast<std::size_t>(i)] = static_cast<float>(nd(eng));
    b[static_cast<std::size_t>(i)] = static_cast<float>(nd(eng));
    q[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(eng() % 255 - 127);
  }
  const float d0 = ephs::dot_f32_scalar(a.data(), b.data(), n);
  const float d1 = ephs::dot_f32_avx2(a.data(), b.data(), n);
  CHECK(std::fabs(d0 - d1) < 1e-3f * (1.f + std::fabs(d0)));
  const float i0 = ephs::dot_i8_f32_scalar(q.data(), b.data(), n);
  const float i1 = ephs::dot_i8_f32_avx2(q.data(), b.data(), n);
  CHECK(std::fabs(i0 - i1) < 1e-3f * (1.f + std::fabs(i0)));
}

float min_gap(const ephs::Sampler& sampler, const float* h, float safety) {
  ephs::Workspace ws;
  std::vector<float> upper(static_cast<std::size_t>(sampler.head().vocab));
  std::vector<float> logits(static_cast<std::size_t>(sampler.head().vocab));
  sampler.upper_bounds(h, safety, upper.data(), ws);
  sampler.logits(h, logits.data());
  float gap = std::numeric_limits<float>::infinity();
  for (int i = 0; i < sampler.head().vocab; ++i) {
    gap = std::min(gap, upper[static_cast<std::size_t>(i)] - logits[static_cast<std::size_t>(i)]);
  }
  return gap;
}

void test_bound_dominates() {
  std::mt19937_64 eng(3);
  for (ephs::Backend backend : {ephs::Backend::Scalar, ephs::Backend::Avx2}) {
    if (backend == ephs::Backend::Avx2 && !ephs::cpu_supports_avx2()) {
      continue;
    }
    ephs::set_backend(backend);
    const ephs::Head head = make_head(eng, 96, 768, 0.02);
    const ephs::Shadow shadow = ephs::build_shadow(head, 8);
    const ephs::Sampler sampler(head, shadow);
    for (int trial = 0; trial < 8; ++trial) {
      const std::vector<float> h = make_h(eng, 768, 1.0);
      const float gap = min_gap(sampler, h.data(), 1e-3f);
      CHECK(gap >= 0.f);
    }
  }
  ephs::set_backend(ephs::Backend::Auto);
}

void test_zero_head_opens_one_row() {
  ephs::set_backend(ephs::Backend::Scalar);
  ephs::Head head;
  head.vocab = 64;
  head.hidden = 16;
  head.w.assign(static_cast<std::size_t>(head.vocab * head.hidden), 0.f);
  ephs::Sampler sampler(head, ephs::build_shadow(head, 4));
  ephs::Rng rng(9);
  const std::vector<float> h(16, 0.3f);
  const std::vector<float> g = draw_gumbels(rng, head.vocab);
  ephs::SampleOptions opt;
  opt.safety = 0.f;
  opt.block_size = 1;
  opt.mode = ephs::CertMode::Aligned;
  opt.temperature = 1.f;
  ephs::Workspace ws;
  const ephs::SampleStats partial = sampler.sample_with_gumbels(h.data(), g.data(), opt, ws);
  const ephs::SampleStats dense = sampler.sample_dense_with_gumbels(h.data(), g.data(), opt, ws);
  CHECK(partial.token == dense.token);
  CHECK(partial.rows_read == 1);
  CHECK(partial.shadow_bytes == sampler.shadow().bytes());
  CHECK(partial.row_bytes == static_cast<std::uint64_t>(partial.rows_read) * 16u * 4u);
  ephs::set_backend(ephs::Backend::Auto);
}

void test_coupling() {
  std::mt19937_64 eng(17);
  const std::vector<ephs::Backend> backends = {ephs::Backend::Scalar, ephs::cpu_supports_avx2()
                                                                          ? ephs::Backend::Avx2
                                                                          : ephs::Backend::Scalar};
  for (ephs::Backend backend : backends) {
    ephs::set_backend(backend);
    for (int trial = 0; trial < 40; ++trial) {
      const int hidden = 32;
      const int vocab = 48;
      ephs::Head head = make_head(eng, vocab, hidden, 0.15);
      ephs::Sampler sampler(head, ephs::build_shadow(head, 4));
      const std::vector<float> h = make_h(eng, hidden, 1.0);
      CHECK(min_gap(sampler, h.data(), 1e-3f) >= 0.f);
      ephs::Rng noise(1000 + static_cast<std::uint64_t>(trial) +
                      (backend == ephs::Backend::Avx2 ? 50 : 0));
      const std::vector<float> g = draw_gumbels(noise, vocab);
      ephs::Workspace ws;
      const std::vector<int> blocks = {1, 3, 16, 48};
      const std::vector<float> temps = {0.5f, 1.1f};
      const std::vector<int> topks = {0, 1, 4, 7};
      for (int block : blocks) {
        for (float temperature : temps) {
          for (int top_k : topks) {
            for (ephs::CertMode mode : {ephs::CertMode::Aligned, ephs::CertMode::Split}) {
              ephs::SampleOptions opt;
              opt.temperature = temperature;
              opt.block_size = block;
              opt.top_k = top_k;
              opt.mode = mode;
              opt.safety = 1e-3f;
              const ephs::SampleStats partial =
                  sampler.sample_with_gumbels(h.data(), g.data(), opt, ws);
              const ephs::SampleStats dense =
                  sampler.sample_dense_with_gumbels(h.data(), g.data(), opt, ws);
              CHECK(partial.token == dense.token);
              CHECK(partial.token >= 0);
              CHECK(partial.rows_read >= 1);
              CHECK(partial.rows_read <= vocab);
              CHECK(partial.head_bytes() == partial.shadow_bytes + partial.row_bytes);
            }
          }
        }
      }
    }
  }
  ephs::set_backend(ephs::Backend::Auto);
}

void test_huge_safety_reads_everything() {
  std::mt19937_64 eng_w(4);
  std::mt19937_64 eng_h(5);
  ephs::Head head = make_head(eng_w, 30, 16, 0.1);
  ephs::Sampler sampler(head, ephs::build_shadow(head, 4));
  const std::vector<float> h = make_h(eng_h, 16, 1.0);
  ephs::Rng rng(6);
  const std::vector<float> g = draw_gumbels(rng, head.vocab);
  ephs::SampleOptions opt;
  opt.safety = 1e6f;
  opt.block_size = 5;
  opt.mode = ephs::CertMode::Split;
  ephs::Workspace ws;
  const ephs::SampleStats stats = sampler.sample_with_gumbels(h.data(), g.data(), opt, ws);
  CHECK(stats.rows_read == head.vocab);
}

struct ChiCase {
  ephs::CertMode mode;
  int block;
  float temperature;
  int top_k;
  bool online_split;
};

void test_chisquare_and_ks() {
  std::mt19937_64 eng(21);
  const int vocab = 12;
  const int hidden = 16;
  ephs::Head head = make_head(eng, vocab, hidden, 0.05);
  // A modest spike keeps the distribution identifiable without starving any atom.
  for (int j = 0; j < hidden; ++j) {
    head.w[static_cast<std::size_t>(j)] += (j == 0) ? 0.4f : 0.f;
  }
  ephs::set_backend(ephs::Backend::Scalar);
  ephs::Sampler sampler(head, ephs::build_shadow(head, 4));
  const std::vector<float> h = make_h(eng, hidden, 1.0);
  std::vector<float> logits(static_cast<std::size_t>(vocab));
  sampler.logits(h.data(), logits.data());

  const std::vector<ChiCase> cases = {
      {ephs::CertMode::Aligned, 1, 1.f, 0, false},  {ephs::CertMode::Aligned, 5, 0.5f, 0, false},
      {ephs::CertMode::Split, 1, 1.f, 0, true},     {ephs::CertMode::Split, 7, 1.2f, 0, true},
      {ephs::CertMode::Aligned, 4, 0.8f, 5, false}, {ephs::CertMode::Split, 3, 1.f, 4, false},
  };
  const int n_samples = 20000;
  for (const ChiCase& cs : cases) {
    ephs::SampleOptions opt;
    opt.mode = cs.mode;
    opt.block_size = cs.block;
    opt.temperature = cs.temperature;
    opt.top_k = cs.top_k;
    opt.safety = 1e-3f;
    std::vector<char> keep(static_cast<std::size_t>(vocab), 1);
    if (cs.top_k > 0) {
      // The certified top-k set does not depend on noise. Recover it from one dense call's
      // support by marking the k best logits with the same tie break the sampler uses.
      std::vector<int> idx(static_cast<std::size_t>(vocab));
      std::iota(idx.begin(), idx.end(), 0);
      std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float la = logits[static_cast<std::size_t>(a)];
        const float lb = logits[static_cast<std::size_t>(b)];
        if (la != lb) {
          return la > lb;
        }
        return a < b;
      });
      std::fill(keep.begin(), keep.end(), 0);
      for (int i = 0; i < cs.top_k; ++i) {
        keep[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] = 1;
      }
    }
    std::vector<double> prob;
    softmax_from_logits(logits, keep, cs.temperature, prob);
    double min_expected = 1e300;
    int support = 0;
    for (int i = 0; i < vocab; ++i) {
      if (prob[static_cast<std::size_t>(i)] == 0.0) {
        continue;
      }
      ++support;
      min_expected = std::min(min_expected, prob[static_cast<std::size_t>(i)] * n_samples);
    }
    if (!(min_expected >= 5.0)) {
      std::cerr << "min_expected " << min_expected << " block " << cs.block << " T "
                << cs.temperature << " top_k " << cs.top_k << " online " << cs.online_split << "\n";
    }
    CHECK(min_expected >= 5.0);

    std::vector<int> counts(static_cast<std::size_t>(vocab), 0);
    ephs::Rng rng(99);
    ephs::Workspace ws;
    std::vector<double> pit;
    pit.reserve(4000);
    for (int s = 0; s < n_samples; ++s) {
      ephs::SampleStats stats;
      if (cs.online_split) {
        stats = sampler.sample(h.data(), opt, rng, ws);
      } else {
        const std::vector<float> g = draw_gumbels(rng, vocab);
        stats = sampler.sample_with_gumbels(h.data(), g.data(), opt, ws);
        const ephs::SampleStats dense =
            sampler.sample_dense_with_gumbels(h.data(), g.data(), opt, ws);
        CHECK(stats.token == dense.token);
      }
      CHECK(stats.token >= 0 && stats.token < vocab);
      CHECK(keep[static_cast<std::size_t>(stats.token)]);
      ++counts[static_cast<std::size_t>(stats.token)];
      if (static_cast<int>(pit.size()) < 4000) {
        double cdf = 0.0;
        for (int i = 0; i < stats.token; ++i) {
          cdf += prob[static_cast<std::size_t>(i)];
        }
        const double u = cdf + rng.uniform_open() * prob[static_cast<std::size_t>(stats.token)];
        pit.push_back(u);
      }
    }
    double chi = 0.0;
    for (int i = 0; i < vocab; ++i) {
      const double expected = prob[static_cast<std::size_t>(i)] * n_samples;
      if (expected == 0.0) {
        CHECK(counts[static_cast<std::size_t>(i)] == 0);
        continue;
      }
      const double diff = counts[static_cast<std::size_t>(i)] - expected;
      chi += diff * diff / expected;
    }
    const double p = ephs::test::chi2_sf(chi, support - 1);
    CHECK(p > 1e-8);
    const double ks_p =
        ephs::test::ks_sf(ephs::test::ks_uniform_statistic(pit), static_cast<int>(pit.size()));
    CHECK(ks_p > 1e-6);
  }
  ephs::set_backend(ephs::Backend::Auto);
}

void test_dump_roundtrip() {
  std::mt19937_64 eng_w(8);
  std::mt19937_64 eng_h(9);
  ephs::Head head = make_head(eng_w, 5, 4, 0.2);
  std::vector<float> states = make_h(eng_h, 4 * 3, 1.0);
  const std::string path = "/tmp/ephs_test_head.bin";
  ephs::save_head_dump(path, head, states.data(), 3);
  std::vector<float> loaded_states;
  std::int32_t n = 0;
  const ephs::Head loaded = ephs::load_head_dump(path, loaded_states, n);
  CHECK(n == 3);
  CHECK(loaded.vocab == 5);
  CHECK(loaded.hidden == 4);
  CHECK(loaded.w == head.w);
  CHECK(loaded_states == states);
  std::filesystem::remove(path);
}

void test_rejects_bad_options() {
  std::mt19937_64 eng(1);
  ephs::Head head = make_head(eng, 4, 4, 0.1);
  ephs::Sampler sampler(head, ephs::build_shadow(head, 2));
  ephs::Workspace ws;
  ephs::Rng rng(1);
  const std::vector<float> h(4, 0.f);
  ephs::SampleOptions opt;
  opt.temperature = 0.f;
  bool threw = false;
  try {
    sampler.sample(h.data(), opt, rng, ws);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  bool groups_threw = false;
  try {
    ephs::build_shadow(head, 3);
  } catch (const std::invalid_argument&) {
    groups_threw = true;
  }
  CHECK(groups_threw);
}

} // namespace

int main() {
  test_special_functions();
  test_gumbel_marginal(1);
  test_truncated_gumbel(2);
  test_dot_backends();
  test_bound_dominates();
  test_zero_head_opens_one_row();
  test_huge_safety_reads_everything();
  test_coupling();
  test_chisquare_and_ks();
  test_dump_roundtrip();
  test_rejects_bad_options();
  if (g_fails != 0) {
    std::cerr << g_fails << " check(s) failed\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
