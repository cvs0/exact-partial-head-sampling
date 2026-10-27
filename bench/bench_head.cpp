#include "ephs/dump.hpp"
#include "ephs/gemv.hpp"
#include "ephs/sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void flush_range(const void* ptr, std::size_t bytes) {
  const auto* c = static_cast<const char*>(ptr);
  for (std::size_t i = 0; i < bytes; i += 64) {
    _mm_clflushopt(const_cast<char*>(c + i));
  }
  _mm_sfence();
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.compare(0, key.size(), key) == 0) {
      const auto pos = line.find(':');
      if (pos != std::string::npos) {
        std::string value = line.substr(pos + 1);
        while (!value.empty() && value.front() == ' ') {
          value.erase(value.begin());
        }
        return value;
      }
    }
  }
  return "unknown";
}

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

double entropy_nats(const std::vector<float>& logits, float temperature) {
  double m = -1e300;
  for (float z : logits) {
    m = std::max(m, static_cast<double>(z) / temperature);
  }
  double zsum = 0.0;
  double acc = 0.0;
  std::vector<double> e(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i) {
    e[i] = std::exp(static_cast<double>(logits[i]) / temperature - m);
    zsum += e[i];
  }
  for (double v : e) {
    const double p = v / zsum;
    if (p > 0.0) {
      acc -= p * std::log(p);
    }
  }
  return acc;
}

struct Config {
  float temperature;
  int top_k;
  int block;
  ephs::CertMode mode;
  bool flush;
  const char* mode_name;
};

struct Agg {
  double us_mean = 0;
  double us_median = 0;
  double tokens_per_sec = 0;
  double byte_ratio_mean = 0;
  double bytes_mean = 0;
  double rows_mean = 0;
  std::vector<double> quartile_entropy;
  std::vector<double> quartile_byte_ratio;
};

Agg time_partial(const ephs::Sampler& sampler, const std::vector<float>& states, int n_states,
                 const Config& cfg, std::uint64_t seed, const std::vector<double>* entropy,
                 std::uint64_t dense_bytes) {
  ephs::SampleOptions opt;
  opt.temperature = cfg.temperature;
  opt.top_k = cfg.top_k;
  opt.block_size = cfg.block;
  opt.mode = cfg.mode;
  opt.safety = 1e-3f;
  const int hidden = sampler.head().hidden;
  ephs::Workspace ws;
  ephs::Rng warmup_rng(seed);
  const float* h0 = states.data();
  sampler.sample(h0, opt, warmup_rng, ws);

  std::vector<double> us(static_cast<std::size_t>(n_states));
  std::vector<double> bytes(static_cast<std::size_t>(n_states));
  std::vector<double> rows(static_cast<std::size_t>(n_states));
  ephs::Rng rng(seed);
  for (int i = 0; i < n_states; ++i) {
    const float* h = states.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden);
    if (cfg.flush) {
      flush_range(sampler.head().w.data(), sampler.head().w.size() * sizeof(float));
      flush_range(sampler.shadow().codes.data(), sampler.shadow().codes.size());
      flush_range(sampler.shadow().scale.data(), sampler.shadow().scale.size() * sizeof(float));
      flush_range(sampler.shadow().group_norm.data(),
                  sampler.shadow().group_norm.size() * sizeof(float));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const ephs::SampleStats stats = sampler.sample(h, opt, rng, ws);
    const auto t1 = std::chrono::steady_clock::now();
    us[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    bytes[static_cast<std::size_t>(i)] = static_cast<double>(stats.head_bytes());
    rows[static_cast<std::size_t>(i)] = static_cast<double>(stats.rows_read);
  }
  Agg agg;
  agg.us_mean = std::accumulate(us.begin(), us.end(), 0.0) / n_states;
  std::vector<double> sorted = us;
  std::sort(sorted.begin(), sorted.end());
  agg.us_median = sorted[sorted.size() / 2];
  agg.tokens_per_sec = 1e6 / agg.us_mean;
  agg.bytes_mean = std::accumulate(bytes.begin(), bytes.end(), 0.0) / n_states;
  agg.rows_mean = std::accumulate(rows.begin(), rows.end(), 0.0) / n_states;
  agg.byte_ratio_mean = agg.bytes_mean / static_cast<double>(dense_bytes);
  if (entropy != nullptr) {
    std::vector<double> ent = *entropy;
    std::vector<std::size_t> order(ent.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return ent[a] < ent[b]; });
    const std::size_t q = order.size() / 4;
    for (int bin = 0; bin < 4; ++bin) {
      const std::size_t begin = static_cast<std::size_t>(bin) * q;
      const std::size_t end = (bin == 3) ? order.size() : begin + q;
      double es = 0, bs = 0;
      for (std::size_t k = begin; k < end; ++k) {
        es += ent[order[k]];
        bs += bytes[order[k]];
      }
      const double n = static_cast<double>(end - begin);
      agg.quartile_entropy.push_back(es / n);
      agg.quartile_byte_ratio.push_back((bs / n) / static_cast<double>(dense_bytes));
    }
  }
  return agg;
}

Agg time_dense(const ephs::Sampler& sampler, const std::vector<float>& states, int n_states,
               const Config& cfg, std::uint64_t seed) {
  ephs::SampleOptions opt;
  opt.temperature = cfg.temperature;
  opt.top_k = cfg.top_k;
  opt.block_size = cfg.block;
  opt.mode = cfg.mode;
  const int hidden = sampler.head().hidden;
  ephs::Workspace ws;
  ephs::Rng warmup_rng(seed + 1);
  sampler.sample_dense(states.data(), opt, warmup_rng, ws);
  std::vector<double> us(static_cast<std::size_t>(n_states));
  ephs::Rng rng(seed + 1);
  for (int i = 0; i < n_states; ++i) {
    const float* h = states.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden);
    if (cfg.flush) {
      flush_range(sampler.head().w.data(), sampler.head().w.size() * sizeof(float));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sampler.sample_dense(h, opt, rng, ws);
    const auto t1 = std::chrono::steady_clock::now();
    us[static_cast<std::size_t>(i)] = std::chrono::duration<double, std::micro>(t1 - t0).count();
  }
  Agg agg;
  agg.us_mean = std::accumulate(us.begin(), us.end(), 0.0) / n_states;
  std::sort(us.begin(), us.end());
  agg.us_median = us[us.size() / 2];
  agg.tokens_per_sec = 1e6 / agg.us_mean;
  agg.bytes_mean =
      static_cast<double>(sampler.head().vocab) * sampler.head().hidden * sizeof(float);
  agg.rows_mean = sampler.head().vocab;
  agg.byte_ratio_mean = 1.0;
  return agg;
}

int mismatches(const ephs::Sampler& sampler, const std::vector<float>& states, int n_states,
               const Config& cfg, std::uint64_t seed) {
  ephs::SampleOptions opt;
  opt.temperature = cfg.temperature;
  opt.top_k = cfg.top_k;
  opt.block_size = cfg.block;
  opt.mode = cfg.mode;
  opt.safety = 1e-3f;
  const int hidden = sampler.head().hidden;
  const int vocab = sampler.head().vocab;
  ephs::Workspace ws;
  ephs::Rng rng(seed);
  int bad = 0;
  for (int i = 0; i < n_states; ++i) {
    const float* h = states.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden);
    std::vector<float> g(static_cast<std::size_t>(vocab));
    for (float& x : g) {
      x = static_cast<float>(rng.gumbel());
    }
    const ephs::SampleStats partial = sampler.sample_with_gumbels(h, g.data(), opt, ws);
    const ephs::SampleStats dense = sampler.sample_dense_with_gumbels(h, g.data(), opt, ws);
    if (partial.token != dense.token) {
      ++bad;
    }
  }
  return bad;
}

float global_min_gap(const ephs::Sampler& sampler, const std::vector<float>& states, int n_states,
                     float safety) {
  const int hidden = sampler.head().hidden;
  const int vocab = sampler.head().vocab;
  ephs::Workspace ws;
  std::vector<float> upper(static_cast<std::size_t>(vocab));
  std::vector<float> logits(static_cast<std::size_t>(vocab));
  float gap = std::numeric_limits<float>::infinity();
  for (int i = 0; i < n_states; ++i) {
    const float* h = states.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(hidden);
    sampler.upper_bounds(h, safety, upper.data(), ws);
    sampler.logits(h, logits.data());
    for (int v = 0; v < vocab; ++v) {
      gap = std::min(gap, upper[static_cast<std::size_t>(v)] - logits[static_cast<std::size_t>(v)]);
    }
  }
  return gap;
}

void append_run(std::ostream& out, bool& first, const char* kind, const Config& cfg, const Agg& agg,
                int bad, int n_timed) {
  if (!first) {
    out << ",\n";
  }
  first = false;
  out << "  {\"kind\":\"" << kind << "\",\"temperature\":" << cfg.temperature
      << ",\"top_k\":" << cfg.top_k << ",\"block\":" << cfg.block << ",\"n_timed\":" << n_timed
      << ",\"mode\":\"" << cfg.mode_name << "\",\"flush\":" << (cfg.flush ? "true" : "false")
      << ",\"us_mean\":" << agg.us_mean << ",\"us_median\":" << agg.us_median
      << ",\"tokens_per_sec\":" << agg.tokens_per_sec
      << ",\"byte_ratio_mean\":" << agg.byte_ratio_mean << ",\"bytes_mean\":" << agg.bytes_mean
      << ",\"rows_mean\":" << agg.rows_mean << ",\"mismatches\":" << bad;
  if (!agg.quartile_entropy.empty()) {
    out << ",\"entropy_quartiles\":[";
    for (std::size_t i = 0; i < agg.quartile_entropy.size(); ++i) {
      if (i) {
        out << ",";
      }
      out << "{\"entropy_mean\":" << agg.quartile_entropy[i]
          << ",\"byte_ratio_mean\":" << agg.quartile_byte_ratio[i] << "}";
    }
    out << "]";
  }
  out << "}";
}

} // namespace

int main(int argc, char** argv) {
  std::string dump_path;
  std::string out_path;
  int groups = 8;
  std::uint64_t seed = 12345;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };
    if (arg == "--dump") {
      dump_path = need("--dump");
    } else if (arg == "--out") {
      out_path = need("--out");
    } else if (arg == "--groups") {
      groups = std::stoi(need("--groups"));
    } else if (arg == "--seed") {
      seed = std::stoull(need("--seed"));
    } else {
      std::cerr << "unknown argument " << arg << "\n";
      return 2;
    }
  }
  if (dump_path.empty() || out_path.empty()) {
    std::cerr << "usage: ephs_bench --dump HEAD.bin --out bench.json [--groups 8] [--seed N]\n";
    return 2;
  }

  std::vector<float> states;
  std::int32_t n_states = 0;
  const ephs::Head head = ephs::load_head_dump(dump_path, states, n_states);
  const ephs::Shadow shadow = ephs::build_shadow(head, groups);
  const ephs::Sampler sampler(head, shadow);
  const std::uint64_t dense_bytes = static_cast<std::uint64_t>(head.vocab) *
                                    static_cast<std::uint64_t>(head.hidden) * sizeof(float);

  const float min_gap = global_min_gap(sampler, states, n_states, 1e-3f);
  std::cout << "states " << n_states << " vocab " << head.vocab << " hidden " << head.hidden
            << " groups " << groups << " min_gap " << min_gap << " backend "
            << (ephs::active_backend() == ephs::Backend::Avx2 ? "avx2" : "scalar") << std::endl;
  if (!(min_gap >= 0.f)) {
    std::cerr << "certificate violated: min(U - logit) = " << min_gap << "\n";
    return 1;
  }

  const std::vector<Config> exact_cfgs = {
      {0.5f, 0, 1, ephs::CertMode::Aligned, false, "aligned"},
      {1.0f, 0, 1, ephs::CertMode::Aligned, false, "aligned"},
      {1.2f, 0, 1, ephs::CertMode::Aligned, false, "aligned"},
      {1.0f, 0, 1, ephs::CertMode::Split, false, "split"},
      {1.0f, 0, 32, ephs::CertMode::Aligned, false, "aligned"},
      {1.0f, 1, 1, ephs::CertMode::Aligned, false, "aligned"},
      {1.0f, 50, 1, ephs::CertMode::Aligned, false, "aligned"},
      {0.5f, 50, 1, ephs::CertMode::Aligned, false, "aligned"},
  };
  int bad_total = 0;
  for (const Config& cfg : exact_cfgs) {
    const int bad = mismatches(sampler, states, n_states, cfg, seed);
    std::cout << "exact T=" << cfg.temperature << " k=" << cfg.top_k << " block=" << cfg.block
              << " " << cfg.mode_name << " mismatches " << bad << std::endl;
    bad_total += bad;
  }
  if (bad_total != 0) {
    std::cerr << "partial and dense tokens disagreed on " << bad_total << " draws\n";
    return 1;
  }

  std::vector<Config> timed;
  for (float temperature : {0.5f, 0.8f, 1.0f, 1.2f}) {
    for (ephs::CertMode mode : {ephs::CertMode::Aligned, ephs::CertMode::Split}) {
      timed.push_back(Config{temperature, 0, 1, mode, false,
                             mode == ephs::CertMode::Aligned ? "aligned" : "split"});
    }
  }
  for (int block : {32, 128, 256}) {
    for (ephs::CertMode mode : {ephs::CertMode::Aligned, ephs::CertMode::Split}) {
      timed.push_back(Config{1.0f, 0, block, mode, false,
                             mode == ephs::CertMode::Aligned ? "aligned" : "split"});
    }
  }
  for (int top_k : {1, 50}) {
    for (float temperature : {0.5f, 1.0f, 1.2f}) {
      timed.push_back(Config{temperature, top_k, 1, ephs::CertMode::Aligned, false, "aligned"});
    }
  }
  // Cache-flushed timings are limited to the byte-optimal shape. clflushopt of the
  // whole head is the dominant cost, and 32 states are enough for a latency mean.
  for (float temperature : {0.5f, 0.8f, 1.0f, 1.2f}) {
    timed.push_back(Config{temperature, 0, 1, ephs::CertMode::Aligned, true, "aligned"});
  }
  timed.push_back(Config{1.0f, 50, 1, ephs::CertMode::Aligned, true, "aligned"});

  std::ofstream out(out_path);
  if (!out) {
    std::cerr << "cannot write " << out_path << "\n";
    return 1;
  }
  out.setf(std::ios::fmtflags(0), std::ios::floatfield);
  out.precision(10);
  out << "{\n";
  out << " \"cpu\":\"" << json_escape(cpu_model()) << "\",\n";
  out << " \"backend\":\"" << (ephs::active_backend() == ephs::Backend::Avx2 ? "avx2" : "scalar")
      << "\",\n";
  out << " \"vocab\":" << head.vocab << ",\n";
  out << " \"hidden\":" << head.hidden << ",\n";
  out << " \"groups\":" << groups << ",\n";
  out << " \"n_states\":" << n_states << ",\n";
  out << " \"seed\":" << seed << ",\n";
  out << " \"safety\":0.001,\n";
  out << " \"min_logit_gap\":" << min_gap << ",\n";
  out << " \"dense_bytes\":" << dense_bytes << ",\n";
  out << " \"shadow_bytes\":" << shadow.bytes() << ",\n";
  out << " \"runs\":[\n";
  bool first = true;
  for (const Config& cfg : timed) {
    std::vector<double> ent;
    const std::vector<double>* ent_ptr = nullptr;
    if (cfg.block == 1 && cfg.top_k == 0 && cfg.mode == ephs::CertMode::Aligned && !cfg.flush) {
      ent.resize(static_cast<std::size_t>(n_states));
      std::vector<float> logits(static_cast<std::size_t>(head.vocab));
      for (int i = 0; i < n_states; ++i) {
        const float* h =
            states.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(head.hidden);
        sampler.logits(h, logits.data());
        ent[static_cast<std::size_t>(i)] = entropy_nats(logits, cfg.temperature);
      }
      ent_ptr = &ent;
    }
    const int n_timed = cfg.flush ? std::min(n_states, 32) : n_states;
    const Agg partial = time_partial(sampler, states, n_timed, cfg, seed, ent_ptr, dense_bytes);
    const Agg dense = time_dense(sampler, states, n_timed, cfg, seed);
    std::cout << "T=" << cfg.temperature << " k=" << cfg.top_k << " block=" << cfg.block << " "
              << cfg.mode_name << (cfg.flush ? " cold" : " hot") << " n=" << n_timed << " ratio "
              << partial.byte_ratio_mean << " partial_us " << partial.us_mean << " dense_us "
              << dense.us_mean << std::endl;
    append_run(out, first, "partial", cfg, partial, 0, n_timed);
    append_run(out, first, "dense", cfg, dense, 0, n_timed);
  }
  out << "\n ]\n}\n";
  std::cout << "wrote " << out_path << "\n";
  return 0;
}
