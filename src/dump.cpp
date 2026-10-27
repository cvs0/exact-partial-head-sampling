#include "ephs/dump.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace ephs {
namespace {

void write_u32(std::ostream& out, std::uint32_t v) {
  const unsigned char b[4] = {
      static_cast<unsigned char>(v & 0xffu),
      static_cast<unsigned char>((v >> 8) & 0xffu),
      static_cast<unsigned char>((v >> 16) & 0xffu),
      static_cast<unsigned char>((v >> 24) & 0xffu),
  };
  out.write(reinterpret_cast<const char*>(b), 4);
}

std::uint32_t read_u32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) {
    throw std::runtime_error("truncated head dump");
  }
  return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
         (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
}

void write_f32_le(std::ostream& out, const float* data, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    unsigned char b[4];
    std::memcpy(b, data + i, 4);
    const unsigned probe = 1u;
    if (*reinterpret_cast<const unsigned char*>(&probe) == 0) {
      std::swap(b[0], b[3]);
      std::swap(b[1], b[2]);
    }
    out.write(reinterpret_cast<char*>(b), 4);
  }
}

void read_f32_le(std::istream& in, float* data, std::size_t n) {
  const unsigned probe = 1u;
  const bool big = *reinterpret_cast<const unsigned char*>(&probe) == 0;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in) {
      throw std::runtime_error("truncated head dump");
    }
    if (big) {
      std::swap(b[0], b[3]);
      std::swap(b[1], b[2]);
    }
    std::memcpy(data + i, b, 4);
  }
}

} // namespace

void save_head_dump(const std::string& path, const Head& head, const float* states,
                    std::int32_t n_states) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out.write("EPHS0001", 8);
  write_u32(out, static_cast<std::uint32_t>(head.vocab));
  write_u32(out, static_cast<std::uint32_t>(head.hidden));
  write_u32(out, static_cast<std::uint32_t>(n_states));
  write_f32_le(out, head.w.data(), head.w.size());
  write_f32_le(out, states,
               static_cast<std::size_t>(n_states) * static_cast<std::size_t>(head.hidden));
  if (!out) {
    throw std::runtime_error("failed while writing " + path);
  }
}

Head load_head_dump(const std::string& path, std::vector<float>& states, std::int32_t& n_states) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot read " + path);
  }
  char magic[8];
  in.read(magic, 8);
  if (!in || std::string(magic, 8) != "EPHS0001") {
    throw std::runtime_error("bad head dump magic in " + path);
  }
  const auto vocab = static_cast<std::int32_t>(read_u32(in));
  const auto hidden = static_cast<std::int32_t>(read_u32(in));
  n_states = static_cast<std::int32_t>(read_u32(in));
  if (vocab < 1 || hidden < 1 || n_states < 1) {
    throw std::runtime_error("head dump has an empty dimension");
  }
  Head head;
  head.vocab = vocab;
  head.hidden = hidden;
  head.w.resize(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(hidden));
  states.resize(static_cast<std::size_t>(n_states) * static_cast<std::size_t>(hidden));
  read_f32_le(in, head.w.data(), head.w.size());
  read_f32_le(in, states.data(), states.size());
  return head;
}

} // namespace ephs
