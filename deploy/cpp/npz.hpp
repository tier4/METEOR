// Minimal .npz / .npy reader (zlib) for ego_motion.npz.
//
// An .npz is a (possibly deflate-compressed) zip of .npy members. The files
// this demo reads hold little-endian C-order arrays: "v0" '<f4' (N,) and
// "pose" '<f4' (N,3) (verified against real ego_motion.npz headers); <f8 and
// integer dtypes are converted for safety. Everything is returned as float.
#pragma once

#include <zlib.h>

#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct NpyArray {
  std::vector<size_t> shape;
  std::vector<float> data;
  size_t rows() const { return shape.empty() ? 0 : shape[0]; }
};

namespace npz_detail {

inline uint16_t rd16(const uint8_t* p) {
  return (uint16_t)(p[0] | (p[1] << 8));
}
inline uint32_t rd32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

inline std::vector<uint8_t> inflateRaw(const uint8_t* src, size_t n,
                                       size_t outN) {
  std::vector<uint8_t> out(outN);
  z_stream zs;
  std::memset(&zs, 0, sizeof(zs));
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK)
    throw std::runtime_error("npz: inflateInit2 failed");
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = (uInt)n;
  zs.next_out = out.data();
  zs.avail_out = (uInt)outN;
  int r = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (r != Z_STREAM_END) throw std::runtime_error("npz: inflate failed");
  return out;
}

inline NpyArray parseNpy(const std::vector<uint8_t>& b) {
  if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0)
    throw std::runtime_error("npy: bad magic");
  int major = b[6];
  size_t hlen, hoff;
  if (major == 1) {
    hlen = rd16(b.data() + 8);
    hoff = 10;
  } else {
    hlen = rd32(b.data() + 8);
    hoff = 12;
  }
  if (hoff + hlen > b.size()) throw std::runtime_error("npy: bad header len");
  std::string h((const char*)b.data() + hoff, hlen);
  auto after = [&](const char* key) {
    size_t p = h.find(key);
    if (p == std::string::npos)
      throw std::runtime_error(std::string("npy: header key ") + key);
    return p + std::strlen(key);
  };
  size_t dp = after("'descr':");
  size_t q0 = h.find('\'', dp), q1 = h.find('\'', q0 + 1);
  std::string descr = h.substr(q0 + 1, q1 - q0 - 1);
  if (h.find("'fortran_order': False") == std::string::npos)
    throw std::runtime_error("npy: fortran order unsupported");
  size_t sp = after("'shape':");
  size_t p0 = h.find('(', sp), p1 = h.find(')', p0);
  std::string shp = h.substr(p0 + 1, p1 - p0 - 1);
  NpyArray a;
  size_t pos = 0;
  while (pos < shp.size()) {
    while (pos < shp.size() && !std::isdigit((unsigned char)shp[pos])) ++pos;
    if (pos >= shp.size()) break;
    size_t e = 0;
    a.shape.push_back(std::stoull(shp.substr(pos), &e));
    pos += e;
  }
  size_t n = 1;
  for (size_t s : a.shape) n *= s;
  const uint8_t* d = b.data() + hoff + hlen;
  size_t avail = b.size() - hoff - hlen;
  auto need = [&](size_t bytes) {
    if (avail < bytes) throw std::runtime_error("npy: truncated data");
  };
  a.data.resize(n);
  if (descr == "<f4") {
    need(n * 4);
    std::memcpy(a.data.data(), d, n * 4);
  } else if (descr == "<f8") {
    need(n * 8);
    for (size_t i = 0; i < n; ++i) {
      double v;
      std::memcpy(&v, d + 8 * i, 8);
      a.data[i] = (float)v;
    }
  } else if (descr == "<f2") {          // float16 (lidar_bev "lb" は fp16)
    need(n * 2);
    for (size_t i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, d + 2 * i, 2);
      uint32_t s_ = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      float v;
      if (e == 0) v = std::ldexp((float)m, -24);
      else if (e == 31) v = m ? NAN : INFINITY;
      else v = std::ldexp((float)(m | 0x400), (int)e - 25);
      a.data[i] = s_ ? -v : v;
    }
  } else if (descr == "<i4") {
    need(n * 4);
    for (size_t i = 0; i < n; ++i) {
      int32_t v;
      std::memcpy(&v, d + 4 * i, 4);
      a.data[i] = (float)v;
    }
  } else if (descr == "<i8") {
    need(n * 8);
    for (size_t i = 0; i < n; ++i) {
      int64_t v;
      std::memcpy(&v, d + 8 * i, 8);
      a.data[i] = (float)v;
    }
  } else {
    throw std::runtime_error("npy: unsupported dtype " + descr);
  }
  return a;
}

}  // namespace npz_detail

inline std::map<std::string, NpyArray> npz_load(const std::string& path) {
  using namespace npz_detail;
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npz: cannot open " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  if (buf.size() < 22) throw std::runtime_error("npz: too small");
  // end-of-central-directory record: scan back (max 64 KB comment)
  size_t i = buf.size() - 22;
  size_t stop = buf.size() > 22 + 65535 ? buf.size() - 22 - 65535 : 0;
  for (;; --i) {
    if (rd32(&buf[i]) == 0x06054b50u) break;
    if (i == stop) throw std::runtime_error("npz: no EOCD");
  }
  uint16_t nent = rd16(&buf[i + 10]);
  uint32_t cdOff = rd32(&buf[i + 16]);
  std::map<std::string, NpyArray> out;
  size_t p = cdOff;
  for (int e = 0; e < nent; ++e) {
    if (p + 46 > buf.size() || rd32(&buf[p]) != 0x02014b50u)
      throw std::runtime_error("npz: bad central directory");
    uint16_t method = rd16(&buf[p + 10]);
    uint32_t csize = rd32(&buf[p + 20]), usize = rd32(&buf[p + 24]);
    uint16_t nlen = rd16(&buf[p + 28]), elen = rd16(&buf[p + 30]);
    uint16_t clen = rd16(&buf[p + 32]);
    uint32_t lho = rd32(&buf[p + 42]);
    std::string name((const char*)&buf[p + 46], nlen);
    p += 46 + (size_t)nlen + elen + clen;
    if (lho + 30 > buf.size()) throw std::runtime_error("npz: bad offset");
    uint16_t lnlen = rd16(&buf[lho + 26]), lelen = rd16(&buf[lho + 28]);
    const uint8_t* data = &buf[lho + 30 + lnlen + lelen];
    std::vector<uint8_t> raw;
    if (method == 0)
      raw.assign(data, data + usize);
    else if (method == 8)
      raw = inflateRaw(data, csize, usize);
    else
      continue;  // unsupported member: skip (caller checks for keys)
    std::string key = (name.size() > 4 &&
                       name.compare(name.size() - 4, 4, ".npy") == 0)
                          ? name.substr(0, name.size() - 4)
                          : name;
    out[key] = parseNpy(raw);
  }
  return out;
}
