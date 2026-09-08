#include "decode.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr float DET_RES = 0.4f;
constexpr int DET2D_STRIDES[3] = {4, 8, 16};

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// 3x3 peak test == python's "p == dilate(p)": keep iff no neighbour is
// strictly greater (out-of-bounds pads compare as -1 < p, i.e. never win)
inline bool isPeak(const float* plane, int H, int W, int r, int c, float v) {
  for (int dr = -1; dr <= 1; ++dr)
    for (int dc = -1; dc <= 1; ++dc) {
      int rr = r + dr, cc = c + dc;
      if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
      if (plane[(size_t)rr * W + cc] > v) return false;
    }
  return true;
}

}  // namespace

std::vector<Box3D> decode_boxes(const float* hm, const float* reg, int C,
                                int H, int W, float thresh, int topk) {
  // candidates: NMS'd peaks above threshold, best `topk` by score
  std::vector<std::pair<float, size_t>> cand;  // (score, flat idx)
  for (int cls = 0; cls < C; ++cls) {
    const float* plane = hm + (size_t)cls * H * W;
    for (int r = 0; r < H; ++r)
      for (int c = 0; c < W; ++c) {
        float v = plane[(size_t)r * W + c];
        float p = sigmoidf(v);
        if (p <= thresh) continue;
        if (!isPeak(plane, H, W, r, c, v)) continue;
        cand.emplace_back(p, (size_t)cls * H * W + (size_t)r * W + c);
      }
  }
  std::sort(cand.begin(), cand.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  if ((int)cand.size() > topk) cand.resize(topk);
  std::vector<Box3D> boxes;
  boxes.reserve(cand.size());
  for (auto& [sc, i] : cand) {
    int cls = (int)(i / ((size_t)H * W));
    int rc = (int)(i % ((size_t)H * W));
    int ri = rc / W, ci = rc % W;
    auto R = [&](int ch) { return reg[(size_t)ch * H * W + (size_t)ri * W + ci]; };
    boxes.push_back({cls, sc, 80.0f - (ri + R(0)) * DET_RES,
                     50.0f - (ci + R(1)) * DET_RES, std::exp(R(2)),
                     std::exp(R(3)), std::atan2(R(4), R(5))});
  }
  return boxes;
}

std::vector<std::vector<Box2D>> decode_boxes2d_ms(
    const std::array<Scale2D, 3>& scales, float thresh, int topk) {
  std::vector<std::vector<Box2D>> out;
  for (int si = 0; si < 3; ++si) {
    const Scale2D& S = scales[si];
    float s = (float)DET2D_STRIDES[si];
    if (out.empty()) out.resize(S.N);
    for (int ni = 0; ni < S.N; ++ni) {
      std::vector<std::pair<float, int>> cand;  // (score, flat idx in C*h*w)
      const float* hmN = S.hm + (size_t)ni * S.C * S.h * S.w;
      auto clip = [](float x) { return std::max(-50.0f, std::min(50.0f, x)); };
      for (int c = 0; c < S.C; ++c) {
        const float* plane = hmN + (size_t)c * S.h * S.w;
        for (int r = 0; r < S.h; ++r)
          for (int cc = 0; cc < S.w; ++cc) {
            float v = clip(plane[(size_t)r * S.w + cc]);
            float p = sigmoidf(v);
            if (p <= thresh) continue;
            // python clips before the dilate-based peak test
            bool peak = true;
            for (int dr = -1; dr <= 1 && peak; ++dr)
              for (int dc = -1; dc <= 1; ++dc) {
                int rr = r + dr, c2 = cc + dc;
                if (rr < 0 || rr >= S.h || c2 < 0 || c2 >= S.w) continue;
                if (clip(plane[(size_t)rr * S.w + c2]) > v) {
                  peak = false;
                  break;
                }
              }
            if (!peak) continue;
            cand.emplace_back(p, c * S.h * S.w + r * S.w + cc);
          }
      }
      std::sort(cand.begin(), cand.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      if ((int)cand.size() > topk) cand.resize(topk);
      const float* regN = S.reg + (size_t)ni * 4 * S.h * S.w;
      for (auto& [sc, j] : cand) {
        int ccls = j / (S.h * S.w), rc = j % (S.h * S.w);
        int rr = rc / S.w, cc = rc % S.w;
        auto R = [&](int ch) {
          return regN[(size_t)ch * S.h * S.w + (size_t)rr * S.w + cc];
        };
        out[ni].push_back({ccls, sc, (cc + R(1)) * s, (rr + R(0)) * s,
                           std::exp(R(2)) * s, std::exp(R(3)) * s});
      }
    }
  }
  return out;
}
