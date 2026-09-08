// Detection decodes: runtime.py decode_boxes (3D, BEV det grid) and
// viz_np.py decode_boxes2d_ms_np (per-camera multiscale 2D).
#pragma once

#include <array>
#include <vector>

struct Box3D {
  int cls;  // 0 = vehicle, 1 = vru
  float sc, x, y, l, w, yaw;
};

struct Box2D {
  int cls;  // 10-class det head
  float sc, cx, cy, w, h;  // 768x432 px
};

// hm [C,H,W] logits, reg [6,H,W]; det grid 300x250 @ 0.4 m:
// x = 80 - (row + o0) * 0.4, y = 50 - (col + o1) * 0.4
std::vector<Box3D> decode_boxes(const float* hm, const float* reg, int C,
                                int H, int W, float thresh = 0.3f,
                                int topk = 64);

struct Scale2D {
  const float* hm;   // [N,C,h,w] logits (N cameras, C=10)
  const float* reg;  // [N,4,h,w]
  int N, C, h, w;
};

// strides (4, 8, 16); returns one box list per camera
std::vector<std::vector<Box2D>> decode_boxes2d_ms(
    const std::array<Scale2D, 3>& scales, float thresh = 0.3f, int topk = 48);
