#include "render.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <NvInfer.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode.hpp"
#include "viz.hpp"

const char* CAM_IN8[8] = {"CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                          "CAM_BACK_WIDE",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
                          "CAM_FRONT_NARROW", "CAM_BACK_NARROW"};

namespace {

constexpr int EGO_K = 3;
// tile layout order (orin_render.CAM8) -- NOT the model input order
const char* CAM8[8] = {"CAM_FRONT_LEFT", "CAM_FRONT_WIDE", "CAM_FRONT_RIGHT",
                       "CAM_FRONT_NARROW", "CAM_BACK_LEFT", "CAM_BACK_WIDE",
                       "CAM_BACK_RIGHT", "CAM_BACK_NARROW"};
constexpr int CW = 375, CH = 210;
constexpr int VW = 1920, VH = 1080;
constexpr double XF = 80.0;  // forward extent of the grid (rows * 0.2 - XF = rear)
const double YH = 25.0;

const char* envs(const char* k, const char* d) {
  const char* v = std::getenv(k);
  return v ? v : d;
}
double envd(const char* k, double d) {
  const char* v = std::getenv(k);
  return v ? std::atof(v) : d;
}

const TensorView& tv(const OutMap& out, const std::string& nm) {
  auto it = out.find(nm);
  if (it == out.end()) throw std::runtime_error("missing output " + nm);
  return it->second;
}
const float* f32(const OutMap& out, const std::string& nm) {
  return (const float*)tv(out, nm).ptr;
}

// lane / seg2d / depth arrive as uint8 argmax maps from the int8/uint8
// engines; legacy fp32-logits engines get an argmax here (like python).
// Returns a dense u8 buffer with `planes` maps of h*w.
std::vector<uint8_t> asArgmaxU8(const TensorView& v, int planes, int h,
                                int w) {
  std::vector<uint8_t> buf((size_t)planes * h * w);
  size_t n = buf.size();
  std::vector<int64_t> s(v.shape.begin() + 1, v.shape.end());
  if (v.dtype == nvinfer1::DataType::kUINT8 ||
      v.dtype == nvinfer1::DataType::kINT8) {
    std::memcpy(buf.data(), v.ptr, n);
  } else if (v.dtype == nvinfer1::DataType::kINT32) {
    const int32_t* p = (const int32_t*)v.ptr;
    for (size_t i = 0; i < n; ++i) buf[i] = (uint8_t)p[i];
  } else if (v.dtype == nvinfer1::DataType::kFLOAT) {
    int C = (int)s[s.size() - 3];
    const float* p = (const float*)v.ptr;
    for (int pl = 0; pl < planes; ++pl)
      for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
          const float* base = p + ((size_t)pl * C) * h * w + (size_t)r * w + c;
          int best = 0;
          float bv = base[0];
          for (int k = 1; k < C; ++k) {
            float q = base[(size_t)k * h * w];
            if (q > bv) {
              bv = q;
              best = k;
            }
          }
          buf[(size_t)pl * h * w + (size_t)r * w + c] = (uint8_t)best;
        }
  } else {
    throw std::runtime_error("unsupported argmax-map dtype");
  }
  return buf;
}

// orin_render.draw_grid
void draw_grid(cv::Mat& bev, double span, double yh, double viewF, double& sy,
               double& sx, int& cy) {
  int h = bev.rows, w = bev.cols;
  sy = h / span;
  sx = w / (2 * yh);
  cy = (int)(viewF * sy);
  for (int d : {20, 40, 60})
    for (int sign : {1, -1}) {
      int y = (int)(cy - sign * d * sy);
      if (0 <= y && y < h) {
        cv::line(bev, {0, y}, {w, y}, {60, 60, 60}, 1, cv::LINE_AA);
        cv::putText(bev, std::to_string(d) + "m", {4, y - 3},
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, {170, 170, 170}, 1,
                    cv::LINE_AA);
      }
    }
  cv::line(bev, {w / 2, 0}, {w / 2, h}, {90, 90, 90}, 1, cv::LINE_AA);
  std::vector<cv::Point> tri = {{w / 2, cy - 11},
                                {w / 2 - 7, cy + 8},
                                {w / 2 + 7, cy + 8}};
  cv::fillPoly(bev, std::vector<std::vector<cv::Point>>{tri},
               cv::Scalar(255, 255, 255));
}

std::atomic<int> g_prevMode{-1};  // compose_frame._prev_mode

float halfToFloat(uint16_t h) {
  uint32_t s_ = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = std::ldexp((float)m, -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp((float)(m | 0x400), (int)e - 25);
  return s_ ? -v : v;
}

// ---- orin_render.seg_fuse_logit / seg_fuse_np: BEV seg temporal fusion ----
// Shared state across the render workers (same tolerance as python: a worker
// may read a one-frame-stale state; geometry is still right because the warp
// is computed from the stored pose).
std::mutex g_segMx;
cv::Mat g_segAcc;            // [9,H,W] log-probs accumulated (CV_32F, 9 ch)
std::array<float, 3> g_segAccPose{};
bool g_segAccOk = false;
cv::Mat g_segCls, g_segConf;  // argmax fallback state
std::array<float, 3> g_segStPose{};
bool g_segStOk = false;

bool warpM(const float* pose, const std::array<float, 3>& pp, cv::Mat& M) {
  double cp = std::cos(pp[2]), sp = std::sin(pp[2]);
  double dx = pose[0] - pp[0], dy = pose[1] - pp[1];
  double tx = cp * dx + sp * dy, ty = -sp * dx + cp * dy;
  double dyaw = pose[2] - pp[2];
  if (std::fabs(tx) + std::fabs(ty) >= 10.0 || std::fabs(dyaw) >= 0.5) return false;
  double c = std::cos(dyaw), s2 = std::sin(dyaw);
  M = (cv::Mat_<float>(2, 3) << c, s2, -250 * c - 400 * s2 - 5 * ty + 250,
       -s2, c, 250 * s2 - 400 * c - 5 * tx + 400);
  return true;
}

// lane: [H,W] u8 argmax (modified in place). logit: fp16/fp32 [9,H,W] or null.
void seg_fuse(cv::Mat& lane, const TensorView* logit, const float* pose) {
  const int H = lane.rows, W = lane.cols;
  if (H != 800 || W != 500) return;  // the warp constants are for the 800x500 grid
  if (!pose) {
    std::lock_guard<std::mutex> lk(g_segMx);
    g_segAccOk = false; g_segStOk = false; return;
  }
  if (logit) {
    // log_softmax over 9 classes -- outside the lock (the expensive part, ~40 ms)
    cv::Mat lp(H, W, CV_32FC(9));
    const int C = 9;
    std::vector<float> v(C);
    for (int r = 0; r < H; ++r) {
      float* dst = lp.ptr<float>(r);
      for (int c = 0; c < W; ++c) {
        float mx = -1e30f;
        for (int k = 0; k < C; ++k) {
          size_t idx = (size_t)k * H * W + (size_t)r * W + c;
          v[k] = logit->dtype == nvinfer1::DataType::kHALF
                     ? halfToFloat(((const uint16_t*)logit->ptr)[idx])
                     : ((const float*)logit->ptr)[idx];
          mx = std::max(mx, v[k]);
        }
        float se = 0;
        for (int k = 0; k < C; ++k) se += std::exp(v[k] - mx);
        float lse = mx + std::log(se);
        for (int k = 0; k < C; ++k) dst[c * C + k] = v[k] - lse;
      }
    }
    std::lock_guard<std::mutex> lk(g_segMx);
    cv::Mat M;
    if (g_segAccOk && warpM(pose, g_segAccPose, M)) {
      cv::Mat ch[9], wr[9];
      cv::split(g_segAcc, ch);
      for (int k = 0; k < 9; k += 3) {
        cv::Mat m3, w3;
        cv::merge(std::vector<cv::Mat>{ch[k], ch[k + 1], ch[k + 2]}, m3);
        cv::warpAffine(m3, w3, M, {W, H}, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::split(w3, wr + k);
      }
      cv::Mat wmerged;
      cv::merge(std::vector<cv::Mat>(wr, wr + 9), wmerged);
      lp += 0.7f * wmerged;
    }
    g_segAcc = lp;
    g_segAccPose = {pose[0], pose[1], pose[2]};
    g_segAccOk = true;
    for (int r = 175; r < H; ++r) {              // x <= 45 m only
      const float* lr = lp.ptr<float>(r);
      uint8_t* out = lane.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {
        uint8_t cur = out[c];
        if (cur == 3 || cur == 4 || cur == 5 || cur == 6) continue;  // thin classes protected
        int best = 0;
        for (int k = 1; k < 9; ++k)
          if (lr[c * 9 + k] > lr[c * 9 + best]) best = k;
        out[c] = (uint8_t)best;
      }
    }
    return;
  }
  // argmax-only fallback (seg_fuse_np): persistence vote on area classes 1/2/8
  std::lock_guard<std::mutex> lk(g_segMx);
  auto isArea = [](uint8_t k) { return k == 1 || k == 2 || k == 8; };
  cv::Mat M;
  if (g_segStOk && warpM(pose, g_segStPose, M)) {
    cv::Mat wcls, wconf;
    cv::warpAffine(g_segCls, wcls, M, {W, H}, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::warpAffine(g_segConf, wconf, M, {W, H}, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP,
                   cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat fused = lane.clone(), conf(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r) {
      const uint8_t* cur = lane.ptr<uint8_t>(r);
      const uint8_t* wc = wcls.ptr<uint8_t>(r);
      const uint8_t* wf = wconf.ptr<uint8_t>(r);
      uint8_t* fo = fused.ptr<uint8_t>(r);
      uint8_t* cf = conf.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) {
        cf[c] = isArea(cur[c]) ? 3 : 0;
        if (isArea(cur[c]) && wc[c] == cur[c]) cf[c] = (uint8_t)std::min(wf[c] + 1, 6);
        if (cur[c] == 0 && isArea(wc[c]) && wf[c] > 0) { fo[c] = wc[c]; cf[c] = wf[c] - 1; }
      }
    }
    fused.rowRange(175, H).copyTo(lane.rowRange(175, H));
    g_segCls = fused; g_segConf = conf;
  } else {
    g_segCls = lane.clone();
    g_segConf = cv::Mat(H, W, CV_8UC1);
    for (int r = 0; r < H; ++r) {
      const uint8_t* cur = lane.ptr<uint8_t>(r);
      uint8_t* cf = g_segConf.ptr<uint8_t>(r);
      for (int c = 0; c < W; ++c) cf[c] = isArea(cur[c]) ? 3 : 0;
    }
  }
  g_segStPose = {pose[0], pose[1], pose[2]};
  g_segStOk = true;
}

// orin_render.thin_road_edge_np: keep only the innermost 1 px of the edge band
void thin_road_edge(cv::Mat& lane) {
  cv::Mat edge = lane == 6, drv;
  if (cv::countNonZero(edge) == 0) return;
  cv::Mat d1 = lane == 1, d3 = lane == 3, d4 = lane == 4, d5 = lane == 5;
  drv = d1 | d3 | d4 | d5;
  cv::Mat drvDil;
  cv::dilate(drv, drvDil, cv::Mat::ones(3, 3, CV_8U));
  cv::Mat inner = edge & drvDil & ~drv;
  lane.setTo(0, edge);
  lane.setTo(6, inner);
}

// runtime.stationary_head_healthy: INT8 can collapse the 1-ch logit to a constant
bool stationaryHealthy(const float* stat, size_t n) {
  if (!stat || n == 0) return false;
  double s = 0, s2 = 0, mn = 1e30, mx = -1e30; size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    float v = stat[i];
    if (!std::isfinite(v)) continue;
    s += v; s2 += (double)v * v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); ++k;
  }
  if (!k) return false;
  double mean = s / k, var = s2 / k - mean * mean;
  return std::sqrt(std::max(var, 0.0)) >= 0.05 && (mx - mn) >= 0.20;
}

// compose_frame._yaw_tracks: 2.5 m match -> flip suppression + EMA a=0.4
std::mutex g_yawMx;
std::vector<std::array<float, 3>> g_yawTracks;

// rotated-rectangle IoU in the BEV (metres); greedy same-class NMS
double rotIoU(const Box3D& a, const Box3D& b) {
  cv::RotatedRect ra({a.x, a.y}, {a.l, a.w}, (float)(a.yaw * 180.0 / M_PI));
  cv::RotatedRect rb({b.x, b.y}, {b.l, b.w}, (float)(b.yaw * 180.0 / M_PI));
  std::vector<cv::Point2f> pts;
  int r = cv::rotatedRectangleIntersection(ra, rb, pts);
  if (r == cv::INTERSECT_NONE || pts.size() < 3) return 0.0;
  std::vector<cv::Point2f> hull;
  cv::convexHull(pts, hull);
  double inter = cv::contourArea(hull);
  double uni = (double)a.l * a.w + (double)b.l * b.w - inter;
  return inter / std::max(uni, 1e-6);
}

// intersection / smaller area: catches a small box contained in a large one (IoU stays low)
double rotContain(const Box3D& a, const Box3D& b) {
  cv::RotatedRect ra({a.x, a.y}, {a.l, a.w}, (float)(a.yaw * 180.0 / M_PI));
  cv::RotatedRect rb({b.x, b.y}, {b.l, b.w}, (float)(b.yaw * 180.0 / M_PI));
  std::vector<cv::Point2f> pts;
  int r = cv::rotatedRectangleIntersection(ra, rb, pts);
  if (r == cv::INTERSECT_NONE || pts.size() < 3) return 0.0;
  std::vector<cv::Point2f> hull;
  cv::convexHull(pts, hull);
  double inter = cv::contourArea(hull);
  return inter / std::max(std::min((double)a.l * a.w, (double)b.l * b.w), 1e-6);
}

std::vector<Box3D> bevBoxNms(std::vector<Box3D> det, double iouTh) {
  std::sort(det.begin(), det.end(), [](const Box3D& a, const Box3D& b) { return a.sc > b.sc; });
  std::vector<Box3D> kept;
  for (const auto& d : det) {
    bool ok = true;
    for (const auto& k : kept)
      if (k.cls == d.cls && (rotIoU(d, k) > iouTh || rotContain(d, k) > 0.6)) { ok = false; break; }
    if (ok) kept.push_back(d);
  }
  return kept;
}

// ray through pixel (u,v) meets the ground plane (ego z = groundZ)?  Tce = T_cam_ego
bool groundPoint(double u, double v, const float* Kc, const float* T, double groundZ,
                 double& x, double& y) {
  // camera centre c = -R^T t ; direction d = R^T K^-1 [u v 1]
  double dc[3] = {(u - Kc[2]) / Kc[0], (v - Kc[5]) / Kc[4], 1.0};
  double c[3], d[3];
  for (int r = 0; r < 3; ++r) {
    c[r] = -(T[0 * 4 + r] * T[3] + T[1 * 4 + r] * T[7] + T[2 * 4 + r] * T[11]);
    d[r] = T[0 * 4 + r] * dc[0] + T[1 * 4 + r] * dc[1] + T[2 * 4 + r] * dc[2];
  }
  if (d[2] >= -1e-6) return false;
  double lam = (groundZ - c[2]) / d[2];
  if (lam <= 0) return false;
  x = c[0] + lam * d[0]; y = c[1] + lam * d[1];
  return true;
}

std::string trtVersionTag() {
  int v = getInferLibVersion();
  char b[64];
  std::snprintf(b, sizeof(b), "%d.%d.%d", v / 10000, (v / 100) % 100, v % 100);
  return b;
}

std::set<int> hide2d() {
  std::set<int> s;
  std::string v = envs("METEOR_2D_HIDE", "7");
  size_t i = 0;
  while (i < v.size()) {
    size_t j = v.find(',', i);
    if (j == std::string::npos) j = v.size();
    if (j > i) s.insert(std::atoi(v.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return s;
}

}  // namespace

cv::Mat compose_frame(const std::vector<cv::Mat>& raw,
                      const std::vector<std::string>& cams, const float* K,
                      const float* Tc, float v0, const OutMap& out, double dt,
                      double fps_now, const float* pose3, const float* lidarBev) {
  const int N = (int)cams.size();
  static const std::set<int> HIDE = hide2d();
  static const double TH2D = envd("METEOR_TH2D", 0.30);
  static const bool SEG_OV = std::string(envs("METEOR_SEG2D_OVERLAY", "0")) == "1";
  static const bool DEPTH_PANEL = std::string(envs("METEOR_DEPTH_PANEL", "1")) != "0";
  static const int DBINS = (int)envd("METEOR_DEPTH_BINS", 64);
  static const bool UNK2D = std::string(envs("METEOR_UNK2D", "1")) != "0";
  static const double TH_UNK = envd("METEOR_UNK2D_TH", TH2D);
  static const bool SEG_FUSE = std::string(envs("METEOR_SEG_FUSE", "1")) != "0";
  static const bool THIN = std::string(envs("METEOR_NO_THIN", "0")) == "0";
  static const double RISK_GAIN = envd("METEOR_RISK_GAIN", 1.0);
  static const double STAT_TH = envd("METEOR_STAT_LOGIT_THRESH", 0.0);
  static const std::string GPU_TAG = envs("METEOR_GPU_NAME", "Orin (nvgpu)");

  // ---- grid extents from the lane map (set_bev_extent) ----------------
  const auto& laneShape = tv(out, "lane").shape;  // [1,H,W] u8 or [1,C,H,W]
  int laneH = (int)laneShape[laneShape.size() - 2];
  int laneW = (int)laneShape[laneShape.size() - 1];
  const double XR = laneH * 0.2 - XF;  // 800 rows -> 80 m rear
  const double VIEW_F = 60.0, VIEW_R = std::min(60.0, XR);

  // ---- decodes -------------------------------------------------------
  const auto& hmShape = tv(out, "hm").shape;  // [1,2,H,W]
  int detC = (int)hmShape[1], detH = (int)hmShape[2], detW = (int)hmShape[3];
  auto detAll = decode_boxes(f32(out, "hm"), f32(out, "reg"), detC, detH,
                             detW, 0.15f, 64);
  std::vector<Box3D> det;
  for (const auto& b : detAll)  // veh 0.35 / vru 0.15, grid-range clip
    if (b.sc > (b.cls == 0 ? 0.35f : 0.15f) && b.x >= -(XR - 2.0) &&
        b.x <= XF - 2.0)
      det.push_back(b);
  det = bevBoxNms(det, envd("METEOR_BEV_NMS_IOU", 0.3));   // BEV rotated-IoU NMS (2026-09-08)
  {  // temporal yaw smoothing (orin_render 2026-08-13)
    std::lock_guard<std::mutex> lk(g_yawMx);
    std::vector<std::array<float, 3>> next;
    for (auto& b : det) {
      const std::array<float, 3>* best = nullptr; double bd = 1e30;
      for (const auto& t : g_yawTracks) {
        double dd = (b.x - t[0]) * (b.x - t[0]) + (b.y - t[1]) * (b.y - t[1]);
        if (dd < 6.25 && dd < bd) { bd = dd; best = &t; }
      }
      if (best) {
        float py = (*best)[2];
        float dy = std::fmod((double)b.yaw - py + M_PI, 2 * M_PI);
        if (dy < 0) dy += 2 * M_PI;
        dy -= M_PI;
        if (std::fabs(dy) > M_PI / 2) {
          b.yaw = b.yaw + (dy < 0 ? M_PI : -M_PI);
          dy = std::fmod((double)b.yaw - py + M_PI, 2 * M_PI);
          if (dy < 0) dy += 2 * M_PI;
          dy -= M_PI;
        }
        b.yaw = py + 0.4f * dy;
      }
      next.push_back({b.x, b.y, b.yaw});
    }
    g_yawTracks = next;
  }
  std::array<Scale2D, 3> scales;
  for (int i = 0; i < 3; ++i) {
    std::string h = "hm2d_s" + std::to_string(i);
    std::string r = "reg2d_s" + std::to_string(i);
    const auto& hs = tv(out, h).shape;  // [1,N,C,h,w]
    scales[i] = {f32(out, h), f32(out, r), (int)hs[1], (int)hs[2], (int)hs[3],
                 (int)hs[4]};
  }
  auto b2d = decode_boxes2d_ms(scales, (float)TH2D, 48);

  const float* ego = f32(out, "ego");  // [1,42]
  float lg[EGO_K];
  for (int i = 0; i < EGO_K; ++i) lg[i] = ego[12 * EGO_K + i];
  int prev = g_prevMode.load();
  if (prev >= 0 && prev < EGO_K) lg[prev] += 0.35f;
  int kMode = 0;
  for (int i = 1; i < EGO_K; ++i)
    if (lg[i] > lg[kMode]) kMode = i;
  // straight-preference rule (orin_render 2026-08-18)
  if (kMode != 0 && (lg[kMode] - lg[0]) < 1.0f) kMode = 0;
  g_prevMode.store(kMode);
  float mx = *std::max_element(lg, lg + EGO_K);
  double pr[EGO_K], prSum = 0;
  for (int i = 0; i < EGO_K; ++i) {
    pr[i] = std::exp((double)lg[i] - mx);
    prSum += pr[i];
  }
  for (int i = 0; i < EGO_K; ++i) pr[i] /= prSum;
  float sel[15];
  for (int i = 0; i < 12; ++i) sel[i] = ego[kMode * 12 + i];
  for (int i = 0; i < 3; ++i) sel[12 + i] = ego[12 * EGO_K + EGO_K + i];

  std::vector<uint8_t> segp;
  int segH = 0, segW = 0;
  if (SEG_OV && out.count("seg2d")) {
    const auto& segShape = tv(out, "seg2d").shape;
    segH = (int)segShape[segShape.size() - 2];
    segW = (int)segShape[segShape.size() - 1];
    segp = asArgmaxU8(tv(out, "seg2d"), N, segH, segW);
  }
  const auto& depShape = tv(out, "depth").shape;
  int depH = (int)depShape[depShape.size() - 2];
  int depW = (int)depShape[depShape.size() - 1];
  auto dep = asArgmaxU8(tv(out, "depth"), N, depH, depW);
  const float* stat = out.count("stationary") ? f32(out, "stationary") : nullptr;
  int statH = 0, statW = 0;
  if (stat) {
    const auto& statShape = tv(out, "stationary").shape;
    statH = (int)statShape[statShape.size() - 2];
    statW = (int)statShape[statShape.size() - 1];
  }
  const bool statOk = stat && stationaryHealthy(stat, (size_t)statH * statW);
  const float* traj = out.count("traj") ? f32(out, "traj") : nullptr;
  int trajC = 0, trajH = 0, trajW = 0;
  if (traj) {
    const auto& trajShape = tv(out, "traj").shape;
    trajC = (int)trajShape[1];
    trajH = (int)trajShape[2];
    trajW = (int)trajShape[3];
  }

  cv::Mat canvas(VH, VW, CV_8UC3, cv::Scalar(18, 17, 16));

  auto camIndex = [&](const char* chn) {
    for (int i = 0; i < N; ++i)
      if (cams[i] == chn) return i;
    return -1;
  };

  // ---- RGB tiles ------------------------------------------------------
  for (int k8 = 0; k8 < 8; ++k8) {
    const char* chn = CAM8[k8];
    int x0 = 8 + (k8 % 4) * (CW + 6);
    int y0 = 34 + (k8 / 4) * (CH + 26);
    int i = camIndex(chn);
    if (i < 0) {
      cv::putText(canvas, std::string(chn) + " (blank)", {x0 + 70, y0 + CH / 2},
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, {90, 90, 90}, 1, cv::LINE_AA);
      continue;
    }
    cv::Mat img;
    cv::resize(raw[i], img, {CW, CH});
    if (SEG_OV && !segp.empty()) {
      cv::Mat segi(segH, segW, CV_8UC1,
                   (void*)(segp.data() + (size_t)i * segH * segW));
      cv::Mat segr;
      cv::resize(segi, segr, {CW, CH}, 0, 0, cv::INTER_NEAREST);
      cv::Mat ov(CH, CW, CV_8UC3);
      for (int r = 0; r < CH; ++r) {
        const uint8_t* sp = segr.ptr<uint8_t>(r);
        cv::Vec3b* op = ov.ptr<cv::Vec3b>(r);
        for (int c = 0; c < CW; ++c) op[c] = PALETTE_BGR[sp[c] % 9];
      }
      cv::addWeighted(img, 0.75, ov, 0.25, 0, img);
    }
    draw_boxes_on_rgb(img, det, K + (size_t)i * 9, Tc + (size_t)i * 16, CW,
                      CH);
    std::vector<Box2D> shown;
    for (const auto& b : b2d[i])
      if (!HIDE.count(b.cls)) shown.push_back(b);
    draw_boxes2d(img, shown, CW, CH);
    if (std::strcmp(chn, "CAM_FRONT_WIDE") == 0)
      draw_path_ribbon(img, sel, K + (size_t)i * 9, Tc + (size_t)i * 16, CW,
                       CH);
    cv::putText(canvas, chn, {x0, y0 - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {200, 200, 200}, 1, cv::LINE_AA);
    img.copyTo(canvas(cv::Rect(x0, y0, CW, CH)));
  }
  // ---- depth tiles: same tile size and CAM8 order (METEOR_DEPTH_PANEL) ---
  if (DEPTH_PANEL)
    for (int k8 = 0; k8 < 8; ++k8) {
      const char* chn = CAM8[k8];
      int x0 = 8 + (k8 % 4) * (CW + 6);
      int y0 = 34 + (2 + k8 / 4) * (CH + 26);
      int i = camIndex(chn);
      if (i < 0) {
        cv::putText(canvas, "(blank)", {x0 + CW / 2 - 28, y0 + CH / 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {90, 90, 90}, 1,
                    cv::LINE_AA);
        continue;
      }
      cv::Mat depi(depH, depW, CV_8UC1,
                   (void*)(dep.data() + (size_t)i * depH * depW));
      cv::Mat d8;
      cv::resize(depi, d8, {CW, CH}, 0, 0, cv::INTER_NEAREST);
      const double sc = 255.0 / std::max(DBINS - 1, 1);
      for (int r = 0; r < CH; ++r) {
        uint8_t* p = d8.ptr<uint8_t>(r);
        for (int c = 0; c < CW; ++c)
          p[c] = (uint8_t)std::min(255.0, p[c] * sc);
      }
      cv::Mat cm;
      cv::applyColorMap(d8, cm, cv::COLORMAP_TURBO);
      cm.copyTo(canvas(cv::Rect(x0, y0, CW, CH)));
    }
  // ---- BEV pane --------------------------------------------------------
  auto laneBuf = asArgmaxU8(tv(out, "lane"), 1, laneH, laneW);
  cv::Mat laneMat(laneH, laneW, CV_8UC1, laneBuf.data());
  if (SEG_FUSE)
    seg_fuse(laneMat, out.count("lane_logit") ? &out.at("lane_logit") : nullptr, pose3);
  if (THIN) thin_road_edge(laneMat);
  auto& lane = laneBuf;
  // crop_bev_np(lane, XF, XR, VIEW_F, VIEW_R, YH) -- same double math
  double res = (XF + XR) / laneH;
  int r0 = std::max(0, (int)((XF - VIEW_F) / res));
  int r1 = std::min(laneH, (int)((XF + VIEW_R) / res));
  double resw = 2 * 50.0 / laneW;
  int c0 = std::max(0, (int)((50.0 - YH) / resw));
  int c1 = std::min(laneW, (int)((50.0 + YH) / resw));
  int pcH = r1 - r0, pcW = c1 - c0;
  cv::Mat pcCol(pcH, pcW, CV_8UC3);
  for (int r = 0; r < pcH; ++r) {
    const uint8_t* lp = lane.data() + (size_t)(r0 + r) * laneW + c0;
    cv::Vec3b* op = pcCol.ptr<cv::Vec3b>(r);
    for (int c = 0; c < pcW; ++c) {
      uint8_t k = lp[c] % 9;
      // DEMO_PALETTE: sidewalk(2) / parking(8) hidden
      op[c] = (k == 2 || k == 8) ? PALETTE_BGR[0] : PALETTE_BGR[k];
    }
  }
  const int BH2 = 1000;
  int BW2 = (int)((double)BH2 * pcW / pcH);
  cv::Mat bev;
  cv::resize(pcCol, bev, {BW2, BH2}, 0, 0, cv::INTER_NEAREST);
  double span = VIEW_F + VIEW_R;
  double sy2, sx2;
  int cy0;
  draw_grid(bev, span, YH, VIEW_F, sy2, sx2, cy0);
  for (const auto& b : det) {
    if (b.x > VIEW_F || b.x < -VIEW_R || std::fabs(b.y) > YH) continue;
    double cb = std::cos((double)b.yaw), sb = std::sin((double)b.yaw);
    int rr0 = (int)((XF - b.x) / 0.4);
    int cc0 = (int)((50.0 - b.y) / 0.4);
    bool stationary = false;
    if (statOk && 0 <= rr0 && rr0 < statH && 0 <= cc0 && cc0 < statW) {
      stationary = stat[(size_t)rr0 * statW + cc0] > STAT_TH;
    } else if (traj && 0 <= rr0 && rr0 < trajH && 0 <= cc0 && cc0 < trajW) {
      // runtime.stationary_at trajectory fallback: |wp6| < 0.5 m
      float vv[39] = {0};
      for (int q = 0; q < trajC && q < 39; ++q)
        vv[q] = traj[(size_t)q * trajH * trajW + (size_t)rr0 * trajW + cc0];
      const float* wp = vv;
      if (trajC >= 39) {
        int kb = 0;
        for (int q = 1; q < 3; ++q) if (vv[36 + q] > vv[36 + kb]) kb = q;
        wp = vv + kb * 12;
      }
      stationary = std::hypot(wp[10], wp[11]) < 0.5;
    }
    cv::Scalar col = stationary ? cv::Scalar(160, 160, 160)
                     : b.cls < 1 ? cv::Scalar(0, 215, 255)
                                 : cv::Scalar(255, 0, 255);
    std::vector<cv::Point> cor;
    const double off[4][2] = {{b.l / 2.0, b.w / 2.0},
                              {b.l / 2.0, -b.w / 2.0},
                              {-b.l / 2.0, -b.w / 2.0},
                              {-b.l / 2.0, b.w / 2.0}};
    for (auto& o : off) {
      double pxm = b.x + o[0] * cb - o[1] * sb;
      double pym = b.y + o[0] * sb + o[1] * cb;
      cor.emplace_back((int)((YH - pym) * sx2), (int)((VIEW_F - pxm) * sy2));
    }
    cv::polylines(bev, std::vector<std::vector<cv::Point>>{cor}, true, col, 2);
    int cxp = (int)((YH - b.y) * sx2);
    int cyp = (int)((VIEW_F - b.x) * sy2);
    int fxp = (int)((YH - (b.y + (b.l / 2.0) * sb)) * sx2);
    int fyp = (int)((VIEW_F - (b.x + (b.l / 2.0) * cb)) * sy2);
    cv::line(bev, {cxp, cyp}, {fxp, fyp}, col, 2);
    // VRU futures suppressed only -- stationary boxes keep theirs
    if (traj && b.cls != 1 && 0 <= rr0 && rr0 < trajH && 0 <= cc0 &&
        cc0 < trajW) {
      float v[39] = {0};
      for (int q = 0; q < trajC && q < 39; ++q)
        v[q] = traj[(size_t)q * trajH * trajW + (size_t)rr0 * trajW + cc0];
      const float* wps;
      float wbuf[12];
      if (trajC >= 39) {
        int kb = 0;
        for (int q = 1; q < 3; ++q)
          if (v[36 + q] > v[36 + kb]) kb = q;
        std::memcpy(wbuf, v + kb * 12, sizeof(wbuf));
        wps = wbuf;
      } else {
        wps = v;
      }
      std::vector<cv::Point> pts = {{cxp, cyp}};
      for (int q = 0; q < 6; ++q) {
        double fx = b.x + wps[2 * q], fy = b.y + wps[2 * q + 1];
        if (fx > VIEW_F || fx < -VIEW_R || std::fabs(fy) > YH) break;
        pts.emplace_back((int)((YH - fy) * sx2), (int)((VIEW_F - fx) * sy2));
      }
      cv::polylines(bev, std::vector<std::vector<cv::Point>>{pts}, false, col,
                    1);
    }
  }
  // ---- 2D 'obs' (unknown obstacle, class 0) -> BEV via predicted depth ---
  if (UNK2D) {
    std::vector<double> dm(DBINS);
    for (int q = 0; q < DBINS; ++q)
      dm[q] = std::exp(std::log(1.0) + (std::log(79.75) - std::log(1.0)) *
                                          (DBINS > 1 ? (double)q / (DBINS - 1) : 0));
    struct P { double x, y, sc, d; };
    std::vector<P> cand;
    // 期待値深度 (depth_mean, fp16 [1,N,h,w]) があれば優先 (2026-09-07)
    const TensorView* dmv = out.count("depth_mean") ? &out.at("depth_mean") : nullptr;
    int dmH = depH, dmW = depW;
    if (dmv) {
      dmH = (int)dmv->shape[dmv->shape.size() - 2];
      dmW = (int)dmv->shape[dmv->shape.size() - 1];
    }
    auto depthAt = [&](int i, int v, int u) -> double {
      if (dmv) {
        size_t k = (size_t)i * dmH * dmW + (size_t)v * dmW + u;
        if (dmv->dtype == nvinfer1::DataType::kHALF)
          return halfToFloat(((const uint16_t*)dmv->ptr)[k]);
        return ((const float*)dmv->ptr)[k];
      }
      return dm[std::min((int)dep[(size_t)i * depH * depW + (size_t)v * depW + u], DBINS - 1)];
    };
    for (int i = 0; i < N && i < (int)b2d.size(); ++i) {
      const float* Kc = K + (size_t)i * 9;
      const float* T = Tc + (size_t)i * 16;
      static const double GROUND_Z = envd("METEOR_GROUND_Z", 0.0);
      for (const auto& b : b2d[i]) {
        if (b.cls != 0 || b.sc < TH_UNK) continue;
        double pe[3] = {0, 0, GROUND_Z}, d = 0;
        double gx, gy;
        if (groundPoint(b.cx, b.cy + 0.5 * b.h, Kc, T, GROUND_Z, gx, gy) &&
            std::hypot(gx, gy) > 1.5 && std::hypot(gx, gy) < 60.0) {
          pe[0] = gx; pe[1] = gy; d = std::hypot(gx, gy);   // 主: 接地点の幾何 (2026-09-08)
        } else {
          int u = std::clamp((int)(b.cx / 768.0 * dmW), 0, dmW - 1);
          int v = std::clamp((int)((b.cy + 0.35 * b.h) / 432.0 * dmH), 0, dmH - 1);
          std::vector<double> win;
          for (int vv = std::max(0, v - 2); vv < std::min(dmH, v + 3); ++vv)
            for (int uu = std::max(0, u - 2); uu < std::min(dmW, u + 3); ++uu)
              win.push_back(depthAt(i, vv, uu));
          std::sort(win.begin(), win.end());
          d = win[win.size() / 2];
          if (!(1.5 < d && d < 50.0)) continue;
          double pc[3] = {(b.cx - Kc[2]) / Kc[0] * d, (b.cy - Kc[5]) / Kc[4] * d, d};
          double q[3] = {pc[0] - T[3], pc[1] - T[7], pc[2] - T[11]};
          for (int r = 0; r < 3; ++r)
            pe[r] = T[0 * 4 + r] * q[0] + T[1 * 4 + r] * q[1] + T[2 * 4 + r] * q[2];
        }
        if (std::fabs(pe[0]) > 60 || std::fabs(pe[1]) > 25) continue;
        cand.push_back({pe[0], pe[1], b.sc, d});
      }
    }
    std::sort(cand.begin(), cand.end(),
              [](const P& a, const P& b) { return a.sc > b.sc; });
    std::vector<P> kept;
    for (const auto& c : cand) {
      bool far = true;
      for (const auto& k : kept)
        if ((c.x - k.x) * (c.x - k.x) + (c.y - k.y) * (c.y - k.y) <= 2.25) {
          far = false;
          break;
        }
      if (far) kept.push_back(c);
    }
    for (const auto& c : kept) {
      if (c.x > VIEW_F || c.x < -VIEW_R || std::fabs(c.y) > YH) continue;
      cv::Point q((int)((YH - c.y) * sx2), (int)((VIEW_F - c.x) * sy2));
      cv::circle(bev, q, 4, {255, 255, 255}, -1);      // 白の小さな丸のみ — 名前・距離は描かない (2026-09-07 指示)
      cv::circle(bev, q, 5, {40, 40, 40}, 1);
    }
  }
  // LiDAR input overlay (orin_render 2026-09-08): occupied pillar cells as pale dots
  if (lidarBev) {
    const int LH = 400, LW = 250;
    double resL = (XF + XR) / LH;
    int r0l = std::max(0, (int)((XF - VIEW_F) / resL)), r1l = std::min(LH, (int)((XF + VIEW_R) / resL));
    int c0l = std::max(0, (int)((50.0 - YH) / (100.0 / LW))), c1l = std::min(LW, (int)((50.0 + YH) / (100.0 / LW)));
    double sy = (double)bev.rows / std::max(r1l - r0l, 1), sx = (double)bev.cols / std::max(c1l - c0l, 1);
    for (int r = r0l; r < r1l; ++r)
      for (int c = c0l; c < c1l; ++c) {
        // ch = (log-count, max z, mean z, occupancy): occupied and max z > 0.3 m (obstacles, not road returns)
        static const double ZMIN = envd("METEOR_LIDAR_ZMIN", 0.3);
        double occ = lidarBev[(size_t)3 * LH * LW + (size_t)r * LW + c];
        double mz = lidarBev[(size_t)1 * LH * LW + (size_t)r * LW + c];
        if (occ > 0 && mz > ZMIN)
          cv::circle(bev, {(int)((c - c0l + 0.5) * sx), (int)((r - r0l + 0.5) * sy)}, 1, {170, 200, 110}, -1);
      }
  }
  // risk heat map (orin_render: +-40 m TURBO, alpha 0.55 * risk)
  if (out.count("risk")) {
    const auto& rv = out.at("risk");
    int rh = (int)rv.shape[rv.shape.size() - 2], rw = (int)rv.shape[rv.shape.size() - 1];
    cv::Mat rm(rh, rw, CV_32F);
    for (int r = 0; r < rh; ++r)
      for (int c = 0; c < rw; ++c) {
        size_t k = (size_t)r * rw + c;
        float x = rv.dtype == nvinfer1::DataType::kHALF ? halfToFloat(((const uint16_t*)rv.ptr)[k])
                                                         : ((const float*)rv.ptr)[k];
        rm.at<float>(r, c) = 1.0f / (1.0f + std::exp(-x));
      }
    double SPAN = VIEW_F + VIEW_R;
    int BH2_ = bev.rows, BW2_ = bev.cols;
    int rmH = (int)(BH2_ * (40.0 + VIEW_R) / SPAN);
    cv::Mat rmr;
    cv::resize(rm, rmr, {BW2_, rmH}, 0, 0, cv::INTER_LINEAR);
    int y0r = (int)(BH2_ * (VIEW_F - 40.0) / SPAN);
    int y1r = std::min(BH2_, y0r + rmr.rows);
    cv::Mat r8;
    rmr.rowRange(0, y1r - y0r).convertTo(r8, CV_8U, 255.0);
    cv::Mat heat;
    cv::applyColorMap(r8, heat, cv::COLORMAP_TURBO);
    for (int r = 0; r < y1r - y0r; ++r) {
      cv::Vec3b* bp = bev.ptr<cv::Vec3b>(y0r + r);
      const cv::Vec3b* hp = heat.ptr<cv::Vec3b>(r);
      const float* ap = rmr.ptr<float>(r);
      for (int c = 0; c < BW2_; ++c) {
        float a = std::min(1.0f, std::max(0.0f, (float)(ap[c] * RISK_GAIN))) * 0.55f;
        for (int k = 0; k < 3; ++k)
          bp[c][k] = (uint8_t)(bp[c][k] * (1 - a) + hp[c][k] * a);
      }
    }
  }
  // E2E path with waypoint dots (orin_render 2026-08-24 spec: pure green,
  // width 2, radius-3 filled dots)
  std::vector<cv::Point> pts = {{(int)(YH * sx2), cy0}};
  for (int q = 0; q < 6; ++q) {
    double x = ego[kMode * 12 + 2 * q], y = ego[kMode * 12 + 2 * q + 1];
    pts.emplace_back((int)((YH - y) * sx2), (int)((VIEW_F - x) * sy2));
  }
  cv::polylines(bev, std::vector<std::vector<cv::Point>>{pts}, false,
                {0, 255, 0}, 2);
  for (size_t q = 1; q < pts.size(); ++q)
    cv::circle(bev, pts[q], 3, {0, 255, 0}, -1);
  // HUD
  double stDeg = sel[12] * 180.0 / M_PI;
  double acc = sel[13];
  double brk = 1.0 / (1.0 + std::exp(-(double)sel[14]));
  char line[64];
  std::vector<std::string> hud;
  std::snprintf(line, sizeof(line), "v0 %5.1f km/h", v0 * 3.6);
  hud.push_back(line);
  std::snprintf(line, sizeof(line), "steer %+6.1f deg", stDeg);
  hud.push_back(line);
  std::snprintf(line, sizeof(line), "accel %+5.2f m/s2", acc);
  hud.push_back(line);
  std::snprintf(line, sizeof(line), brk > 0.5 ? "BRAKE %.2f" : "brake %.2f",
                brk);
  hud.push_back(line);
  std::snprintf(line, sizeof(line), "mode p=%.2f", pr[kMode]);
  hud.push_back(line);
  for (size_t li = 0; li < hud.size(); ++li)
    cv::putText(bev, hud[li], {6, BH2 - 98 + 20 * (int)li},
                cv::FONT_HERSHEY_SIMPLEX, 0.55, {230, 230, 230}, 1,
                cv::LINE_AA);
  cv::putText(bev, "gray box = stationary", {6, BH2 - 6},
              cv::FONT_HERSHEY_SIMPLEX, 0.42, {160, 160, 160}, 1, cv::LINE_AA);
  int xb0 = 8 + 4 * (CW + 6) + 6;
  int bwFit = VW - xb0 - 8;
  if (bev.cols > bwFit)
    cv::resize(bev, bev, {bwFit, (int)((double)BH2 * bwFit / BW2)});
  int bh = std::min(bev.rows, VH - 40);
  bev(cv::Rect(0, 0, bev.cols, bh))
      .copyTo(canvas(cv::Rect(xb0, 34, bev.cols, bh)));
  // title
  char title[160];
  static const std::string TRT_TAG = trtVersionTag();
  std::snprintf(title, sizeof(title),
                "METEOR - %s - TensorRT %s - on-device "
                "inference+render - %.0f ms infer - %.1f FPS%s",
                GPU_TAG.c_str(), TRT_TAG.c_str(), dt, fps_now,
                lidarBev ? " - LiDAR ON" : "");
  cv::putText(canvas, title, {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.62,
              {240, 240, 240}, 2, cv::LINE_AA);
  return canvas;
}
