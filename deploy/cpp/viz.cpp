#include "viz.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

// viz_np.PALETTE is RGB; every use site there flips to BGR, so store BGR
const cv::Vec3b PALETTE_BGR[9] = {
    {0, 0, 0},      {90, 90, 90},   {160, 90, 140},
    {200, 200, 0},  {255, 255, 255}, {40, 40, 255},
    {0, 140, 255},  {60, 220, 240}, {140, 60, 40}};

namespace {

// DET10_PAL is RGB in viz_np; draw_boxes2d flips per box -- store BGR
const cv::Scalar DET10_PAL_BGR[10] = {
    {255, 255, 255}, {255, 0, 0},   {165, 160, 0}, {200, 0, 100},
    {0, 255, 128},   {0, 255, 255}, {32, 0, 255},  {0, 255, 128},
    {255, 0, 250},   {0, 255, 0}};
const char* DET10_ABBR[10] = {"obs", "car", "trk", "bus", "bcy",
                              "mcy", "ped", "pnt", "tl",  "ts"};

const int BOX_EDGES[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                              {4, 5}, {5, 6}, {6, 7}, {7, 4},
                              {0, 4}, {1, 5}, {2, 6}, {3, 7}};

}  // namespace

void draw_boxes2d(cv::Mat& img, const std::vector<Box2D>& blist, int cw,
                  int ch) {
  double sx = cw / 768.0, sy = ch / 432.0;
  for (const auto& b : blist) {
    cv::Scalar c = DET10_PAL_BGR[b.cls];
    int x1 = (int)((b.cx - b.w / 2) * sx), y1 = (int)((b.cy - b.h / 2) * sy);
    int x2 = (int)((b.cx + b.w / 2) * sx), y2 = (int)((b.cy + b.h / 2) * sy);
    cv::rectangle(img, {x1, y1}, {x2, y2}, c, 2);
    char tag[16];
    std::snprintf(tag, sizeof(tag), "%s%d", DET10_ABBR[b.cls],
                  (int)(b.sc * 100));
    int base = 0;
    cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.38, 1,
                                  &base);
    int ty = std::max(y1, ts.height + 3);
    cv::rectangle(img, {x1, ty - ts.height - 3}, {x1 + ts.width + 2, ty + 1},
                  c, -1);
    cv::putText(img, tag, {x1 + 1, ty - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.38,
                {0, 0, 0}, 1, cv::LINE_AA);
  }
}

void draw_boxes_on_rgb(cv::Mat& img, const std::vector<Box3D>& det,
                       const float* K9, const float* Tce16, int cw, int ch) {
  const double W0 = 768.0, H0 = 432.0;
  double sx = cw / W0, sy = ch / H0;
  const double EPS = 0.25;
  auto proj = [&](const double p[3]) {
    return cv::Point((int)((K9[0] * p[0] / p[2] + K9[2]) * sx),
                     (int)((K9[4] * p[1] / p[2] + K9[5]) * sy));
  };
  for (const auto& b : det) {
    double hgt = b.cls == 0 ? 1.6 : 1.7;
    double cb = std::cos((double)b.yaw), sb = std::sin((double)b.yaw);
    double cors[8][3];
    const double off[4][2] = {{b.l / 2.0, b.w / 2.0},
                              {b.l / 2.0, -b.w / 2.0},
                              {-b.l / 2.0, -b.w / 2.0},
                              {-b.l / 2.0, b.w / 2.0}};
    for (int i = 0; i < 4; ++i) {
      cors[i][0] = b.x + off[i][0] * cb - off[i][1] * sb;
      cors[i][1] = b.y + off[i][0] * sb + off[i][1] * cb;
      cors[i][2] = 0.0;
      cors[i + 4][0] = cors[i][0];
      cors[i + 4][1] = cors[i][1];
      cors[i + 4][2] = hgt;
    }
    double pc[8][3];
    int nfront = 0;
    for (int i = 0; i < 8; ++i) {
      for (int r = 0; r < 3; ++r)
        pc[i][r] = Tce16[r * 4 + 0] * cors[i][0] +
                   Tce16[r * 4 + 1] * cors[i][1] +
                   Tce16[r * 4 + 2] * cors[i][2] + Tce16[r * 4 + 3];
      if (pc[i][2] > EPS) ++nfront;
    }
    if (nfront == 0) continue;  // entirely behind the camera
    cv::Scalar col = b.cls == 0 ? cv::Scalar(0, 215, 255)
                                : cv::Scalar(255, 0, 255);
    std::vector<cv::Point> vis;
    for (const auto& e : BOX_EDGES) {
      double pa[3], pb[3];
      std::copy(pc[e[0]], pc[e[0]] + 3, pa);
      std::copy(pc[e[1]], pc[e[1]] + 3, pb);
      double za = pa[2], zb = pb[2];
      if (za < EPS && zb < EPS) continue;
      if (za < EPS || zb < EPS) {  // clip the edge at the near plane
        double t = (EPS - za) / (zb - za);
        double pclip[3];
        for (int r = 0; r < 3; ++r) pclip[r] = pa[r] + t * (pb[r] - pa[r]);
        if (za < EPS)
          std::copy(pclip, pclip + 3, pa);
        else
          std::copy(pclip, pclip + 3, pb);
      }
      cv::Point A = proj(pa), B = proj(pb);
      if (std::abs(A.x) > cw * 8 || std::abs(B.x) > cw * 8 ||
          std::abs(A.y) > ch * 8 || std::abs(B.y) > ch * 8)
        continue;
      cv::line(img, A, B, col, 1, cv::LINE_AA);
      vis.push_back(A);
      vis.push_back(B);
    }
    if (!vis.empty()) {
      int minU = vis[0].x, maxU = vis[0].x, minV = vis[0].y, maxV = vis[0].y;
      for (const auto& p : vis) {
        minU = std::min(minU, p.x);
        maxU = std::max(maxU, p.x);
        minV = std::min(minV, p.y);
        maxV = std::max(maxV, p.y);
      }
      if (maxU >= 0 && minU < cw && maxV >= 0 && minV < ch) {
        double dist = std::hypot((double)b.x, (double)b.y);
        char lbl[16];
        std::snprintf(lbl, sizeof(lbl), "%.0fm", dist);
        cv::putText(img, lbl, {std::max(0, minU), std::max(12, minV - 4)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv::LINE_AA);
      }
    }
  }
}

namespace {

// np.interp (t nondecreasing, clamped at the ends)
double interp1(double x, const std::vector<double>& xs,
               const std::vector<double>& ys) {
  if (x <= xs.front()) return ys.front();
  if (x >= xs.back()) return ys.back();
  size_t i = std::upper_bound(xs.begin(), xs.end(), x) - xs.begin();
  // xs[i-1] <= x < xs[i]
  double x0 = xs[i - 1], x1 = xs[i];
  if (x1 <= x0) return ys[i];
  double t = (x - x0) / (x1 - x0);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

// np.gradient with unit spacing
std::vector<double> gradient(const std::vector<double>& a) {
  size_t n = a.size();
  std::vector<double> g(n);
  if (n == 1) {
    g[0] = 0;
    return g;
  }
  g[0] = a[1] - a[0];
  g[n - 1] = a[n - 1] - a[n - 2];
  for (size_t i = 1; i + 1 < n; ++i) g[i] = (a[i + 1] - a[i - 1]) / 2.0;
  return g;
}

std::mutex g_ribbonMx;
bool g_ribbonInit = false;
double g_ribbonWps[7][2];

}  // namespace

void draw_path_ribbon(cv::Mat& img, const float* egoSel, const float* K9,
                      const float* Tce16, int cw, int ch) {
  const double W0 = 768.0, H0 = 432.0;
  const double half_w = 0.9, alpha = 0.38;
  const cv::Scalar color(60, 255, 120);
  double wraw[7][2] = {{0.0, 0.0}};
  bool finite = true;
  for (int i = 0; i < 6; ++i) {
    wraw[i + 1][0] = egoSel[2 * i];
    wraw[i + 1][1] = egoSel[2 * i + 1];
    if (!std::isfinite(wraw[i + 1][0]) || !std::isfinite(wraw[i + 1][1]))
      finite = false;
  }
  if (!finite) return;  // NaN frame: draw nothing, keep the EMA state
  std::vector<std::array<double, 2>> wps(7);
  {
    std::lock_guard<std::mutex> lk(g_ribbonMx);
    if (!g_ribbonInit) {
      std::memcpy(g_ribbonWps, wraw, sizeof(wraw));
      g_ribbonInit = true;
    } else {
      for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 2; ++j)
          g_ribbonWps[i][j] = 0.55 * g_ribbonWps[i][j] + 0.45 * wraw[i][j];
    }
    for (int i = 0; i < 7; ++i) wps[i] = {g_ribbonWps[i][0], g_ribbonWps[i][1]};
  }
  double trav = std::hypot(wps[6][0] - wps[0][0], wps[6][1] - wps[0][1]);
  double s = std::min(std::max(trav / 3.0, 0.0), 1.0);
  double dispLen = std::max(trav, 8.0 * s * s * (3 - 2 * s));
  if (dispLen < 0.6) return;
  auto cumlen = [](const std::vector<std::array<double, 2>>& w) {
    std::vector<double> t(w.size());
    t[0] = 0;
    for (size_t i = 1; i < w.size(); ++i)
      t[i] = t[i - 1] + std::hypot(w[i][0] - w[i - 1][0], w[i][1] - w[i - 1][1]);
    return t;
  };
  std::vector<double> t = cumlen(wps);
  if (t.back() < dispLen) {  // extend along the last heading
    size_t n = wps.size();
    double d0 = wps[n - 1][0] - wps[n - 3][0];
    double d1 = wps[n - 1][1] - wps[n - 3][1];
    double nn = std::max(std::hypot(d0, d1), 1e-3);
    d0 /= nn;
    d1 /= nn;
    double ext = dispLen - t.back();
    wps.push_back({wps[n - 1][0] + d0 * ext, wps[n - 1][1] + d1 * ext});
    t = cumlen(wps);
  }
  const int NS = 48;
  std::vector<double> xs(wps.size()), ys(wps.size());
  for (size_t i = 0; i < wps.size(); ++i) {
    xs[i] = wps[i][0];
    ys[i] = wps[i][1];
  }
  double tmax = std::min(t.back(), dispLen);
  std::vector<double> px(NS), py(NS);
  for (int i = 0; i < NS; ++i) {
    double tt = tmax * i / (NS - 1);
    px[i] = interp1(tt, t, xs);
    py[i] = interp1(tt, t, ys);
  }
  std::vector<double> gx = gradient(px), gy = gradient(py);
  double sx = cw / W0, sy = ch / H0;
  auto proj = [&](double x, double y, double& u, double& v, bool& ok) {
    double c0 = Tce16[0] * x + Tce16[1] * y + Tce16[3];
    double c1 = Tce16[4] * x + Tce16[5] * y + Tce16[7];
    double c2 = Tce16[8] * x + Tce16[9] * y + Tce16[11];
    ok = c2 > 0.3;
    double z = std::max(c2, 0.3);
    u = (K9[0] * c0 / z + K9[2]) * sx;
    v = (K9[4] * c1 / z + K9[5]) * sy;
  };
  std::vector<cv::Point2d> L, R;
  for (int i = 0; i < NS; ++i) {
    double th = std::atan2(gy[i], gx[i]);
    double lu, lv, ru, rv;
    bool okl, okr;
    proj(px[i] - half_w * std::sin(th), py[i] + half_w * std::cos(th), lu, lv,
         okl);
    proj(px[i] + half_w * std::sin(th), py[i] - half_w * std::cos(th), ru, rv,
         okr);
    if (okl && okr) {
      L.emplace_back(lu, lv);
      R.emplace_back(ru, rv);
    }
  }
  if ((int)L.size() < 3) return;
  std::vector<cv::Point> poly, lpts, rpts;
  for (const auto& p : L) lpts.emplace_back((int)p.x, (int)p.y);
  for (auto it = R.rbegin(); it != R.rend(); ++it)
    rpts.emplace_back((int)it->x, (int)it->y);
  for (const auto& p : lpts)
    poly.emplace_back(std::max(-cw, std::min(2 * cw, p.x)),
                      std::max(-ch, std::min(2 * ch, p.y)));
  for (const auto& p : rpts)
    poly.emplace_back(std::max(-cw, std::min(2 * cw, p.x)),
                      std::max(-ch, std::min(2 * ch, p.y)));
  cv::Mat ov = img.clone();
  cv::fillPoly(ov, std::vector<std::vector<cv::Point>>{poly}, color);
  cv::polylines(ov, std::vector<std::vector<cv::Point>>{lpts}, false, color, 2);
  cv::polylines(ov, std::vector<std::vector<cv::Point>>{rpts}, false, color, 2);
  cv::addWeighted(img, 1 - alpha, ov, alpha, 0, img);
}
