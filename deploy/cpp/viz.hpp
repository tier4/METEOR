// C++ port of deploy/viz_np.py: palettes + the three camera-overlay helpers
// (2D boxes, 3D wireframes with near-plane clipping, E2E path ribbon).
#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

#include "decode.hpp"

// viz_np.PALETTE (stored RGB there, converted to BGR at draw time -- these
// are already BGR)
extern const cv::Vec3b PALETTE_BGR[9];

// Per-camera 10-class 2D boxes (cls, score, cx, cy, w, h in 768x432 px)
void draw_boxes2d(cv::Mat& img, const std::vector<Box2D>& blist, int cw,
                  int ch);

// Project decoded BEV boxes (ego frame) into a camera and draw 3D wireframes
// with distance labels; edges clipped at the near plane. K9 row-major 3x3,
// Tce16 row-major 4x4 (T_cam_ego).
void draw_boxes_on_rgb(cv::Mat& img, const std::vector<Box3D>& det,
                       const float* K9, const float* Tce16, int cw, int ch);

// Predicted trajectory as a vehicle-width ground ribbon; EMA-smoothed across
// frames (shared state, mutex-guarded), retracting near a stop.
void draw_path_ribbon(cv::Mat& img, const float* egoSel /* >= 12 floats */,
                      const float* K9, const float* Tce16, int cw, int ch);
