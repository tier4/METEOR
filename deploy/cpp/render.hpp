// C++ port of deploy/orin_render.py compose_frame: the full 1920x1080 frame
// layout (camera tiles + optional depth tiles in CAM8 order, BEV pane, HUD).
// 2026-09-06: N-camera (7 or 8, taken from the engine), env switches shared
// with the Python renderer (METEOR_TH2D / METEOR_2D_HIDE /
// METEOR_SEG2D_OVERLAY / METEOR_DEPTH_PANEL / METEOR_DEPTH_BINS /
// METEOR_UNK2D), 2D "obs" -> BEV via predicted depth.
#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "meteor_rt.hpp"

// model input order (bevlane/dataset.py CAMS; 8th = CAM_BACK_NARROW)
extern const char* CAM_IN8[8];

// One 1920x1080 canvas from one frame's raw images + engine outputs.
// raw: N BGR images in model input order; K Nx3x3 row-major; Tc Nx4x4
// (T_cam_ego).
// pose3: global (x, y, yaw) of this frame or nullptr (drives the BEV seg
// temporal fusion exactly like orin_render.seg_fuse_logit / seg_fuse_np).
cv::Mat compose_frame(const std::vector<cv::Mat>& raw,
                      const std::vector<std::string>& cams, const float* K,
                      const float* Tc, float v0, const OutMap& out, double dt,
                      double fps_now, const float* pose3 = nullptr,
                      const float* lidarBev = nullptr /* [4,400,250] or null */);
