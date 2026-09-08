// C++ port of deploy/runtime.py class MeteorRT (2026-09-06 refresh for the
// v147c3Zg generation: no-hist engines, lift plugin, CUDA Graph, zero-copy
// pinned input slots).
//
// Wraps a serialized TensorRT engine behind a per-frame API that owns the
// temporal recurrence:
//   * device-resident raw_bev ring (size max(HIST_OFFS) = 14) spliced into
//     hist_bev with D2D copies -- never staged through the host. Engines
//     exported with --no-hist have no hist tensors: the ring is skipped.
//   * warp thetas derived from consecutive ego poses (make_warp_theta math)
//   * multi-slot pinned host output buffers so a consumer can hold frame t's
//     outputs while frame t+1 infers (zero-copy hand-off)
//   * pinned host INPUT slots: the loader writes the uint8 image stack
//     straight into pinned memory and infer() issues one async H2D from it
//     (runtime.py pinned_input_slots, -4.9 ms on Orin)
//   * skip_outputs: output names left on the device (no D2H)
//   * dtype-adaptive feeds: the uint8-in engines take imgs as uint8
//   * METEOR_PLUGIN_SO: dlopen'd before deserialization (lift plugin)
//   * METEOR_CUDAGRAPH=1: enqueueV3 captured into a CUDA graph after two
//     warm-up frames (needs an engine built with --max-aux-streams 0)
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

struct TensorView {
  const void* ptr = nullptr;
  nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  std::vector<int64_t> shape;  // engine shape, batch included
};
using OutMap = std::map<std::string, TensorView>;

size_t trtDtypeSize(nvinfer1::DataType t);

// runtime.py make_warp_theta: global (x,y,yaw) of previous and current frame
// -> 2x3 affine theta. bev_h/bev_w MUST match the engine's hist_bev grid.
void make_warp_theta(const float* posePrev, const float* poseCur, float th[6],
                     int bev_h = 800, int bev_w = 500);

class MeteorRT {
 public:
  MeteorRT(const std::string& enginePath,
           const std::set<std::string>& skipOutputs = {}, int nOutSlots = 1,
           int nInSlots = 3);
  ~MeteorRT();
  MeteorRT(const MeteorRT&) = delete;
  MeteorRT& operator=(const MeteorRT&) = delete;

  bool imgsAreU8() const { return imgsU8_; }
  bool hasHist() const { return hasHist_; }
  size_t elemsOf(const std::string& nm) const { return count_.at(nm); }
  size_t bytesOf(const std::string& nm) const { return bytes_.at(nm); }
  const std::vector<int64_t>& shapeOf(const std::string& nm) const {
    return shapes_.at(nm);
  }
  bool has(const std::string& nm) const { return shapes_.count(nm) != 0; }
  int nOutSlots() const { return (int)hostSlots_.size(); }
  int nInSlots() const { return (int)inSlots_.size(); }
  // pinned host buffer of imgs' size; fill it and pass it to infer() for the
  // zero-copy path (any other pointer is staged through slot 0)
  void* inputSlot(int i) { return inSlots_.at((size_t)i); }
  bool graphActive() const { return gexec_ != nullptr; }

  // imgsHost: contiguous [1,N,3,432,768] matching the engine's imgs dtype
  // (uint8 or float32); K [1,N,3,3] f32; Tc = T_cam_ego [1,N,4,4] f32;
  // pose3: (x, y, yaw) or nullptr. Returns views into host slot `outSlot`,
  // valid until that slot is reused.
  // lidarBev: [1,4,400,250] fp32 pillar raster or nullptr (= zeros = camera-only);
  // only used when the engine was exported with --with-lidar.
  OutMap infer(const void* imgsHost, const float* K, const float* Tc,
               float v0, const float* pose3, int outSlot,
               const float* lidarBev = nullptr);
  bool hasLidar() const { return shapes_.count("lidar_bev") != 0; }

 private:
  static constexpr int HIST_OFFS[3] = {2, 6, 14};
  static constexpr int RING_N = 14;  // max(HIST_OFFS)

  void launch();

  nvinfer1::IRuntime* rt_ = nullptr;
  nvinfer1::ICudaEngine* eng_ = nullptr;
  nvinfer1::IExecutionContext* ctx_ = nullptr;
  cudaStream_t stream_{};
  void* pluginHandle_ = nullptr;

  std::map<std::string, std::vector<int64_t>> shapes_;
  std::map<std::string, nvinfer1::DataType> dtype_;
  std::map<std::string, size_t> count_, bytes_;
  std::map<std::string, void*> dev_;
  std::map<std::string, void*> hostBase_;              // pinned, slot 0
  std::vector<std::map<std::string, void*>> hostSlots_;
  std::vector<void*> inSlots_;         // pinned imgs slots (zero-copy)
  std::vector<std::string> outNames_;  // D2H'd outputs, OUTPUTS order
  std::set<std::string> skip_;
  bool imgsU8_ = false;
  bool hasHist_ = false;

  // CUDA graph
  bool graphWanted_ = false;
  int warm_ = 0;
  cudaGraphExec_t gexec_ = nullptr;

  // device-resident temporal ring
  size_t slotBytes_ = 0;
  std::vector<void*> ring_;
  std::array<std::array<float, 3>, RING_N> ringPose_{};
  std::array<bool, RING_N> ringPoseOk_{};
  std::array<long, RING_N> ringT_{};
  long t_ = 0;
};
