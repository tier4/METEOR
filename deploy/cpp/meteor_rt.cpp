#include "meteor_rt.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#if __has_include(<NvInferPlugin.h>)
#include <NvInferPlugin.h>
#define METEOR_HAVE_PLUGIN_H 1
#endif

namespace {

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
} gLogger;

std::vector<char> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// runtime.py OUTPUTS -- fixed D2H order (+ the no-hist era extras)
const char* OUTPUTS[] = {
    "lane", "depth", "seg2d", "hm", "reg",
    "hm2d_s0", "hm2d_s1", "hm2d_s2",
    "reg2d_s0", "reg2d_s1", "reg2d_s2",
    "ego", "occ", "traj", "stationary", "tl", "risk", "flow",
    "lg_pts", "lg_meta", "lg_adj", "unk", "lane_logit", "depth_mean", "raw_bev"};

}  // namespace

size_t trtDtypeSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kUINT8: return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL: return 1;
    case nvinfer1::DataType::kINT64: return 8;
    default: return 1;  // kFP8 etc
  }
}

void make_warp_theta(const float* posePrev, const float* poseCur, float th[6],
                     int bev_h, int bev_w) {
  double cp = std::cos((double)posePrev[2]), sp = std::sin((double)posePrev[2]);
  double dx = (double)poseCur[0] - posePrev[0];
  double dy = (double)poseCur[1] - posePrev[1];
  double tx = cp * dx + sp * dy;
  double ty = -sp * dx + cp * dy;
  double dyaw = (double)poseCur[2] - posePrev[2];
  double a = 0.1 * (bev_h - 1);
  double b = 0.1 * (bev_w - 1);
  double cd = std::cos(dyaw), sd = std::sin(dyaw);
  double Cx = cd * (80 - a) - sd * (50 - b) + tx;
  double Cy = sd * (80 - a) + cd * (50 - b) + ty;
  th[0] = (float)cd;
  th[1] = (float)(a * sd / b);
  th[2] = (float)(((50 - b) - Cy) / b);
  th[3] = (float)(-(b / a) * sd);
  th[4] = (float)cd;
  th[5] = (float)(((80 - a) - Cx) / a);
}

constexpr int MeteorRT::HIST_OFFS[3];

MeteorRT::MeteorRT(const std::string& enginePath,
                   const std::set<std::string>& skipOutputs, int nOutSlots,
                   int nInSlots)
    : skip_(skipOutputs) {
  // lift plugin: must be registered before deserialization (runtime.py does
  // ctypes.CDLL(RTLD_GLOBAL) + init_libnvinfer_plugins)
  if (const char* so = std::getenv("METEOR_PLUGIN_SO")) {
    pluginHandle_ = dlopen(so, RTLD_NOW | RTLD_GLOBAL);
    if (!pluginHandle_)
      throw std::runtime_error(std::string("dlopen plugin: ") + dlerror());
#ifdef METEOR_HAVE_PLUGIN_H
    initLibNvInferPlugins(&gLogger, "");
#endif
    std::cerr << "[rt] plugin loaded: " << so << "\n";
  }
  graphWanted_ = std::getenv("METEOR_CUDAGRAPH") &&
                 std::string(std::getenv("METEOR_CUDAGRAPH")) == "1";
  auto blob = readFile(enginePath);
  rt_ = nvinfer1::createInferRuntime(gLogger);
  rt_->setEngineHostCodeAllowed(true);  // version-compatible engines
  eng_ = rt_->deserializeCudaEngine(blob.data(), blob.size());
  if (!eng_) throw std::runtime_error("engine deserialization failed");
  ctx_ = eng_->createExecutionContext();
  ck(cudaStreamCreate(&stream_), "stream");
  for (int i = 0; i < eng_->getNbIOTensors(); ++i) {
    const char* nm = eng_->getIOTensorName(i);
    auto d = eng_->getTensorShape(nm);
    std::vector<int64_t> shp(d.d, d.d + d.nbDims);
    size_t n = 1;
    for (auto v : shp) n *= (size_t)std::max<int64_t>(v, 1);
    auto dt = eng_->getTensorDataType(nm);
    size_t nb = n * trtDtypeSize(dt);
    shapes_[nm] = shp;
    dtype_[nm] = dt;
    count_[nm] = n;
    bytes_[nm] = nb;
    void* h = nullptr;
    ck(cudaHostAlloc(&h, nb, cudaHostAllocDefault), "hostAlloc");
    std::memset(h, 0, nb);
    hostBase_[nm] = h;
    void* dv = nullptr;
    ck(cudaMalloc(&dv, nb), "malloc");
    ck(cudaMemset(dv, 0, nb), "memset");  // optional inputs stay zero
    dev_[nm] = dv;
    ctx_->setTensorAddress(nm, dv);
  }
  imgsU8_ = dtype_.count("imgs") &&
            (dtype_["imgs"] == nvinfer1::DataType::kUINT8 ||
             dtype_["imgs"] == nvinfer1::DataType::kINT8);
  hasHist_ = shapes_.count("hist_bev") && shapes_.count("hist_theta");
  for (const char* nm : OUTPUTS)
    if (shapes_.count(nm) && std::string(nm) != "raw_bev" && !skip_.count(nm))
      outNames_.push_back(nm);
  // slot 0 = the base host buffers; extra slots only for the read outputs
  hostSlots_.resize(std::max(1, nOutSlots));
  hostSlots_[0] = hostBase_;
  for (size_t s = 1; s < hostSlots_.size(); ++s) {
    hostSlots_[s] = hostBase_;
    for (auto& nm : outNames_) {
      void* h = nullptr;
      ck(cudaHostAlloc(&h, bytes_[nm], cudaHostAllocDefault), "hostAlloc");
      hostSlots_[s][nm] = h;
    }
  }
  for (int i = 0; i < std::max(0, nInSlots); ++i) {
    void* h = nullptr;
    ck(cudaHostAlloc(&h, bytes_.at("imgs"), cudaHostAllocDefault),
       "hostAlloc in-slot");
    inSlots_.push_back(h);
  }
  ringT_.fill(-1);
  ringPoseOk_.fill(false);
  if (hasHist_ && shapes_.count("raw_bev")) {
    slotBytes_ = bytes_["raw_bev"];
    ring_.resize(RING_N);
    for (auto& r : ring_) ck(cudaMalloc(&r, slotBytes_), "ring malloc");
  }
  std::cerr << "[rt] engine " << enginePath << " io=" << shapes_.size()
            << " imgs=" << (imgsU8_ ? "uint8" : "fp32")
            << " hist=" << (hasHist_ ? "yes" : "no (baked zero history)")
            << " out_slots=" << hostSlots_.size()
            << " in_slots=" << inSlots_.size()
            << " graph=" << (graphWanted_ ? "wanted" : "off") << "\n";
}

MeteorRT::~MeteorRT() {
  if (gexec_) cudaGraphExecDestroy(gexec_);
  for (auto& kv : dev_) cudaFree(kv.second);
  for (auto& r : ring_) cudaFree(r);
  for (auto& kv : hostBase_) cudaFreeHost(kv.second);
  for (size_t s = 1; s < hostSlots_.size(); ++s)
    for (auto& nm : outNames_) cudaFreeHost(hostSlots_[s][nm]);
  for (auto* p : inSlots_) cudaFreeHost(p);
  cudaStreamDestroy(stream_);
  delete ctx_;
  delete eng_;
  delete rt_;
  if (pluginHandle_) dlclose(pluginHandle_);
}

void MeteorRT::launch() {
  if (gexec_) {
    ck(cudaGraphLaunch(gexec_, stream_), "graph launch");
    return;
  }
  if (graphWanted_ && warm_ >= 2) {
    // static shapes + fixed addresses: capture once, replay forever
    cudaGraph_t g = nullptr;
    bool ok = cudaStreamBeginCapture(stream_,
                                     cudaStreamCaptureModeThreadLocal) ==
              cudaSuccess;
    if (ok) ok = ctx_->enqueueV3(stream_);
    if (cudaStreamEndCapture(stream_, &g) != cudaSuccess) ok = false;
    if (ok && g) {
      cudaGraphExec_t ge = nullptr;
      if (cudaGraphInstantiate(&ge, g, 0) == cudaSuccess && ge) {
        gexec_ = ge;
        std::cerr << "[rt] CUDA Graph captured\n";
      }
      cudaGraphDestroy(g);
    }
    if (!gexec_) {
      cudaGetLastError();  // clear any capture error
      graphWanted_ = false;
      std::cerr << "[rt] CUDA Graph unavailable; plain enqueue\n";
      if (!ctx_->enqueueV3(stream_))
        throw std::runtime_error("enqueueV3 failed");
      return;
    }
    ck(cudaGraphLaunch(gexec_, stream_), "graph launch");
    return;
  }
  ++warm_;
  if (!ctx_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
}

OutMap MeteorRT::infer(const void* imgsHost, const float* K, const float* Tc,
                       float v0, const float* pose3, int outSlot,
                       const float* lidarBev) {
  // splice hist_bev on-device from the raw_bev ring (D2D) -- slots with no
  // history yet are zero-filled with an identity warp, exactly like training
  float ht[18];
  if (hasHist_) {
    char* hb = (char*)dev_.at("hist_bev");
    const auto& hbShape = shapes_.at("hist_bev");
    int histH = (int)hbShape[hbShape.size() - 2];
    int histW = (int)hbShape[hbShape.size() - 1];
    for (int i = 0; i < 3; ++i) {
      long ti = t_ - HIST_OFFS[i];
      int slot = ti >= 0 ? (int)(ti % RING_N) : -1;
      bool valid = !ring_.empty() && ti >= 0 && ringT_[slot] == ti &&
                   pose3 != nullptr && ringPoseOk_[slot];
      char* dst = hb + (size_t)i * slotBytes_;
      if (valid) {
        ck(cudaMemcpyAsync(dst, ring_[slot], slotBytes_,
                           cudaMemcpyDeviceToDevice, stream_), "hist d2d");
        make_warp_theta(ringPose_[slot].data(), pose3, ht + i * 6, histH,
                        histW);
      } else {
        if (slotBytes_)
          ck(cudaMemsetAsync(dst, 0, slotBytes_, stream_), "hist memset");
        const float ident[6] = {1, 0, 0, 0, 1, 0};
        std::memcpy(ht + i * 6, ident, sizeof(ident));
      }
    }
  }
  // feed inputs (cast already done by the caller for imgs)
  auto h2d = [&](const std::string& nm, const void* src, size_t nb) {
    if (!shapes_.count(nm)) return;
    std::memcpy(hostBase_.at(nm), src, nb);
    ck(cudaMemcpyAsync(dev_.at(nm), hostBase_.at(nm), nb,
                       cudaMemcpyHostToDevice, stream_), "h2d");
  };
  bool zeroCopy = std::find(inSlots_.begin(), inSlots_.end(),
                            (void*)imgsHost) != inSlots_.end();
  if (zeroCopy)
    ck(cudaMemcpyAsync(dev_.at("imgs"), imgsHost, bytes_.at("imgs"),
                       cudaMemcpyHostToDevice, stream_), "h2d imgs (pinned)");
  else
    h2d("imgs", imgsHost, bytes_.at("imgs"));
  h2d("K", K, bytes_.at("K"));
  h2d("T_cam_ego", Tc, bytes_.at("T_cam_ego"));
  h2d("v0", &v0, sizeof(float));
  if (hasHist_) h2d("hist_theta", ht, sizeof(ht));
  if (shapes_.count("lidar_bev")) {
    if (lidarBev) h2d("lidar_bev", lidarBev, bytes_.at("lidar_bev"));
    else ck(cudaMemsetAsync(dev_.at("lidar_bev"), 0, bytes_.at("lidar_bev"), stream_),
            "lidar zero");
  }
  if (shapes_.count("lidar_flag")) {
    float fl = lidarBev ? 1.0f : 0.0f;
    h2d("lidar_flag", &fl, sizeof(float));
  }
  launch();
  // store this frame's raw_bev in the device ring (D2D, no host copy)
  if (!ring_.empty()) {
    int slot = (int)(t_ % RING_N);
    ck(cudaMemcpyAsync(ring_[slot], dev_.at("raw_bev"), slotBytes_,
                       cudaMemcpyDeviceToDevice, stream_), "ring d2d");
    ringPoseOk_[slot] = pose3 != nullptr;
    if (pose3) std::copy(pose3, pose3 + 3, ringPose_[slot].begin());
    ringT_[slot] = t_;
  }
  auto& hs = hostSlots_[(size_t)outSlot % hostSlots_.size()];
  for (auto& nm : outNames_)
    ck(cudaMemcpyAsync(hs.at(nm), dev_.at(nm), bytes_.at(nm),
                       cudaMemcpyDeviceToHost, stream_), "d2h");
  ck(cudaStreamSynchronize(stream_), "sync");
  ++t_;
  OutMap out;
  for (auto& nm : outNames_)
    out[nm] = TensorView{hs.at(nm), dtype_.at(nm), shapes_.at(nm)};
  return out;
}
