// MeteorLift: TensorRT 10.x IPluginV3 plugin wrapping the fused gather-form
// frustum lift (deploy/cpp/liftbench, 0.27 ms NHWC / 1.28 ms NCHW on Orin vs
// ~16 ms for the in-engine Myelin+Scatter form).
//
//   inputs : dprob [N,D,108,192] fp16 (post-softmax depth), ctx [N,Cc,108,192]
//   output : lift BEV [1,Cc,lift_h,lift_w] fp16 (pre-upsample; the graph's
//            Resize to 600x500 stays outside)
//   formats: fp16 kLINEAR and kHWC8 accepted on every tensor independently
//            (Cc=96, D=64 are multiples of 8, so kHWC8 is pure NHWC).
//
// The (camera,cell) pair tables are baked as plugin attributes from the rig
// calibration (see dump_lift.py) -- same rig-locking bake_frustum already
// imposes on the engine. K / T_cam_ego become unused engine inputs.
#include <NvInferRuntime.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "lift_kernels.cuh"

using namespace nvinfer1;

namespace {

constexpr char const* kNAME = "MeteorLift";
constexpr char const* kVER = "1";
constexpr char const* kNS = "";

struct Tables {
  int32_t n_cams{}, depth_bins{}, cc{}, hf{}, wf{}, lift_h{}, lift_w{};
  // fused mode: input 0 is RAW depth logits (softmax inside), output is the
  // bilinearly upsampled [1, cc, out_h, out_w] BEV, fp32 when out_fp32
  // (writes the graph's raw_bev directly, no Resize / no output reformat).
  int32_t fold_softmax{0}, out_h{0}, out_w{0}, out_fp32{0};
  std::vector<int32_t> rowptr, col, cam, b0;
  std::vector<float> ix, iy, fr;
};

#define CKRT(x) do { cudaError_t e = (x); if (e != cudaSuccess) return -1; \
  } while (0)

class MeteorLiftPlugin : public IPluginV3, public IPluginV3OneCore,
                         public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  explicit MeteorLiftPlugin(std::shared_ptr<Tables> t) : t_(std::move(t)) {}
  ~MeteorLiftPlugin() override { freeDev(); }

  // ---- IPluginV3
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type)
      noexcept override {
    if (type == PluginCapabilityType::kBUILD)
      return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME)
      return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
  }
  IPluginV3* clone() noexcept override {
    return new MeteorLiftPlugin(t_);           // tables shared, dev lazily
  }

  // ---- IPluginV3OneCore
  char const* getPluginName() const noexcept override { return kNAME; }
  char const* getPluginVersion() const noexcept override { return kVER; }
  char const* getPluginNamespace() const noexcept override { return kNS; }

  // ---- IPluginV3OneBuild
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t)
      noexcept override { return 0; }
  bool supportsFormatCombination(int32_t pos,
                                 DynamicPluginTensorDesc const* inOut,
                                 int32_t nbInputs, int32_t) noexcept override {
    // fp32-linear must be accepted on every tensor: the INT8 calibration
    // graph executes the plugin with fp32 io (enqueue converts on the fly).
    auto const& d = inOut[pos].desc;
    if (d.type == DataType::kFLOAT) return d.format == TensorFormat::kLINEAR;
    if (pos == nbInputs && t_->out_fp32) return false;  // output fixed fp32
    return d.type == DataType::kHALF &&
           (d.format == TensorFormat::kLINEAR ||
            d.format == TensorFormat::kHWC8);
  }
  int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                             DataType const*, int32_t) const noexcept override {
    if (nbOutputs != 1) return -1;
    outputTypes[0] = t_->out_fp32 ? DataType::kFLOAT : DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                          DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nbOutputs, IExprBuilder& eb)
      noexcept override {
    if (nbInputs != 2 || nbOutputs != 1) return -1;
    int oh = t_->out_h > 0 ? t_->out_h : t_->lift_h;
    int ow = t_->out_w > 0 ? t_->out_w : t_->lift_w;
    outputs[0].nbDims = 4;
    outputs[0].d[0] = eb.constant(1);
    outputs[0].d[1] = eb.constant(t_->cc);
    outputs[0].d[2] = eb.constant(oh);
    outputs[0].d[3] = eb.constant(ow);
    return 0;
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t,
                          DynamicPluginTensorDesc const*, int32_t)
      const noexcept override {
    // worst case: fp16 copies of both inputs (fp32-input calibration pass)
    // + the pre-upsample lift buffer
    size_t hw = (size_t)t_->hf * t_->wf;
    return (size_t)t_->n_cams * (t_->depth_bins + t_->cc) * hw * 2 +
           (size_t)t_->lift_h * t_->lift_w * t_->cc * 2;
  }

  // ---- IPluginV3OneRuntime
  int32_t onShapeChange(PluginTensorDesc const*, int32_t,
                        PluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc,
                  PluginTensorDesc const* outputDesc,
                  void const* const* inputs, void* const* outputs,
                  void* workspace, cudaStream_t stream) noexcept override {
    if (!dev_ready_ && upload() != 0) return -1;
    bool dpn = inputDesc[0].format == TensorFormat::kHWC8;
    bool cxn = inputDesc[1].format == TensorFormat::kHWC8;
    bool outn = outputDesc[0].format == TensorFormat::kHWC8;
    bool out_f32 = outputDesc[0].type == DataType::kFLOAT;
    int G2 = t_->lift_h * t_->lift_w;
    size_t hw = (size_t)t_->hf * t_->wf;
    size_t dlogN = (size_t)t_->n_cams * t_->depth_bins * hw;
    size_t ctxN = (size_t)t_->n_cams * t_->cc * hw;
    // workspace layout: [dlog16][ctx16][lift16]
    __half* ws_dlog = (__half*)workspace;
    __half* ws_ctx = ws_dlog + dlogN;
    __half* ws_lift = ws_ctx + ctxN;
    cudaError_t e;
    const __half* dp = (const __half*)inputs[0];
    const __half* cx = (const __half*)inputs[1];
    if (inputDesc[0].type == DataType::kFLOAT) {   // calibration-graph pass
      e = liftk::launch_f32toh((const float*)inputs[0], ws_dlog, dlogN,
                               stream);
      if (e != cudaSuccess) return -1;
      dp = ws_dlog;
      dpn = false;
    }
    if (inputDesc[1].type == DataType::kFLOAT) {
      e = liftk::launch_f32toh((const float*)inputs[1], ws_ctx, ctxN, stream);
      if (e != cudaSuccess) return -1;
      cx = ws_ctx;
      cxn = false;
    }
    bool up = t_->out_h > 0 && t_->out_h != t_->lift_h;
    if (t_->fold_softmax || up) {
      __half* ws = up ? ws_lift : (__half*)outputs[0];
      if (t_->fold_softmax)
        e = liftk::launch_lift_fused(
            dpn, cxn, dp, cx, d_rowptr_, d_col_, d_cam_, d_ix_, d_iy_,
            d_b0_, d_fr_, ws, t_->depth_bins, t_->cc, t_->hf, t_->wf, G2,
            stream);
      else
        e = liftk::launch_lift(
            dpn, cxn, true, dp, cx, d_rowptr_, d_col_, d_cam_, d_ix_,
            d_iy_, d_b0_, d_fr_, ws, t_->depth_bins, t_->cc, t_->hf, t_->wf,
            G2, stream, false);
      if (e != cudaSuccess) return -1;
      if (up)
        e = liftk::launch_upsample(out_f32, outn, ws, outputs[0], t_->cc,
                                   t_->lift_h, t_->lift_w, t_->out_h,
                                   t_->out_w, stream);
    } else {
      e = liftk::launch_lift(
          dpn, cxn, outn, dp, cx, d_rowptr_, d_col_, d_cam_, d_ix_, d_iy_,
          d_b0_, d_fr_, outputs[0], t_->depth_bins, t_->cc, t_->hf, t_->wf,
          G2, stream, out_f32);
    }
    return e == cudaSuccess ? 0 : -1;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    return clone();
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override {
    ser_.clear();
    auto& t = *t_;
    ser_.emplace_back("n_cams", &t.n_cams, PluginFieldType::kINT32, 1);
    ser_.emplace_back("depth_bins", &t.depth_bins, PluginFieldType::kINT32, 1);
    ser_.emplace_back("cc", &t.cc, PluginFieldType::kINT32, 1);
    ser_.emplace_back("hf", &t.hf, PluginFieldType::kINT32, 1);
    ser_.emplace_back("wf", &t.wf, PluginFieldType::kINT32, 1);
    ser_.emplace_back("lift_h", &t.lift_h, PluginFieldType::kINT32, 1);
    ser_.emplace_back("lift_w", &t.lift_w, PluginFieldType::kINT32, 1);
    ser_.emplace_back("fold_softmax", &t.fold_softmax,
                      PluginFieldType::kINT32, 1);
    ser_.emplace_back("out_h", &t.out_h, PluginFieldType::kINT32, 1);
    ser_.emplace_back("out_w", &t.out_w, PluginFieldType::kINT32, 1);
    ser_.emplace_back("out_fp32", &t.out_fp32, PluginFieldType::kINT32, 1);
    auto arr = [&](char const* n, void const* p, PluginFieldType ty,
                   int32_t len) { ser_.emplace_back(n, p, ty, len); };
    arr("rowptr", t.rowptr.data(), PluginFieldType::kINT32,
        (int32_t)t.rowptr.size());
    arr("col", t.col.data(), PluginFieldType::kINT32, (int32_t)t.col.size());
    arr("cam", t.cam.data(), PluginFieldType::kINT32, (int32_t)t.cam.size());
    arr("b0", t.b0.data(), PluginFieldType::kINT32, (int32_t)t.b0.size());
    arr("ix", t.ix.data(), PluginFieldType::kFLOAT32, (int32_t)t.ix.size());
    arr("iy", t.iy.data(), PluginFieldType::kFLOAT32, (int32_t)t.iy.size());
    arr("fr", t.fr.data(), PluginFieldType::kFLOAT32, (int32_t)t.fr.size());
    fc_.nbFields = (int32_t)ser_.size();
    fc_.fields = ser_.data();
    return &fc_;
  }

 private:
  int32_t upload() noexcept {
    auto up = [](auto& dst, auto const& src) -> cudaError_t {
      size_t n = src.size() * sizeof(src[0]);
      cudaError_t e = cudaMalloc((void**)&dst, n);
      if (e != cudaSuccess) return e;
      return cudaMemcpy(dst, src.data(), n, cudaMemcpyHostToDevice);
    };
    CKRT(up(d_rowptr_, t_->rowptr));
    CKRT(up(d_col_, t_->col));
    CKRT(up(d_cam_, t_->cam));
    CKRT(up(d_b0_, t_->b0));
    CKRT(up(d_ix_, t_->ix));
    CKRT(up(d_iy_, t_->iy));
    CKRT(up(d_fr_, t_->fr));
    dev_ready_ = true;
    return 0;
  }
  void freeDev() noexcept {
    for (void* p : {(void*)d_rowptr_, (void*)d_col_, (void*)d_cam_,
                    (void*)d_b0_, (void*)d_ix_, (void*)d_iy_, (void*)d_fr_})
      if (p) cudaFree(p);
    d_rowptr_ = d_col_ = d_cam_ = d_b0_ = nullptr;
    d_ix_ = d_iy_ = d_fr_ = nullptr;
    dev_ready_ = false;
  }

  std::shared_ptr<Tables> t_;
  int32_t *d_rowptr_{}, *d_col_{}, *d_cam_{}, *d_b0_{};
  float *d_ix_{}, *d_iy_{}, *d_fr_{};
  bool dev_ready_{false};
  std::vector<PluginField> ser_;
  PluginFieldCollection fc_{};
};

class MeteorLiftCreator : public IPluginCreatorV3One {
 public:
  MeteorLiftCreator() {
    static std::vector<PluginField> f = {
        {"n_cams", nullptr, PluginFieldType::kINT32, 1},
        {"depth_bins", nullptr, PluginFieldType::kINT32, 1},
        {"cc", nullptr, PluginFieldType::kINT32, 1},
        {"hf", nullptr, PluginFieldType::kINT32, 1},
        {"wf", nullptr, PluginFieldType::kINT32, 1},
        {"lift_h", nullptr, PluginFieldType::kINT32, 1},
        {"lift_w", nullptr, PluginFieldType::kINT32, 1},
        {"fold_softmax", nullptr, PluginFieldType::kINT32, 1},
        {"out_h", nullptr, PluginFieldType::kINT32, 1},
        {"out_w", nullptr, PluginFieldType::kINT32, 1},
        {"out_fp32", nullptr, PluginFieldType::kINT32, 1},
        {"rowptr", nullptr, PluginFieldType::kINT32, 1},
        {"col", nullptr, PluginFieldType::kINT32, 1},
        {"cam", nullptr, PluginFieldType::kINT32, 1},
        {"b0", nullptr, PluginFieldType::kINT32, 1},
        {"ix", nullptr, PluginFieldType::kFLOAT32, 1},
        {"iy", nullptr, PluginFieldType::kFLOAT32, 1},
        {"fr", nullptr, PluginFieldType::kFLOAT32, 1},
    };
    fc_.nbFields = (int32_t)f.size();
    fc_.fields = f.data();
  }
  char const* getPluginName() const noexcept override { return kNAME; }
  char const* getPluginVersion() const noexcept override { return kVER; }
  char const* getPluginNamespace() const noexcept override { return kNS; }
  PluginFieldCollection const* getFieldNames() noexcept override {
    return &fc_;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc,
                          TensorRTPhase) noexcept override {
    if (!fc) return nullptr;
    auto t = std::make_shared<Tables>();
    for (int32_t i = 0; i < fc->nbFields; ++i) {
      auto const& f = fc->fields[i];
      std::string n = f.name;
      auto iv = [&](std::vector<int32_t>& v) {
        v.assign((int32_t const*)f.data, (int32_t const*)f.data + f.length);
      };
      auto fv = [&](std::vector<float>& v) {
        v.assign((float const*)f.data, (float const*)f.data + f.length);
      };
      if (n == "n_cams") t->n_cams = *(int32_t const*)f.data;
      else if (n == "depth_bins") t->depth_bins = *(int32_t const*)f.data;
      else if (n == "cc") t->cc = *(int32_t const*)f.data;
      else if (n == "hf") t->hf = *(int32_t const*)f.data;
      else if (n == "wf") t->wf = *(int32_t const*)f.data;
      else if (n == "lift_h") t->lift_h = *(int32_t const*)f.data;
      else if (n == "lift_w") t->lift_w = *(int32_t const*)f.data;
      else if (n == "fold_softmax") t->fold_softmax = *(int32_t const*)f.data;
      else if (n == "out_h") t->out_h = *(int32_t const*)f.data;
      else if (n == "out_w") t->out_w = *(int32_t const*)f.data;
      else if (n == "out_fp32") t->out_fp32 = *(int32_t const*)f.data;
      else if (n == "rowptr") iv(t->rowptr);
      else if (n == "col") iv(t->col);
      else if (n == "cam") iv(t->cam);
      else if (n == "b0") iv(t->b0);
      else if (n == "ix") fv(t->ix);
      else if (n == "iy") fv(t->iy);
      else if (n == "fr") fv(t->fr);
    }
    if (t->rowptr.size() != (size_t)t->lift_h * t->lift_w + 1 ||
        t->col.size() != t->cam.size() || t->ix.size() != t->col.size())
      return nullptr;
    return new MeteorLiftPlugin(std::move(t));
  }

 private:
  PluginFieldCollection fc_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(MeteorLiftCreator);
