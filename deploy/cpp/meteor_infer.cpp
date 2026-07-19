// METEOR C++ TensorRT runtime — engine build + streaming inference on raw
// t4dataset scenes. Mirrors deploy/runtime.py + infer_t4dataset.py:
//   * builds (or reuses) a cached fp16 engine from --onnx via the TRT API
//   * parses t4dataset annotation JSONs (sample / sample_data /
//     calibrated_sensor / sensor / ego_pose) with nlohmann::json
//   * preprocesses the 8 cameras exactly like training (768x432, RGB,
//     ImageNet mean/std, CHW)
//   * maintains the 3-slot temporal memory ring (raw_bev fed back with
//     ego-motion warp thetas at t-0.4 / -1.2 / -2.8 s)
//   * zero-fills optional inputs (lidar_bev / kin) -> camera-only mode is
//     bit-equal to training by construction
//   * decodes 3D boxes / lane map / K=3 ego paths and writes an overlay
//     video (front camera | BEV). The full 12-panel visualisation stays in
//     Python (deploy/visualize.py).
//
// Usage:
//   meteor_infer --engine E.engine --t4d SCENE_DIR [--video out.mp4]
//   meteor_infer --onnx  M.onnx    --t4d SCENE_DIR [--limit N]
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

static const int IMG_W = 768, IMG_H = 432;
static const int BEV_H = 800, BEV_W = 500;
static const float DET_RES = 0.4f;
static const int HIST_OFFS[3] = {2, 6, 14};
static const char* CAMS[8] = {
    "CAM_FRONT_WIDE", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK_WIDE",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
    "CAM_FRONT_NARROW", "CAM_BACK_NARROW"};

struct Logger : nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[trt] " << msg << "\n";
  }
} gLogger;

static std::vector<char> readFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { std::cerr << "cannot open " << p << "\n"; exit(1); }
  return {std::istreambuf_iterator<char>(f), {}};
}

// ---------------------------------------------------------------- engine
static std::string buildEngine(const std::string& onnx) {
  std::string eng = onnx.substr(0, onnx.rfind('.')) + "_fp16.engine";
  std::ifstream probe(eng);
  if (probe.good()) {
    std::cout << "[engine] reusing cached " << eng << "\n";
    return eng;
  }
  std::cout << "[engine] building " << eng << " (fp16, one-time)\n";
  auto* builder = nvinfer1::createInferBuilder(gLogger);
  auto* net = builder->createNetworkV2(
      1u << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
  auto* parser = nvonnxparser::createParser(*net, gLogger);
  auto blob = readFile(onnx);
  if (!parser->parse(blob.data(), blob.size())) {
    for (int i = 0; i < parser->getNbErrors(); ++i)
      std::cerr << parser->getError(i)->desc() << "\n";
    exit(1);
  }
  auto* cfg = builder->createBuilderConfig();
  cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 8ull << 30);
  cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
  auto* ser = builder->buildSerializedNetwork(*net, *cfg);
  if (!ser) { std::cerr << "build failed\n"; exit(1); }
  std::ofstream out(eng, std::ios::binary);
  out.write((const char*)ser->data(), ser->size());
  std::cout << "[engine] saved " << eng << "\n";
  return eng;
}

// ------------------------------------------------------------- t4dataset
struct Pose { double x = 0, y = 0, yaw = 0; bool ok = false; };

static double quatYaw(const json& q) {  // [w,x,y,z]
  double w = q[0], x = q[1], y = q[2], z = q[3];
  return std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

struct Scene {
  std::vector<std::string> sampleTokens;                 // ordered
  // per sample: cam -> image path; lidar ego pose
  std::map<std::string, std::map<std::string, std::string>> images;
  std::map<std::string, Pose> egoPose;
  float K[8][3][3];                                      // scaled to 768x432
  float Tce[8][4][4];                                    // T_cam_ego
};

static void mat4FromRT(const json& rot, const json& tr, double M[4][4]) {
  double w = rot[0], x = rot[1], y = rot[2], z = rot[3];
  double R[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
      {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
      {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) M[i][j] = R[i][j];
    M[i][3] = tr[i];
    M[3][i] = 0;
  }
  M[3][3] = 1;
}

static void invertRT(const double M[4][4], float out[4][4]) {
  // rigid transform inverse: R' = R^T, t' = -R^T t
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) out[i][j] = (float)M[j][i];
    out[i][3] = 0;
    out[3][i] = 0;
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) out[i][3] -= (float)(M[j][i] * M[j][3]);
  out[3][3] = 1;
}

static Scene loadScene(const std::string& root) {
  auto J = [&](const std::string& n) {
    return json::parse(readFile(root + "/annotation/" + n));
  };
  json samples = J("sample.json"), sdata = J("sample_data.json");
  json calibs = J("calibrated_sensor.json"), sensors = J("sensor.json");
  json egos = J("ego_pose.json");
  std::map<std::string, std::string> sensorName;
  for (auto& s : sensors) sensorName[s["token"]] = s["channel"];
  std::map<std::string, json> calibByTok;
  for (auto& c : calibs) calibByTok[c["token"]] = c;
  std::map<std::string, Pose> egoByTok;
  for (auto& e : egos) {
    Pose p;
    p.x = e["translation"][0];
    p.y = e["translation"][1];
    p.yaw = quatYaw(e["rotation"]);
    p.ok = true;
    egoByTok[e["token"]] = p;
  }
  // order samples by timestamp
  std::vector<std::pair<long long, std::string>> ts;
  for (auto& s : samples)
    ts.push_back({(long long)s["timestamp"], s["token"]});
  std::sort(ts.begin(), ts.end());
  Scene sc;
  for (auto& t : ts) sc.sampleTokens.push_back(t.second);
  bool calDone[8] = {false};
  for (auto& d : sdata) {
    if (!d["is_key_frame"].get<bool>()) continue;
    std::string ch = sensorName[calibByTok[d["calibrated_sensor_token"]]
                                    ["sensor_token"]];
    std::string sm = d["sample_token"];
    int ci = -1;
    for (int i = 0; i < 8; ++i)
      if (ch == CAMS[i]) ci = i;
    if (ci >= 0) {
      sc.images[sm][ch] = root + "/" + d["filename"].get<std::string>();
      if (!calDone[ci]) {
        json cal = calibByTok[d["calibrated_sensor_token"]];
        double Te[4][4];
        mat4FromRT(cal["rotation"], cal["translation"], Te);  // T_ego_cam
        invertRT(Te, sc.Tce[ci]);
        double sx = double(IMG_W) / double(d["width"]);
        double sy = double(IMG_H) / double(d["height"]);
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) {
            double v = cal["camera_intrinsic"][r][c];
            sc.K[ci][r][c] = (float)(r == 0 ? v * sx : r == 1 ? v * sy : v);
          }
        calDone[ci] = true;
      }
    } else if (ch == "LIDAR_CONCAT") {
      sc.egoPose[sm] = egoByTok[d["ego_pose_token"]];
    }
  }
  return sc;
}

// ------------------------------------------------------------ TRT runner
struct Runner {
  nvinfer1::IRuntime* rt{};
  nvinfer1::ICudaEngine* eng{};
  nvinfer1::IExecutionContext* ctx{};
  cudaStream_t stream{};
  std::map<std::string, void*> dev;
  std::map<std::string, std::vector<float>> host;
  std::map<std::string, size_t> count;
  std::map<std::string, nvinfer1::Dims> dims;

  explicit Runner(const std::string& path) {
    auto blob = readFile(path);
    rt = nvinfer1::createInferRuntime(gLogger);
    eng = rt->deserializeCudaEngine(blob.data(), blob.size());
    ctx = eng->createExecutionContext();
    cudaStreamCreate(&stream);
    for (int i = 0; i < eng->getNbIOTensors(); ++i) {
      const char* nm = eng->getIOTensorName(i);
      auto d = eng->getTensorShape(nm);
      size_t n = 1;
      for (int k = 0; k < d.nbDims; ++k) n *= (size_t)d.d[k];
      dims[nm] = d;
      count[nm] = n;
      host[nm].assign(n, 0.f);
      cudaMalloc(&dev[nm], n * sizeof(float));
      cudaMemset(dev[nm], 0, n * sizeof(float));  // optional inputs = zeros
      ctx->setTensorAddress(nm, dev[nm]);
    }
  }
  void put(const std::string& nm) {
    cudaMemcpyAsync(dev[nm], host[nm].data(), count[nm] * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
  }
  void get(const std::string& nm) {
    cudaMemcpyAsync(host[nm].data(), dev[nm], count[nm] * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
  }
};

static void warpTheta(const Pose& prev, const Pose& cur, float th[6]) {
  double cp = std::cos(prev.yaw), sp = std::sin(prev.yaw);
  double dx = cur.x - prev.x, dy = cur.y - prev.y;
  double tx = cp * dx + sp * dy, ty = -sp * dx + cp * dy;
  double dyaw = cur.yaw - prev.yaw;
  double a = 0.1 * (BEV_H - 1), b = 0.1 * (BEV_W - 1);
  double cd = std::cos(dyaw), sd = std::sin(dyaw);
  double Cx = cd * (80 - a) - sd * (50 - b) + tx;
  double Cy = sd * (80 - a) + cd * (50 - b) + ty;
  th[0] = (float)cd;            th[1] = (float)(a * sd / b);
  th[2] = (float)(((50 - b) - Cy) / b);
  th[3] = (float)(-(b / a) * sd); th[4] = (float)cd;
  th[5] = (float)(((80 - a) - Cx) / a);
}

struct Box { int cls; float sc, x, y, l, w, yaw; };

int main(int argc, char** argv) {
  std::string onnx, engine, t4d, video;
  int limit = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--onnx") onnx = nxt();
    else if (a == "--engine") engine = nxt();
    else if (a == "--t4d") t4d = nxt();
    else if (a == "--video") video = nxt();
    else if (a == "--limit") limit = std::stoi(nxt());
  }
  if (engine.empty() && onnx.empty()) { std::cerr << "need --onnx/--engine\n"; return 1; }
  if (t4d.empty()) { std::cerr << "need --t4d\n"; return 1; }
  if (engine.empty()) engine = buildEngine(onnx);

  Scene sc = loadScene(t4d);
  std::cout << sc.sampleTokens.size() << " samples\n";
  Runner run(engine);
  std::memcpy(run.host["K"].data(), sc.K, sizeof(sc.K));
  std::memcpy(run.host["T_cam_ego"].data(), sc.Tce, sizeof(sc.Tce));
  run.put("K"); run.put("T_cam_ego");

  cv::VideoWriter vw;
  if (!video.empty())
    vw.open(video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 10,
            cv::Size(IMG_W + 300, IMG_H));

  const float MEAN[3] = {0.485f, 0.456f, 0.406f};
  const float STD[3] = {0.229f, 0.224f, 0.225f};
  std::map<int, std::pair<std::vector<float>, Pose>> ring;  // t -> raw_bev
  Pose prevPose; double prevT = -1; float v0 = 0;
  int t = 0, done = 0;

  for (size_t si = 0; si < sc.sampleTokens.size(); si += 2) {
    const auto& tok = sc.sampleTokens[si];
    auto it = sc.images.find(tok);
    if (it == sc.images.end() || it->second.size() < 8) continue;
    cv::Mat front;
    for (int ci = 0; ci < 8; ++ci) {
      cv::Mat im = cv::imread(it->second[CAMS[ci]]);
      if (im.empty()) { std::cerr << "bad image\n"; continue; }
      cv::resize(im, im, cv::Size(IMG_W, IMG_H));
      if (ci == 0) front = im.clone();
      float* dst = run.host["imgs"].data() + (size_t)ci * 3 * IMG_H * IMG_W;
      for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
          cv::Vec3b p = im.at<cv::Vec3b>(y, x);  // BGR
          for (int c = 0; c < 3; ++c)
            dst[(size_t)c * IMG_H * IMG_W + (size_t)y * IMG_W + x] =
                ((p[2 - c] / 255.f) - MEAN[c]) / STD[c];
        }
    }
    Pose pose = sc.egoPose.count(tok) ? sc.egoPose[tok] : Pose{};
    if (pose.ok && prevT >= 0) {
      double d = std::hypot(pose.x - prevPose.x, pose.y - prevPose.y);
      v0 = (float)(d / 0.2);
    }
    // temporal ring
    for (int s = 0; s < 3; ++s) {
      auto h = ring.find(t - HIST_OFFS[s]);
      float* hb = run.host["hist_bev"].data() +
                  (size_t)s * 96 * BEV_H * BEV_W;
      float* ht = run.host["hist_theta"].data() + (size_t)s * 6;
      if (h == ring.end() || !pose.ok || !h->second.second.ok) {
        std::memset(hb, 0, sizeof(float) * 96 * BEV_H * BEV_W);
        float ident[6] = {1, 0, 0, 0, 1, 0};
        std::memcpy(ht, ident, sizeof(ident));
      } else {
        std::memcpy(hb, h->second.first.data(),
                    sizeof(float) * 96 * BEV_H * BEV_W);
        warpTheta(h->second.second, pose, ht);
      }
    }
    run.host["v0"][0] = v0;
    run.put("imgs"); run.put("v0"); run.put("hist_bev"); run.put("hist_theta");
    run.ctx->enqueueV3(run.stream);
    run.get("lane"); run.get("hm"); run.get("reg"); run.get("ego");
    run.get("raw_bev");
    cudaStreamSynchronize(run.stream);
    ring[t] = {run.host["raw_bev"], pose};
    for (auto itr = ring.begin(); itr != ring.end();)
      itr = (itr->first < t - 14) ? ring.erase(itr) : std::next(itr);

    // ---- decode 3D boxes (port of runtime.decode_boxes) ----
    auto& hm = run.host["hm"]; auto& rg = run.host["reg"];
    int H = run.dims["hm"].d[2], W = run.dims["hm"].d[3];
    std::vector<Box> boxes;
    for (int cls = 0; cls < 2; ++cls)
      for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
          float v = hm[(size_t)cls * H * W + (size_t)r * W + c];
          float p = 1.f / (1.f + std::exp(-v));
          if (p < 0.35f) continue;
          bool peak = true;
          for (int dr = -1; dr <= 1 && peak; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
              int rr = r + dr, cc = c + dc;
              if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
              float q = hm[(size_t)cls * H * W + (size_t)rr * W + cc];
              if (q > v) { peak = false; break; }
            }
          if (!peak) continue;
          auto R = [&](int ch) {
            return rg[(size_t)ch * H * W + (size_t)r * W + c];
          };
          boxes.push_back({cls, p, 80.f - (r + R(0)) * DET_RES,
                           50.f - (c + R(1)) * DET_RES,
                           std::exp(R(2)), std::exp(R(3)),
                           std::atan2(R(4), R(5))});
        }
    // ---- overlay: front cam | mini BEV ----
    if (vw.isOpened()) {
      cv::Mat fr(IMG_H, IMG_W + 300, CV_8UC3, cv::Scalar(0, 0, 0));
      front.copyTo(fr(cv::Rect(0, 0, IMG_W, IMG_H)));
      auto& lane = run.host["lane"];
      int LC = run.dims["lane"].d[1];
      cv::Mat bev(300, 250, CV_8UC3, cv::Scalar(20, 20, 20));
      static const cv::Vec3b PAL[9] = {
          {0, 0, 0}, {80, 80, 80}, {0, 0, 0}, {200, 180, 0},
          {255, 255, 255}, {0, 0, 255}, {0, 165, 255},
          {180, 105, 255}, {60, 60, 60}};
      for (int r = 0; r < 300; ++r)
        for (int c = 0; c < 250; ++c) {
          int gr = 200 + r, gc = 125 + c;  // x +40..-20, y +25..-25
          int best = 0; float bv = -1e9f;
          for (int k = 0; k < LC && k < 9; ++k) {
            float v = lane[(size_t)k * BEV_H * BEV_W +
                           (size_t)gr * BEV_W + gc];
            if (v > bv) { bv = v; best = k; }
          }
          bev.at<cv::Vec3b>(r, c) = PAL[best];
        }
      cv::resize(bev, bev, cv::Size(250, 300));
      auto px = [&](float x, float y) {
        return cv::Point(int((25.f - y) / 0.2f * (250.f / 250.f)),
                         int((40.f - x) / 0.2f * (300.f / 300.f)));
      };
      for (auto& b : boxes)
        if (b.x < 40 && b.x > -20 && std::fabs(b.y) < 25)
          cv::circle(bev, px(b.x, b.y), 4,
                     b.cls == 0 ? cv::Scalar(0, 215, 255)
                                : cv::Scalar(255, 0, 255), -1);
      auto& e = run.host["ego"];
      int kb = 0; float bcf = -1e9f;
      for (int k = 0; k < 3; ++k)
        if (e[36 + k] > bcf) { bcf = e[36 + k]; kb = k; }
      cv::Point prev = px(0, 0);
      for (int wp = 0; wp < 6; ++wp) {
        cv::Point q = px(e[kb * 12 + wp * 2], e[kb * 12 + wp * 2 + 1]);
        cv::line(bev, prev, q, cv::Scalar(0, 255, 0), 2);
        prev = q;
      }
      bev.copyTo(fr(cv::Rect(IMG_W + 25, 60, 250, 300)));
      cv::putText(fr, cv::format("v0 %.1f km/h  boxes %zu  (C++ TensorRT)",
                                 v0 * 3.6f, boxes.size()),
                  {10, IMG_H - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                  {0, 255, 0}, 1);
      vw.write(fr);
    }
    prevPose = pose; prevT = 0.2 * t;
    ++t; ++done;
    if (done % 20 == 0) std::cout << done << " frames\n";
    if (limit && done >= limit) break;
  }
  if (vw.isOpened()) vw.release();
  std::cout << "done " << done << " frames\n";
  return 0;
}
