// Real-time METEOR demo on the Orin, C++ port of deploy/orin_realtime.py:
// pipelined inference + rendering (2026-09-06 refresh: N cameras from the
// engine, pinned zero-copy input slots, no-hist engines, CUDA Graph,
// --bench for an infer-only latency figure comparable to bench_rt_zc.py).
//
//   loader thread   disk -> raw frames, stacked straight into a pinned slot
//   producer thread owns the TRT engine (MeteorRT), bounded queue out
//   sequencer       tags frames with sequence numbers
//   2 render workers compose_frame in parallel
//   main thread     re-orders by sequence and writes the video
//
//   METEOR_PLUGIN_SO=... METEOR_CUDAGRAPH=1 \
//   meteor_realtime --engine eng/v147c3Zg_int8.engine --root valday \
//       --out out/rt_cpp.mp4 --limit 300 [--loop] [--bench 40]
#include <opencv2/opencv.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "meteor_rt.hpp"
#include "npz.hpp"
#include "render.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

constexpr int IMG_W = 768, IMG_H = 432;

double nowSec() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// bounded MPMC queue; close() releases all waiters
template <typename T>
class BQueue {
 public:
  explicit BQueue(size_t cap = SIZE_MAX) : cap_(cap) {}
  bool put(T v) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    cv_.notify_all();
    return true;
  }
  bool get(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    cv_.notify_all();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closed_ = true;
    cv_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;
  size_t cap_;
  bool closed_ = false;
};

struct RawItem {
  std::vector<cv::Mat> raw;   // N BGR images, model input order
  int inSlot = -1;            // pinned imgs slot holding the stacked uint8
  std::vector<float> K;       // N*9
  std::vector<float> Tc;      // N*16 (T_cam_ego = inv(T_ego_cam))
  float v0 = 8.0f;
  bool hasPose = false;
  std::array<float, 3> pose{};
  std::vector<float> lidar;   // [4,400,250] fp32 or empty (METEOR_LIDAR=1)
};

struct InferItem {
  std::shared_ptr<RawItem> rw;
  OutMap out;
  double dt = 0;  // infer ms
  int slot = 0;
};

struct SeqItem {
  long seq;
  std::shared_ptr<InferItem> it;
};

struct DoneItem {
  long seq;
  cv::Mat canvas;
  double dt, rms;
};

std::string readAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// invert 4x4 (np.linalg.inv equivalent)
void inv4x4(const double M[16], float out[16]) {
  cv::Mat m(4, 4, CV_64F, const_cast<double*>(M));
  cv::Mat mi = m.inv();
  for (int i = 0; i < 16; ++i)
    out[i] = (float)mi.at<double>(i / 4, i % 4);
}

// BGR HWC -> RGB CHW uint8 into dst (3*H*W)
void stackRGB(const cv::Mat& im, uint8_t* dst) {
  for (int ch = 0; ch < 3; ++ch) {
    uint8_t* d = dst + (size_t)ch * IMG_H * IMG_W;
    for (int y = 0; y < IMG_H; ++y) {
      const uint8_t* sp = im.ptr<uint8_t>(y);
      uint8_t* dp = d + (size_t)y * IMG_W;
      for (int x = 0; x < IMG_W; ++x) dp[x] = sp[3 * x + (2 - ch)];
    }
  }
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine, root = "fast", outPath;
  int stride = 1, limit = 0, bench = 0;
  bool loop = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nxt = [&]() { return std::string(argv[++i]); };
    if (a == "--engine") engine = nxt();
    else if (a == "--root") root = nxt();
    else if (a == "--stride") stride = std::stoi(nxt());
    else if (a == "--out") outPath = nxt();
    else if (a == "--limit") limit = std::stoi(nxt());
    else if (a == "--bench") bench = std::stoi(nxt());
    else if (a == "--loop") loop = true;
    else {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 1;
    }
  }
  if (engine.empty()) {
    std::fprintf(stderr, "need --engine\n");
    return 1;
  }

  const int N_SLOTS = 4, N_IN = bench ? 20 : 5;
  std::set<std::string> skip = {"occ", "flow", "unk", "pl", "tl",
                                "lg_pts", "lg_meta", "lg_adj"};
  if (const char* s = std::getenv("METEOR_SKIP_OUT")) {  // like bench_rt_zc
    skip.clear();
    std::string v = s;
    size_t i = 0;
    while (i < v.size()) {
      size_t j = v.find(',', i);
      if (j == std::string::npos) j = v.size();
      if (j > i) skip.insert(v.substr(i, j - i));
      i = j + 1;
    }
  }
  MeteorRT rt(engine, skip, N_SLOTS, N_IN);
  const bool useLidar = std::getenv("METEOR_LIDAR") && std::string(std::getenv("METEOR_LIDAR")) == "1"
                        && rt.hasLidar();
  if (std::getenv("METEOR_LIDAR") && !rt.hasLidar())
    std::fprintf(stderr, "[rt] METEOR_LIDAR set but the engine has no lidar_bev input (camera-only export)\n");
  if (useLidar) std::fprintf(stderr, "[rt] LiDAR input ON (lidar_bev from scene/lidar_bev/NNNN.npz)\n");
  const int N = (int)rt.shapeOf("imgs")[1];
  std::vector<std::string> cams(CAM_IN8, CAM_IN8 + N);
  const bool u8 = rt.imgsAreU8();
  if (!u8) {
    std::fprintf(stderr, "fp32-imgs engines are not supported by this build "
                         "(export with --uint8-in)\n");
    return 1;
  }
  std::fprintf(stderr, "[rt] %d cameras: ", N);
  for (auto& c : cams) std::fprintf(stderr, "%s ", c.c_str());
  std::fprintf(stderr, "\n");

  std::vector<std::string> scenes;
  for (const auto& e : fs::directory_iterator(root))
    if (e.is_directory() && fs::is_regular_file(e.path() / "manifest.json"))
      scenes.push_back(e.path().filename().string());
  std::sort(scenes.begin(), scenes.end());
  if (scenes.empty()) {
    std::fprintf(stderr, "no scenes under %s\n", root.c_str());
    return 1;
  }

  // scene calibration (K, T_cam_ego) in model input order
  auto loadCalib = [&](const json& m, std::vector<float>& K,
                       std::vector<float>& Tc) {
    K.assign((size_t)N * 9, 0.f);
    Tc.assign((size_t)N * 16, 0.f);
    for (int ci = 0; ci < N; ++ci) {
      if (!m["cams"].contains(cams[ci])) return false;
      const auto& cam = m["cams"][cams[ci]];
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
          K[ci * 9 + r * 3 + c] = (float)cam["K"][r][c].get<double>();
      double Te[16];
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          Te[r * 4 + c] = cam["T_ego_cam"][r][c].get<double>();
      inv4x4(Te, Tc.data() + ci * 16);
    }
    return true;
  };

  // ---- bench: infer-only latency on the first scene's frames ----------
  if (bench) {
    const auto& s = scenes[0];
    json m = json::parse(readAll(root + "/" + s + "/manifest.json"));
    std::vector<float> K, Tc;
    if (!loadCalib(m, K, Tc)) {
      std::fprintf(stderr, "scene %s lacks a camera\n", s.c_str());
      return 1;
    }
    NpyArray v0s, poses;
    try {
      auto z = npz_load(root + "/" + s + "/ego_motion.npz");
      if (z.count("v0")) v0s = std::move(z["v0"]);
      if (z.count("pose")) poses = std::move(z["pose"]);
    } catch (...) {
    }
    const auto& frames = m["frames"];
    int nf = 0;
    std::vector<float> v0v;
    std::vector<std::array<float, 3>> pv;
    for (size_t fi = 3; fi < frames.size() && nf < N_IN; ++fi) {
      const auto& f = frames[fi];
      uint8_t* dst = (uint8_t*)rt.inputSlot(nf);
      bool ok = true;
      for (int ci = 0; ci < N; ++ci) {
        std::string rel = "_";
        if (f.contains("imgs") && f["imgs"].contains(cams[ci]))
          rel = f["imgs"][cams[ci]].get<std::string>();
        cv::Mat im = cv::imread(root + "/" + s + "/" + rel);
        if (im.empty()) { ok = false; break; }
        if (im.cols != IMG_W || im.rows != IMG_H)
          cv::resize(im, im, {IMG_W, IMG_H});
        stackRGB(im, dst + (size_t)ci * 3 * IMG_H * IMG_W);
      }
      if (!ok) continue;
      long fr = f.contains("frame") ? f["frame"].get<long>() : -1;
      v0v.push_back((fr >= 0 && fr < (long)v0s.rows()) ? v0s.data[fr] : 8.0f);
      std::array<float, 3> p{0, 0, 0};
      if (!poses.data.empty() && fr >= 0 && fr < (long)poses.rows())
        for (int j = 0; j < 3; ++j) p[j] = poses.data[fr * 3 + j];
      pv.push_back(p);
      ++nf;
    }
    std::fprintf(stderr, "[bench] %d frames loaded into pinned slots\n", nf);
    for (int i = 0; i < 8; ++i)
      rt.infer(rt.inputSlot(i % nf), K.data(), Tc.data(), v0v[i % nf],
               pv[i % nf].data(), 0);
    std::vector<double> ts;
    for (int i = 0; i < bench; ++i) {
      double t = nowSec();
      rt.infer(rt.inputSlot(i % nf), K.data(), Tc.data(), v0v[i % nf],
               pv[i % nf].data(), i % N_SLOTS);
      ts.push_back((nowSec() - t) * 1000.0);
    }
    double mean = 0;
    for (double x : ts) mean += x;
    mean /= ts.size();
    std::printf("[bench C++] %s: 中央値 %.1f ms  平均 %.1f ms  (n=%d, graph=%s, "
                "zero-copy in, out_slots=%d)\n",
                engine.c_str(), median(ts), mean, bench,
                rt.graphActive() ? "on" : "off", N_SLOTS);
    return 0;
  }

  BQueue<std::shared_ptr<RawItem>> qRaw(3);
  BQueue<std::shared_ptr<InferItem>> q(2);
  BQueue<std::shared_ptr<SeqItem>> qSeq(6);
  BQueue<std::shared_ptr<DoneItem>> doneQ;
  BQueue<int> freeSlots, freeIn;
  for (int i = 0; i < N_SLOTS; ++i) freeSlots.put(i);
  for (int i = 0; i < N_IN; ++i) freeIn.put(i);
  std::atomic<bool> stop{false};

  // ---- loader: disk -> raw frames stacked into a pinned input slot ------
  std::thread thL([&] {
    while (!stop.load()) {
      for (const auto& s : scenes) {
        json m;
        try {
          m = json::parse(readAll(root + "/" + s + "/manifest.json"));
        } catch (...) {
          continue;
        }
        std::vector<float> K, Tc;
        if (!loadCalib(m, K, Tc)) continue;
        NpyArray v0s, poses;
        bool haveEmo = false;
        try {
          auto z = npz_load(root + "/" + s + "/ego_motion.npz");
          if (z.count("v0")) {
            v0s = std::move(z["v0"]);
            haveEmo = true;
          }
          if (z.count("pose")) poses = std::move(z["pose"]);
        } catch (...) {
        }
        const auto& frames = m["frames"];
        for (size_t fi = 0; fi < frames.size(); fi += (size_t)stride) {
          if (stop.load()) return;
          const auto& f = frames[fi];
          auto item = std::make_shared<RawItem>();
          item->raw.resize(N);
          bool ok = true;
          for (int ci = 0; ci < N; ++ci) {
            std::string rel = "_";
            if (f.contains("imgs") && f["imgs"].contains(cams[ci]))
              rel = f["imgs"][cams[ci]].get<std::string>();
            cv::Mat im = cv::imread(root + "/" + s + "/" + rel);
            if (im.empty()) {
              ok = false;
              break;
            }
            if (im.cols != IMG_W || im.rows != IMG_H)
              cv::resize(im, im, {IMG_W, IMG_H});
            item->raw[ci] = im;
          }
          if (!ok) continue;
          int si;
          if (!freeIn.get(si)) return;  // wait for a free pinned slot
          uint8_t* dst = (uint8_t*)rt.inputSlot(si);
          for (int ci = 0; ci < N; ++ci)
            stackRGB(item->raw[ci], dst + (size_t)ci * 3 * IMG_H * IMG_W);
          item->inSlot = si;
          item->K = K;
          item->Tc = Tc;
          long fr = f.contains("frame") ? f["frame"].get<long>() : -1;
          item->v0 = (haveEmo && fr >= 0 && fr < (long)v0s.rows())
                         ? v0s.data[fr]
                         : 8.0f;
          if (!poses.data.empty() && fr >= 0 && fr < (long)poses.rows()) {
            item->hasPose = true;
            for (int j = 0; j < 3; ++j) item->pose[j] = poses.data[fr * 3 + j];
          }
          if (useLidar && fr >= 0) {
            std::string lp = (f.contains("lidar_bev") ? f["lidar_bev"].get<std::string>()
                                                       : "lidar_bev/" + std::string(4 - std::min(4, (int)std::to_string(fr).size()), '0') + std::to_string(fr) + ".npz");
            try {
              auto z = npz_load(root + "/" + s + "/" + lp);
              if (z.count("lb") && z["lb"].data.size() == (size_t)4 * 400 * 250)
                item->lidar = std::move(z["lb"].data);
            } catch (...) {
            }
          }
          if (!qRaw.put(item)) return;
        }
      }
      if (!loop) break;
    }
    qRaw.put(nullptr);
  });

  // ---- producer: owns the TRT engine -----------------------------------
  std::thread thI([&] {
    while (!stop.load()) {
      std::shared_ptr<RawItem> item;
      if (!qRaw.get(item)) return;
      if (!item) break;
      int slot;
      if (!freeSlots.get(slot)) return;  // blocks until a consumer released
      double t0 = nowSec();
      auto inf = std::make_shared<InferItem>();
      inf->out = rt.infer(rt.inputSlot(item->inSlot), item->K.data(),
                          item->Tc.data(), item->v0,
                          item->hasPose ? item->pose.data() : nullptr, slot,
                          item->lidar.empty() ? nullptr : item->lidar.data());
      inf->dt = (nowSec() - t0) * 1000.0;
      freeIn.put(item->inSlot);  // H2D done (infer is synchronous)
      item->inSlot = -1;
      inf->rw = item;
      inf->slot = slot;
      if (!q.put(inf)) return;
    }
    q.put(nullptr);
  });

  cv::VideoWriter vw;
  if (!outPath.empty()) {
    fs::path parent = fs::path(outPath).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
    double fps = std::atof(std::getenv("METEOR_REC_FPS")
                               ? std::getenv("METEOR_REC_FPS") : "10");
    vw.open(outPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
            {1920, 1080});
  }
  long n = 0;
  double tStart = nowSec();
  std::vector<double> inferMs, renderMs;

  // ---- sequencer --------------------------------------------------------
  std::thread thS([&] {
    long i = 0;
    while (true) {
      std::shared_ptr<InferItem> item;
      if (!q.get(item)) return;
      if (!item) {
        qSeq.put(nullptr);
        return;
      }
      if (!qSeq.put(std::make_shared<SeqItem>(SeqItem{i, item}))) return;
      ++i;
    }
  });

  // ---- render pool: 2 workers ------------------------------------------
  auto renderWorker = [&] {
    while (true) {
      std::shared_ptr<SeqItem> si;
      if (!qSeq.get(si)) return;
      if (!si) {
        qSeq.put(nullptr);
        doneQ.put(nullptr);
        return;
      }
      double t0 = nowSec();
      double fpsNow = si->seq / std::max(nowSec() - tStart, 1e-3);
      cv::Mat canvas =
          compose_frame(si->it->rw->raw, cams, si->it->rw->K.data(),
                        si->it->rw->Tc.data(), si->it->rw->v0, si->it->out,
                        si->it->dt, fpsNow,
                        si->it->rw->hasPose ? si->it->rw->pose.data() : nullptr,
                        si->it->rw->lidar.empty() ? nullptr : si->it->rw->lidar.data());
      freeSlots.put(si->it->slot);  // slot data fully consumed
      auto d = std::make_shared<DoneItem>();
      d->seq = si->seq;
      d->canvas = canvas;
      d->dt = si->it->dt;
      d->rms = (nowSec() - t0) * 1000.0;
      doneQ.put(d);
    }
  };
  std::vector<std::thread> workers;
  const int nWorkers = std::getenv("METEOR_RENDER_WORKERS")
                           ? std::atoi(std::getenv("METEOR_RENDER_WORKERS")) : 3;
  for (int i = 0; i < std::max(1, nWorkers); ++i) workers.emplace_back(renderWorker);

  // ---- ordered output ----------------------------------------------------
  std::map<long, std::shared_ptr<DoneItem>> pending;
  long nextSeq = 0;
  int ended = 0;
  while (ended < (int)workers.size()) {
    std::shared_ptr<DoneItem> d;
    if (!doneQ.get(d)) break;
    if (!d) {
      ++ended;
      continue;
    }
    pending[d->seq] = d;
    bool hitLimit = false;
    while (pending.count(nextSeq)) {
      auto cur = pending[nextSeq];
      pending.erase(nextSeq);
      renderMs.push_back(cur->rms);
      inferMs.push_back(cur->dt);
      ++n;
      ++nextSeq;
      if (vw.isOpened()) vw.write(cur->canvas);
      if (n % 50 == 0)
        std::fprintf(stderr, "[rt] %ld frames  %.1f FPS  infer %.0f ms\n", n,
                     n / std::max(nowSec() - tStart, 1e-3), cur->dt);
      if (limit && n >= limit) {
        hitLimit = true;
        break;
      }
    }
    if (hitLimit) break;
  }
  stop.store(true);
  qRaw.close();
  q.close();
  qSeq.close();
  freeSlots.close();
  freeIn.close();
  doneQ.close();
  thL.join();
  thI.join();
  thS.join();
  for (auto& w : workers) w.join();
  if (vw.isOpened()) vw.release();

  double el = nowSec() - tStart;
  auto mean = [](const std::vector<double>& v) {
    std::vector<double> u(v.size() > 6 ? v.begin() + 3 : v.begin(), v.end());
    if (u.empty()) return 0.0;
    double s = 0;
    for (double x : u) s += x;
    return s / u.size();
  };
  std::printf("frames=%ld wall=%.1fs -> %.1f FPS  (infer %.0f ms median %.0f, "
              "render %.0f ms, pipelined, graph=%s)\n",
              n, el, n / el, mean(inferMs), median(inferMs), mean(renderMs),
              rt.graphActive() ? "on" : "off");
  return 0;
}
