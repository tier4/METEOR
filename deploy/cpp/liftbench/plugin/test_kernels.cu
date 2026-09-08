// Verifies every layout combination of the plugin kernel against the model's
// dumped lift output before an engine build is spent on it.
//   usage: test_kernels <tables_dir>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "lift_kernels.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(e)); exit(1); } } while (0)

template <typename T>
std::vector<T> load(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(1); }
  std::vector<T> v(n);
  f.read((char*)v.data(), n * sizeof(T));
  if ((size_t)f.gcount() != n * sizeof(T)) {
    fprintf(stderr, "short read %s\n", p.c_str()); exit(1);
  }
  return v;
}

template <typename T>
T* todev(const std::vector<T>& v) {
  T* d; CK(cudaMalloc(&d, v.size() * sizeof(T)));
  CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "tables";
  auto meta = nlohmann::json::parse(std::ifstream(dir + "/meta.json"));
  int N = meta["N"], Cc = meta["Cc"], D = meta["D"], Hf = meta["Hf"],
      Wf = meta["Wf"], G2 = meta["G2"], P = meta["P"];
  int HW = Hf * Wf;
  auto cam = load<int>(dir + "/pair_cam.bin", P);
  auto ix = load<float>(dir + "/pair_ix.bin", P);
  auto iy = load<float>(dir + "/pair_iy.bin", P);
  auto b0 = load<int>(dir + "/pair_b0.bin", P);
  auto fr = load<float>(dir + "/pair_fr.bin", P);
  auto rowptr = load<int>(dir + "/csr_rowptr.bin", G2 + 1);
  auto col = load<int>(dir + "/csr_col.bin", P);
  auto dprob = load<__half>(dir + "/dprob.bin", (size_t)N * D * HW);
  auto ctx = load<__half>(dir + "/ctx.bin", (size_t)N * Cc * HW);
  auto ref = load<float>(dir + "/ref_lift.bin", (size_t)Cc * G2);

  std::vector<__half> dprob_t(dprob.size()), ctx_t(ctx.size());
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < D; ++c)
      for (int hw = 0; hw < HW; ++hw)
        dprob_t[((size_t)n * HW + hw) * D + c] =
            dprob[((size_t)n * D + c) * HW + hw];
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < Cc; ++c)
      for (int hw = 0; hw < HW; ++hw)
        ctx_t[((size_t)n * HW + hw) * Cc + c] =
            ctx[((size_t)n * Cc + c) * HW + hw];

  int *d_rp = todev(rowptr), *d_col = todev(col), *d_cam = todev(cam),
      *d_b0 = todev(b0);
  float *d_ix = todev(ix), *d_iy = todev(iy), *d_fr = todev(fr);
  __half *d_dp = todev(dprob), *d_cx = todev(ctx);
  __half *d_dpt = todev(dprob_t), *d_cxt = todev(ctx_t);
  __half* d_out; CK(cudaMalloc(&d_out, (size_t)G2 * Cc * 2));

  // ---- fused mode: logits in, softmax + lift + x4 upsample inside -------
  int OH = meta["out_h"], OW = meta["out_w"];
  auto dlog = load<__half>(dir + "/dlog.bin", (size_t)N * D * HW);
  auto ref_up = load<float>(dir + "/ref_lift_up.bin", (size_t)Cc * OH * OW);
  std::vector<__half> dlog_t(dlog.size());
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < D; ++c)
      for (int hw = 0; hw < HW; ++hw)
        dlog_t[((size_t)n * HW + hw) * D + c] =
            dlog[((size_t)n * D + c) * HW + hw];
  __half* d_dl = todev(dlog);
  __half* d_dlt = todev(dlog_t);
  __half* d_ws; CK(cudaMalloc(&d_ws, (size_t)G2 * Cc * 2));
  float* d_up; CK(cudaMalloc(&d_up, (size_t)Cc * OH * OW * 4));
  int fails = 0;
  for (int m = 0; m < 4; ++m) {
    bool dpn = m & 1, cxn = m & 2;
    CK(liftk::launch_lift_fused(dpn, cxn, dpn ? d_dlt : d_dl,
                                cxn ? d_cxt : d_cx, d_rp, d_col, d_cam, d_ix,
                                d_iy, d_b0, d_fr, d_ws, D, Cc, Hf, Wf, G2, 0));
    CK(liftk::launch_upsample(true, false, d_ws, d_up, Cc, meta["lift_h"],
                              meta["lift_w"], OH, OW, 0));
    CK(cudaDeviceSynchronize());
    std::vector<float> h((size_t)Cc * OH * OW);
    CK(cudaMemcpy(h.data(), d_up, h.size() * 4, cudaMemcpyDeviceToHost));
    double mabs = 0;
    for (size_t i = 0; i < h.size(); ++i)
      mabs = fmax(mabs, fabs((double)h[i] - ref_up[i]));
    bool ok = mabs < 0.05;
    fails += !ok;
    printf("FUSED dlog=%s ctx=%s out=f32/chw(600x500)  max|diff| %.4e  %s\n",
           dpn ? "nhwc" : "nchw", cxn ? "nhwc" : "nchw", mabs,
           ok ? "OK" : "FAIL");
  }
  {  // timing split of the fused path (NHWC inputs, fp32 CHW out)
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    for (int i = 0; i < 20; ++i) {
      liftk::launch_lift_fused(true, true, d_dlt, d_cxt, d_rp, d_col, d_cam,
                               d_ix, d_iy, d_b0, d_fr, d_ws, D, Cc, Hf, Wf,
                               G2, 0);
      liftk::launch_upsample(true, false, d_ws, d_up, Cc, meta["lift_h"],
                             meta["lift_w"], OH, OW, 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int i = 0; i < 100; ++i)
      liftk::launch_lift_fused(true, true, d_dlt, d_cxt, d_rp, d_col, d_cam,
                               d_ix, d_iy, d_b0, d_fr, d_ws, D, Cc, Hf, Wf,
                               G2, 0);
    cudaEventRecord(e1);
    for (int i = 0; i < 100; ++i)
      liftk::launch_upsample(true, false, d_ws, d_up, Cc, meta["lift_h"],
                             meta["lift_w"], OH, OW, 0);
    cudaEventRecord(e2);
    cudaEventSynchronize(e2);
    float t1, t2;
    cudaEventElapsedTime(&t1, e0, e1);
    cudaEventElapsedTime(&t2, e1, e2);
    printf("TIMING lift_fused %.4f ms, upsample(f32/chw) %.4f ms\n",
           t1 / 100, t2 / 100);
  }
  for (int m = 0; m < 8; ++m) {
    bool dpn = m & 1, cxn = m & 2, outn = m & 4;
    CK(cudaMemset(d_out, 0, (size_t)G2 * Cc * 2));
    CK(liftk::launch_lift(dpn, cxn, outn, dpn ? d_dpt : d_dp,
                          cxn ? d_cxt : d_cx, d_rp, d_col, d_cam, d_ix, d_iy,
                          d_b0, d_fr, d_out, D, Cc, Hf, Wf, G2, 0));
    CK(cudaDeviceSynchronize());
    std::vector<__half> h((size_t)G2 * Cc);
    CK(cudaMemcpy(h.data(), d_out, h.size() * 2, cudaMemcpyDeviceToHost));
    double mabs = 0;
    for (int cell = 0; cell < G2; ++cell)
      for (int c = 0; c < Cc; ++c) {
        double o = __half2float(outn ? h[(size_t)cell * Cc + c]
                                     : h[(size_t)c * G2 + cell]);
        double d = fabs(o - ref[(size_t)c * G2 + cell]);
        if (d > mabs) mabs = d;
      }
    bool ok = mabs < 0.05;
    fails += !ok;
    printf("dp=%s ctx=%s out=%s  max|diff| %.4e  %s\n",
           dpn ? "nhwc" : "nchw", cxn ? "nhwc" : "nchw",
           outn ? "nhwc" : "nchw", mabs, ok ? "OK" : "FAIL");
  }
  return fails;
}
