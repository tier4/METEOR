// liftbench: standalone benchmark of the METEOR frustum lift on real tables.
//
// The lift, per (camera,cell) pair p with constant tables (cam, cell, ix, iy,
// b0, fr) derived from the rig calibration (see dump_lift.py):
//   w[p]      = bilerp(dprob[cam, b0], ix, iy) * (1-fr)
//             + bilerp(dprob[cam, b0+1], ix, iy) * fr + 0.05
//   val[p,c]  = bilerp(ctx[cam, c], ix, iy) * w[p]
//   num[cell,c] = sum_{p in cell} val[p,c] ;  den[cell] = sum w[p]
//   bev[cell,c] = num / max(den, 1e-4)
// bilinear = grid_sample align_corners=False, zeros padding, on 108x192.
//
// Variants:
//   scatter_nchw / scatter_nhwc : fused kernel, atomicAdd into BEV (baseline,
//                                 reproduces the in-engine ScatterND form)
//   gather_nchw / gather_nhwc   : fused kernel, one block per output cell
//   tc_spmm                     : stage1 elementwise W[P,Cp] + cusparseSpMM
//                                 (CSR 0/1 selection matrix) + divide
//   tc_panelk                   : stage1 + padded fixed-K (K<=3) panel
//                                 gather-sum + divide.  cublasHgemm on the
//                                 per-cell panels degenerates to m=1 batched
//                                 GEMMs at this K distribution (1..3, mean
//                                 1.7), so the panel reduction is done by a
//                                 plain kernel -- see report.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(e)); exit(1); } } while (0)
#define CKS(x) do { cusparseStatus_t s = (x); \
  if (s != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSPARSE %s:%d status %d\n", __FILE__, __LINE__, \
            (int)s); exit(1); } } while (0)

static int N, Cc, D, Hf, Wf, G2, P, KMAX;
static const int CP = 104;          // stage-1 W row stride: 96 ctx + w + pad

// ---------------------------------------------------------------- helpers
struct Taps { int i[4]; float w[4]; };

__device__ __forceinline__ Taps taps(float ix, float iy, int H, int W) {
  int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
  float fx = ix - x0, fy = iy - y0;
  Taps t;
  int xs[2] = {x0, x0 + 1}, ys[2] = {y0, y0 + 1};
  float wx[2] = {1.f - fx, fx}, wy[2] = {1.f - fy, fy};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int xx = xs[j & 1], yy = ys[j >> 1];
    bool in = (xx >= 0 && xx < W && yy >= 0 && yy < H);
    t.i[j] = in ? yy * W + xx : 0;
    t.w[j] = in ? wx[j & 1] * wy[j >> 1] : 0.f;
  }
  return t;
}

// pair weight from dprob. NHWC: [cam][y][x][D] ; NCHW: [cam][D][y][x]
template <bool NHWC>
__device__ __forceinline__ float pair_w(const __half* dp, int cam, int b0,
                                        float fr, const Taps& t,
                                        int D_, int HW) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float d0, d1;
    if (NHWC) {
      const __half* b = dp + ((size_t)cam * HW + t.i[j]) * D_ + b0;
      d0 = __half2float(b[0]); d1 = __half2float(b[1]);
    } else {
      const __half* b = dp + ((size_t)cam * D_ + b0) * HW + t.i[j];
      d0 = __half2float(b[0]); d1 = __half2float(b[HW]);
    }
    a += t.w[j] * (d0 * (1.f - fr) + d1 * fr);
  }
  return a + 0.05f;
}

// ctx bilerp for one channel. NHWC: [cam][y][x][Cc] ; NCHW: [cam][Cc][y][x]
template <bool NHWC>
__device__ __forceinline__ float ctx_bl(const __half* cx, int cam, int c,
                                        const Taps& t, int Cc_, int HW) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half* b = NHWC ? cx + ((size_t)cam * HW + t.i[j]) * Cc_ + c
                           : cx + ((size_t)cam * Cc_ + c) * HW + t.i[j];
    a += t.w[j] * __half2float(*b);
  }
  return a;
}

// ------------------------------------------------------- variant kernels
// (a) scatter: block = pair, threads = Cc, atomicAdd into num/den
template <bool NHWC>
__global__ void k_scatter(const __half* __restrict__ dp,
                          const __half* __restrict__ cx,
                          const int* cam_, const int* cell_,
                          const float* ix_, const float* iy_,
                          const int* b0_, const float* fr_,
                          float* __restrict__ num, float* __restrict__ den,
                          int D_, int Cc_, int Hf_, int Wf_) {
  int p = blockIdx.x, c = threadIdx.x;
  __shared__ float sw;
  __shared__ Taps st;
  int cam = cam_[p], cell = cell_[p];
  if (c == 0) {
    st = taps(ix_[p], iy_[p], Hf_, Wf_);
    sw = pair_w<NHWC>(dp, cam, b0_[p], fr_[p], st, D_, Hf_ * Wf_);
    atomicAdd(&den[cell], sw);
  }
  __syncthreads();
  float v = ctx_bl<NHWC>(cx, cam, c, st, Cc_, Hf_ * Wf_) * sw;
  atomicAdd(&num[(size_t)cell * Cc_ + c], v);
}

__global__ void k_finalize(const float* num, const float* den, __half* out,
                           int Cc_) {
  int cell = blockIdx.x, c = threadIdx.x;
  float d = fmaxf(den[cell], 1e-4f);
  out[(size_t)cell * Cc_ + c] = __float2half(num[(size_t)cell * Cc_ + c] / d);
}

// (b) fused gather: block = cell, threads = Cc, csr pair list
template <bool NHWC>
__global__ void k_gather(const __half* __restrict__ dp,
                         const __half* __restrict__ cx,
                         const int* __restrict__ rowptr,
                         const int* __restrict__ col,
                         const int* cam_, const float* ix_, const float* iy_,
                         const int* b0_, const float* fr_,
                         __half* __restrict__ out,
                         int D_, int Cc_, int Hf_, int Wf_) {
  int cell = blockIdx.x, c = threadIdx.x;
  int r0 = rowptr[cell], k = rowptr[cell + 1] - r0;
  __shared__ float sw[8];
  __shared__ Taps st[8];
  __shared__ int scam[8];
  if (c < k) {
    int p = col[r0 + c];
    scam[c] = cam_[p];
    st[c] = taps(ix_[p], iy_[p], Hf_, Wf_);
    sw[c] = pair_w<NHWC>(dp, scam[c], b0_[p], fr_[p], st[c], D_, Hf_ * Wf_);
  }
  __syncthreads();
  float acc = 0.f, den = 0.f;
  for (int j = 0; j < k; ++j) {
    den += sw[j];
    acc += ctx_bl<NHWC>(cx, scam[j], c, st[j], Cc_, Hf_ * Wf_) * sw[j];
  }
  out[(size_t)cell * Cc_ + c] = __float2half(acc / fmaxf(den, 1e-4f));
}

// (c/d) stage 1: W[p, 0..95] = val, W[p, 96] = w.  block = pair, threads = Cc
template <bool F16>
__global__ void k_stage1(const __half* __restrict__ dp,
                         const __half* __restrict__ cx,
                         const int* cam_, const float* ix_, const float* iy_,
                         const int* b0_, const float* fr_,
                         void* __restrict__ Wm,
                         int D_, int Cc_, int Hf_, int Wf_) {
  int p = blockIdx.x, c = threadIdx.x;
  __shared__ float sw;
  __shared__ Taps st;
  int cam = cam_[p];
  if (c == 0) {
    st = taps(ix_[p], iy_[p], Hf_, Wf_);
    sw = pair_w<true>(dp, cam, b0_[p], fr_[p], st, D_, Hf_ * Wf_);
    if (F16) ((__half*)Wm)[(size_t)p * CP + Cc_] = __float2half(sw);
    else     ((float*)Wm)[(size_t)p * CP + Cc_] = sw;
  }
  __syncthreads();
  float v = ctx_bl<true>(cx, cam, c, st, Cc_, Hf_ * Wf_) * sw;
  if (F16) ((__half*)Wm)[(size_t)p * CP + c] = __float2half(v);
  else     ((float*)Wm)[(size_t)p * CP + c] = v;
}

template <bool F16>
__global__ void k_divide(const void* Cm, __half* out, int Cc_) {
  int cell = blockIdx.x, c = threadIdx.x;
  float num = F16 ? __half2float(((const __half*)Cm)[(size_t)cell * CP + c])
                  : ((const float*)Cm)[(size_t)cell * CP + c];
  float den = F16 ? __half2float(((const __half*)Cm)[(size_t)cell * CP + Cc_])
                  : ((const float*)Cm)[(size_t)cell * CP + Cc_];
  out[(size_t)cell * Cc_ + c] = __float2half(num / fmaxf(den, 1e-4f));
}

// (d) padded fixed-K panel reduction over stage-1 W (pad row = P, zeros)
template <bool F16>
__global__ void k_panelk(const void* Wm, const int* __restrict__ tab,
                         __half* __restrict__ out, int Cc_, int K_) {
  int cell = blockIdx.x, c = threadIdx.x;
  float acc = 0.f, den = 0.f;
  for (int k = 0; k < K_; ++k) {
    int p = tab[cell * K_ + k];
    if (F16) {
      acc += __half2float(((const __half*)Wm)[(size_t)p * CP + c]);
      den += __half2float(((const __half*)Wm)[(size_t)p * CP + Cc_]);
    } else {
      acc += ((const float*)Wm)[(size_t)p * CP + c];
      den += ((const float*)Wm)[(size_t)p * CP + Cc_];
    }
  }
  out[(size_t)cell * Cc_ + c] = __float2half(acc / fmaxf(den, 1e-4f));
}

// context: a plain NCHW->NHWC transpose (what a plugin needs if TRT will not
// hand it kHWC8/kHWC16 directly)
__global__ void k_nchw2nhwc(const __half* in, __half* out, int C, int HW) {
  int n = blockIdx.z;
  int hw = blockIdx.x * blockDim.x + threadIdx.x;
  int c = blockIdx.y;
  if (hw < HW)
    out[((size_t)n * HW + hw) * C + c] = in[((size_t)n * C + c) * HW + hw];
}

// ------------------------------------------------------------------ host
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

struct Timer {
  cudaEvent_t a, b;
  Timer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  template <typename F>
  float run(F f, cudaStream_t s, int warm = 20, int iters = 200) {
    for (int i = 0; i < warm; ++i) f(s);
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < iters; ++i) f(s);
    CK(cudaEventRecord(b, s));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
  }
};

static void check(const char* name, const __half* d_out,
                  const std::vector<float>& ref, float ms) {
  std::vector<__half> h((size_t)G2 * Cc);
  CK(cudaMemcpy(h.data(), d_out, h.size() * sizeof(__half),
                cudaMemcpyDeviceToHost));
  double mabs = 0, mrel = 0;
  for (int cell = 0; cell < G2; ++cell)
    for (int c = 0; c < Cc; ++c) {
      double o = __half2float(h[(size_t)cell * Cc + c]);
      double r = ref[(size_t)c * G2 + cell];
      double d = fabs(o - r);
      mabs = fmax(mabs, d);
      if (fabs(r) > 0.05) mrel = fmax(mrel, d / fabs(r));
    }
  bool ok = mabs < 0.05 || mrel < 1e-2;
  printf("%-14s %8.4f ms   max|diff| %.4e  maxrel %.4e  %s\n",
         name, ms, mabs, mrel, ok ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "tables";
  auto meta = nlohmann::json::parse(std::ifstream(dir + "/meta.json"));
  N = meta["N"]; Cc = meta["Cc"]; D = meta["D"];
  Hf = meta["Hf"]; Wf = meta["Wf"]; G2 = meta["G2"]; P = meta["P"];
  KMAX = meta["k_max"];
  int HW = Hf * Wf;
  printf("liftbench: N=%d Cc=%d D=%d feat %dx%d G2=%d P=%d kmax=%d\n",
         N, Cc, D, Hf, Wf, G2, P, KMAX);
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s SM%d%d\n", prop.name, prop.major, prop.minor);

  auto cam = load<int>(dir + "/pair_cam.bin", P);
  auto cell = load<int>(dir + "/pair_cell.bin", P);
  auto ix = load<float>(dir + "/pair_ix.bin", P);
  auto iy = load<float>(dir + "/pair_iy.bin", P);
  auto b0 = load<int>(dir + "/pair_b0.bin", P);
  auto fr = load<float>(dir + "/pair_fr.bin", P);
  auto rowptr = load<int>(dir + "/csr_rowptr.bin", G2 + 1);
  auto col = load<int>(dir + "/csr_col.bin", P);
  auto dprob = load<__half>(dir + "/dprob.bin", (size_t)N * D * HW);
  auto ctx = load<__half>(dir + "/ctx.bin", (size_t)N * Cc * HW);
  auto ref = load<float>(dir + "/ref_lift.bin", (size_t)Cc * G2);

  // host transposes to NHWC
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

  // padded fixed-K table (pad slot = row P of the stage-1 W buffer = zeros)
  std::vector<int> tab((size_t)G2 * KMAX, P);
  for (int cll = 0; cll < G2; ++cll)
    for (int j = rowptr[cll]; j < rowptr[cll + 1]; ++j)
      tab[(size_t)cll * KMAX + (j - rowptr[cll])] = col[j];

  int *d_cam = todev(cam), *d_cell = todev(cell), *d_b0 = todev(b0);
  float *d_ix = todev(ix), *d_iy = todev(iy), *d_fr = todev(fr);
  int *d_rp = todev(rowptr), *d_col = todev(col), *d_tab = todev(tab);
  __half *d_dp = todev(dprob), *d_cx = todev(ctx);
  __half *d_dpt = todev(dprob_t), *d_cxt = todev(ctx_t);

  float* d_num; CK(cudaMalloc(&d_num, (size_t)G2 * Cc * 4));
  float* d_den; CK(cudaMalloc(&d_den, (size_t)G2 * 4));
  __half* d_out; CK(cudaMalloc(&d_out, (size_t)G2 * Cc * 2));
  __half* d_W16; CK(cudaMalloc(&d_W16, (size_t)(P + 1) * CP * 2));
  CK(cudaMemset(d_W16, 0, (size_t)(P + 1) * CP * 2));
  float* d_W32; CK(cudaMalloc(&d_W32, (size_t)(P + 1) * CP * 4));
  CK(cudaMemset(d_W32, 0, (size_t)(P + 1) * CP * 4));
  __half* d_C16; CK(cudaMalloc(&d_C16, (size_t)G2 * CP * 2));
  float* d_C32; CK(cudaMalloc(&d_C32, (size_t)G2 * CP * 4));
  std::vector<__half> ones16(P, __float2half(1.f));
  std::vector<float> ones32(P, 1.f);
  __half* d_v16 = todev(ones16);
  float* d_v32 = todev(ones32);

  cudaStream_t s; CK(cudaStreamCreate(&s));
  Timer T;

  // ---- cusparse SpMM setup: probe fp16, fall back to fp32 ---------------
  cusparseHandle_t h; CKS(cusparseCreate(&h)); cusparseSetStream(h, s);
  float alpha = 1.f, beta = 0.f;
  bool spmm16 = true;
  cusparseSpMatDescr_t A16, A32;
  cusparseDnMatDescr_t B16, C16d, B32, C32d;
  void* d_buf = nullptr; size_t bufsz = 0;
  {
    cusparseStatus_t st = cusparseCreateCsr(
        &A16, G2, P, P, d_rp, d_col, d_v16, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
    if (st == CUSPARSE_STATUS_SUCCESS) {
      CKS(cusparseCreateDnMat(&B16, P + 1, CP, CP, d_W16, CUDA_R_16F,
                              CUSPARSE_ORDER_ROW));
      CKS(cusparseCreateDnMat(&C16d, G2, CP, CP, d_C16, CUDA_R_16F,
                              CUSPARSE_ORDER_ROW));
      // B declared P+1 rows but A has P cols; make a P-row view
      cusparseDnMatDescr_t Bv;
      CKS(cusparseCreateDnMat(&Bv, P, CP, CP, d_W16, CUDA_R_16F,
                              CUSPARSE_ORDER_ROW));
      B16 = Bv;
      st = cusparseSpMM_bufferSize(
          h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha, A16, B16, &beta, C16d, CUDA_R_32F,
          CUSPARSE_SPMM_CSR_ALG2, &bufsz);
    }
    if (st != CUSPARSE_STATUS_SUCCESS) {
      printf("[spmm] fp16 SpMM unsupported (status %d) -> fp32 fallback\n",
             (int)st);
      spmm16 = false;
    }
  }
  if (!spmm16) {
    CKS(cusparseCreateCsr(&A32, G2, P, P, d_rp, d_col, d_v32,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CKS(cusparseCreateDnMat(&B32, P, CP, CP, d_W32, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
    CKS(cusparseCreateDnMat(&C32d, G2, CP, CP, d_C32, CUDA_R_32F,
                            CUSPARSE_ORDER_ROW));
    CKS(cusparseSpMM_bufferSize(
        h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, A32, B32, &beta, C32d, CUDA_R_32F,
        CUSPARSE_SPMM_CSR_ALG2, &bufsz));
  }
  CK(cudaMalloc(&d_buf, bufsz ? bufsz : 4));

  // ------------------------------------------------------------- variants
  auto scatter_nchw = [&](cudaStream_t st) {
    cudaMemsetAsync(d_num, 0, (size_t)G2 * Cc * 4, st);
    cudaMemsetAsync(d_den, 0, (size_t)G2 * 4, st);
    k_scatter<false><<<P, Cc, 0, st>>>(d_dp, d_cx, d_cam, d_cell, d_ix, d_iy,
                                       d_b0, d_fr, d_num, d_den, D, Cc, Hf, Wf);
    k_finalize<<<G2, Cc, 0, st>>>(d_num, d_den, d_out, Cc);
  };
  auto scatter_nhwc = [&](cudaStream_t st) {
    cudaMemsetAsync(d_num, 0, (size_t)G2 * Cc * 4, st);
    cudaMemsetAsync(d_den, 0, (size_t)G2 * 4, st);
    k_scatter<true><<<P, Cc, 0, st>>>(d_dpt, d_cxt, d_cam, d_cell, d_ix, d_iy,
                                      d_b0, d_fr, d_num, d_den, D, Cc, Hf, Wf);
    k_finalize<<<G2, Cc, 0, st>>>(d_num, d_den, d_out, Cc);
  };
  auto gather_nchw = [&](cudaStream_t st) {
    k_gather<false><<<G2, Cc, 0, st>>>(d_dp, d_cx, d_rp, d_col, d_cam, d_ix,
                                       d_iy, d_b0, d_fr, d_out, D, Cc, Hf, Wf);
  };
  auto gather_nhwc = [&](cudaStream_t st) {
    k_gather<true><<<G2, Cc, 0, st>>>(d_dpt, d_cxt, d_rp, d_col, d_cam, d_ix,
                                      d_iy, d_b0, d_fr, d_out, D, Cc, Hf, Wf);
  };
  auto stage1 = [&](cudaStream_t st) {
    if (spmm16)
      k_stage1<true><<<P, Cc, 0, st>>>(d_dpt, d_cxt, d_cam, d_ix, d_iy, d_b0,
                                       d_fr, d_W16, D, Cc, Hf, Wf);
    else
      k_stage1<false><<<P, Cc, 0, st>>>(d_dpt, d_cxt, d_cam, d_ix, d_iy, d_b0,
                                        d_fr, d_W32, D, Cc, Hf, Wf);
  };
  auto tc_spmm = [&](cudaStream_t st) {
    stage1(st);
    if (spmm16) {
      cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A16, B16, &beta,
                   C16d, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, d_buf);
      k_divide<true><<<G2, Cc, 0, st>>>(d_C16, d_out, Cc);
    } else {
      cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A32, B32, &beta,
                   C32d, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, d_buf);
      k_divide<false><<<G2, Cc, 0, st>>>(d_C32, d_out, Cc);
    }
  };
  auto tc_panelk = [&](cudaStream_t st) {
    stage1(st);
    if (spmm16)
      k_panelk<true><<<G2, Cc, 0, st>>>(d_W16, d_tab, d_out, Cc, KMAX);
    else
      k_panelk<false><<<G2, Cc, 0, st>>>(d_W32, d_tab, d_out, Cc, KMAX);
  };
  __half* d_scr; CK(cudaMalloc(&d_scr, (size_t)N * (Cc + D) * HW * 2));
  auto transpose = [&](cudaStream_t st) {
    dim3 g((HW + 255) / 256, Cc, N);
    k_nchw2nhwc<<<g, 256, 0, st>>>(d_cx, d_scr, Cc, HW);
    dim3 g2((HW + 255) / 256, D, N);
    k_nchw2nhwc<<<g2, 256, 0, st>>>(d_dp, d_scr + (size_t)N * Cc * HW, D, HW);
  };

  struct V { const char* name; std::function<void(cudaStream_t)> f; bool chk; };
  std::vector<V> vs = {
      {"scatter_nchw", scatter_nchw, true},
      {"scatter_nhwc", scatter_nhwc, true},
      {"gather_nchw", gather_nchw, true},
      {"gather_nhwc", gather_nhwc, true},
      {"tc_spmm", tc_spmm, true},
      {"tc_panelk", tc_panelk, true},
      {"stage1_only", stage1, false},
      {"nchw2nhwc", transpose, false},
  };
  printf("[spmm] path: %s\n", spmm16 ? "fp16 io / fp32 compute"
                                     : "fp32 io / fp32 compute");
  for (auto& v : vs) {
    CK(cudaMemsetAsync(d_out, 0, (size_t)G2 * Cc * 2, s));
    v.f(s);                                    // one correctness run
    CK(cudaStreamSynchronize(s));
    CK(cudaGetLastError());
    float ms = T.run(v.f, s);
    if (v.chk) check(v.name, d_out, ref, ms);
    else printf("%-14s %8.4f ms   (context, no output check)\n", v.name, ms);
  }
  return 0;
}
