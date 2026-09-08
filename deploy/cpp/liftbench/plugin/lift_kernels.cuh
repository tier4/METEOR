// Fused gather-form METEOR lift kernel, shared by the TRT plugin and the
// standalone tester. Math identical to liftbench.cu's k_gather (the 0.27 ms
// winner), plus selectable layouts for dprob / ctx / output:
//   LINEAR = NCHW, HWC8 = NHWC (all channel counts here are multiples of 8,
//   so HWC8 carries no padding).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace liftk {

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

template <bool NHWC>
__device__ __forceinline__ float pair_w(const __half* dp, int cam, int b0,
                                        float fr, const Taps& t,
                                        int D, int HW) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float d0, d1;
    if (NHWC) {
      const __half* b = dp + ((size_t)cam * HW + t.i[j]) * D + b0;
      d0 = __half2float(b[0]); d1 = __half2float(b[1]);
    } else {
      const __half* b = dp + ((size_t)cam * D + b0) * HW + t.i[j];
      d0 = __half2float(b[0]); d1 = __half2float(b[HW]);
    }
    a += t.w[j] * (d0 * (1.f - fr) + d1 * fr);
  }
  return a + 0.05f;
}

template <bool NHWC>
__device__ __forceinline__ float ctx_bl(const __half* cx, int cam, int c,
                                        const Taps& t, int Cc, int HW) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half* b = NHWC ? cx + ((size_t)cam * HW + t.i[j]) * Cc + c
                           : cx + ((size_t)cam * Cc + c) * HW + t.i[j];
    a += t.w[j] * __half2float(*b);
  }
  return a;
}

// fp32 -> fp16 elementwise (calibration-graph inputs arrive as fp32 linear)
__global__ void k_f32toh(const float* __restrict__ in,
                         __half* __restrict__ out, size_t n) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = __float2half(in[i]);
}

inline cudaError_t launch_f32toh(const float* in, __half* out, size_t n,
                                 cudaStream_t s) {
  int b = 256, g = (int)min((n + b - 1) / b, (size_t)4096);
  k_f32toh<<<g, b, 0, s>>>(in, out, n);
  return cudaGetLastError();
}

// one block per lift cell, Cc threads; requires K(pairs per cell) <= 8
template <bool DPN, bool CXN>
__global__ void k_lift(const __half* __restrict__ dp,
                       const __half* __restrict__ cx,
                       const int* __restrict__ rowptr,
                       const int* __restrict__ col,
                       const int* __restrict__ cam_,
                       const float* __restrict__ ix_,
                       const float* __restrict__ iy_,
                       const int* __restrict__ b0_,
                       const float* __restrict__ fr_,
                       void* __restrict__ out,
                       int D, int Cc, int Hf, int Wf, int G2, bool out_nhwc,
                       bool out_f32) {
  int cell = blockIdx.x, c = threadIdx.x;
  int r0 = rowptr[cell], k = rowptr[cell + 1] - r0;
  __shared__ float sw[8];
  __shared__ Taps st[8];
  __shared__ int scam[8];
  if (c < k) {
    int p = col[r0 + c];
    scam[c] = cam_[p];
    st[c] = taps(ix_[p], iy_[p], Hf, Wf);
    sw[c] = pair_w<DPN>(dp, scam[c], b0_[p], fr_[p], st[c], D, Hf * Wf);
  }
  __syncthreads();
  float acc = 0.f, den = 0.f;
  for (int j = 0; j < k; ++j) {
    den += sw[j];
    acc += ctx_bl<CXN>(cx, scam[j], c, st[j], Cc, Hf * Wf) * sw[j];
  }
  float v = acc / fmaxf(den, 1e-4f);
  size_t o = out_nhwc ? (size_t)cell * Cc + c : (size_t)c * G2 + cell;
  if (out_f32) ((float*)out)[o] = v;
  else         ((__half*)out)[o] = __float2half(v);
}

// ---- fused variant: raw LOGITS in (softmax computed per sampled pixel,
// fp32, full 64-bin denominator), lift into a [G2, Cc] fp16 workspace, then
// bilinear x4 upsample folded (replaces the graph's Softmax and Resize).
template <bool DPN, bool CXN>
__global__ void k_lift_fused(const __half* __restrict__ dlog,
                             const __half* __restrict__ cx,
                             const int* __restrict__ rowptr,
                             const int* __restrict__ col,
                             const int* __restrict__ cam_,
                             const float* __restrict__ ix_,
                             const float* __restrict__ iy_,
                             const int* __restrict__ b0_,
                             const float* __restrict__ fr_,
                             __half* __restrict__ ws,
                             int D, int Cc, int Hf, int Wf) {
  int cell = blockIdx.x, c = threadIdx.x;
  int r0 = rowptr[cell], k = rowptr[cell + 1] - r0;
  __shared__ float sw[8], sfr[8], part[8][4];
  __shared__ Taps st[8];
  __shared__ int scam[8], sb0[8];
  int HW = Hf * Wf;
  if (c < k) {
    int p = col[r0 + c];
    scam[c] = cam_[p];
    sb0[c] = b0_[p];
    sfr[c] = fr_[p];
    st[c] = taps(ix_[p], iy_[p], Hf, Wf);
  }
  __syncthreads();
  if (c < 4 * k) {                       // one thread per (pair, tap)
    int kk = c >> 2, j = c & 3;
    float tw = st[kk].w[j], v = 0.f;
    if (tw != 0.f) {
      int b0 = sb0[kk];
      float fr = sfr[kk];
      const __half* base = DPN
          ? dlog + ((size_t)scam[kk] * HW + st[kk].i[j]) * D
          : dlog + (size_t)scam[kk] * D * HW + st[kk].i[j];
      int stride = DPN ? 1 : HW;
      float m = -1e30f;
      for (int d = 0; d < D; ++d)
        m = fmaxf(m, __half2float(base[(size_t)d * stride]));
      float Z = 0.f, l0 = 0.f, l1 = 0.f;
      for (int d = 0; d < D; ++d) {
        float e = expf(__half2float(base[(size_t)d * stride]) - m);
        Z += e;
        if (d == b0) l0 = e;
        if (d == b0 + 1) l1 = e;
      }
      v = tw * (l0 * (1.f - fr) + l1 * fr) / Z;
    }
    part[kk][j] = v;
  }
  __syncthreads();
  if (c < k)
    sw[c] = part[c][0] + part[c][1] + part[c][2] + part[c][3] + 0.05f;
  __syncthreads();
  float acc = 0.f, den = 0.f;
  for (int j = 0; j < k; ++j) {
    den += sw[j];
    acc += ctx_bl<CXN>(cx, scam[j], c, st[j], Cc, HW) * sw[j];
  }
  ws[(size_t)cell * Cc + c] = __float2half(acc / fmaxf(den, 1e-4f));
}

// bilinear upsample (align_corners=False, torch/ONNX half_pixel semantics)
// from the [lh*lw, Cc] fp16 workspace to the full BEV, fp32 or fp16 out.
// Grid-stride with the thread->element mapping chosen so consecutive
// threads write consecutive OUTPUT addresses in either layout (the first
// version used a thread-per-channel block and strided CHW writes 1.2 MB
// apart -- measured 12 ms in-engine for what is a 115 MB write).
__device__ __forceinline__ float upsample_one(const __half* __restrict__ ws,
                                              int Cc, int lh, int lw,
                                              int oh, int ow, int y, int x,
                                              int c) {
  float sy = (y + 0.5f) * lh / (float)oh - 0.5f;
  float sx = (x + 0.5f) * lw / (float)ow - 0.5f;
  int y0 = (int)floorf(sy), x0 = (int)floorf(sx);
  float fy = sy - y0, fx = sx - x0;
  int y0c = min(max(y0, 0), lh - 1), y1c = min(max(y0 + 1, 0), lh - 1);
  int x0c = min(max(x0, 0), lw - 1), x1c = min(max(x0 + 1, 0), lw - 1);
  auto L = [&](int yy, int xx) {
    return __half2float(ws[((size_t)yy * lw + xx) * Cc + c]);
  };
  return (1.f - fy) * ((1.f - fx) * L(y0c, x0c) + fx * L(y0c, x1c))
       + fy * ((1.f - fx) * L(y1c, x0c) + fx * L(y1c, x1c));
}

// CHW out: grid (oh, Cc), threads along x -> coalesced writes, no divides.
// The two source cell-rows this output row needs are staged in shared
// memory once per block (the HWC workspace makes per-tap reads strided).
template <typename TO>
__global__ void k_upsample_chw(const __half* __restrict__ ws,
                               TO* __restrict__ out, int Cc, int lh, int lw,
                               int oh, int ow) {
  int y = blockIdx.x, c = blockIdx.y;
  __shared__ __half r0[128], r1[128];        // lw <= 128 (125 here)
  float sy = (y + 0.5f) * lh / (float)oh - 0.5f;
  int y0 = (int)floorf(sy);
  float fy = sy - y0;
  int y0c = min(max(y0, 0), lh - 1), y1c = min(max(y0 + 1, 0), lh - 1);
  for (int t = threadIdx.x; t < lw; t += blockDim.x) {
    r0[t] = ws[((size_t)y0c * lw + t) * Cc + c];
    r1[t] = ws[((size_t)y1c * lw + t) * Cc + c];
  }
  __syncthreads();
  for (int x = threadIdx.x; x < ow; x += blockDim.x) {
    float sx = (x + 0.5f) * lw / (float)ow - 0.5f;
    int x0 = (int)floorf(sx);
    float fx = sx - x0;
    int x0c = min(max(x0, 0), lw - 1), x1c = min(max(x0 + 1, 0), lw - 1);
    float t0 = (1.f - fx) * __half2float(r0[x0c])
             + fx * __half2float(r0[x1c]);
    float t1 = (1.f - fx) * __half2float(r1[x0c])
             + fx * __half2float(r1[x1c]);
    out[((size_t)c * oh + y) * ow + x] = (TO)((1.f - fy) * t0 + fy * t1);
  }
}

// NHWC out: block per pixel, threads over channels -> coalesced writes
template <typename TO>
__global__ void k_upsample_nhwc(const __half* __restrict__ ws,
                                TO* __restrict__ out, int Cc, int lh, int lw,
                                int oh, int ow) {
  int pix = blockIdx.x, c = threadIdx.x;
  out[(size_t)pix * Cc + c] =
      (TO)upsample_one(ws, Cc, lh, lw, oh, ow, pix / ow, pix % ow, c);
}

inline cudaError_t launch_lift_fused(bool dp_nhwc, bool cx_nhwc,
                                     const __half* dlog, const __half* cx,
                                     const int* rowptr, const int* col,
                                     const int* cam, const float* ix,
                                     const float* iy, const int* b0,
                                     const float* fr, __half* ws,
                                     int D, int Cc, int Hf, int Wf, int G2,
                                     cudaStream_t s) {
  dim3 g(G2), b(Cc);
#define LF(A, B) k_lift_fused<A, B><<<g, b, 0, s>>>(dlog, cx, rowptr, col, \
    cam, ix, iy, b0, fr, ws, D, Cc, Hf, Wf)
  if (dp_nhwc) { if (cx_nhwc) LF(true, true); else LF(true, false); }
  else         { if (cx_nhwc) LF(false, true); else LF(false, false); }
#undef LF
  return cudaGetLastError();
}

inline cudaError_t launch_upsample(bool out_fp32, bool out_nhwc,
                                   const __half* ws, void* out, int Cc,
                                   int lh, int lw, int oh, int ow,
                                   cudaStream_t s) {
  if (out_nhwc) {
    dim3 g(oh * ow), b(Cc);
    if (out_fp32)
      k_upsample_nhwc<float><<<g, b, 0, s>>>(ws, (float*)out, Cc, lh, lw,
                                             oh, ow);
    else
      k_upsample_nhwc<__half><<<g, b, 0, s>>>(ws, (__half*)out, Cc, lh, lw,
                                              oh, ow);
  } else {
    if (lw > 128) return cudaErrorInvalidValue;   // shared row buffer bound
    dim3 g(oh, Cc), b(256);
    if (out_fp32)
      k_upsample_chw<float><<<g, b, 0, s>>>(ws, (float*)out, Cc, lh, lw,
                                            oh, ow);
    else
      k_upsample_chw<__half><<<g, b, 0, s>>>(ws, (__half*)out, Cc, lh, lw,
                                             oh, ow);
  }
  return cudaGetLastError();
}

inline cudaError_t launch_lift(bool dp_nhwc, bool cx_nhwc, bool out_nhwc,
                               const __half* dp, const __half* cx,
                               const int* rowptr, const int* col,
                               const int* cam, const float* ix,
                               const float* iy, const int* b0,
                               const float* fr, void* out,
                               int D, int Cc, int Hf, int Wf, int G2,
                               cudaStream_t s, bool out_f32 = false) {
  dim3 g(G2), b(Cc);
#define L(A, B) k_lift<A, B><<<g, b, 0, s>>>(dp, cx, rowptr, col, cam, ix, \
    iy, b0, fr, out, D, Cc, Hf, Wf, G2, out_nhwc, out_f32)
  if (dp_nhwc) { if (cx_nhwc) L(true, true); else L(true, false); }
  else         { if (cx_nhwc) L(false, true); else L(false, false); }
#undef L
  return cudaGetLastError();
}

}  // namespace liftk
