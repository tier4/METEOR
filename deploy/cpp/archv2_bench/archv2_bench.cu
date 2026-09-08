// Standalone Orin microbenchmarks for METEOR Architecture v2.
//
// These kernels deliberately avoid framework overhead and materialized
// multi-plane BEVs.  They answer whether each proposed structural primitive
// fits the deploy latency budget before a training implementation exists.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#define CK(x) do { cudaError_t ck_err = (x); if (ck_err != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, \
          cudaGetErrorString(ck_err)); exit(1); } } while (0)

struct Result {
  std::string name;
  float mean, p50, p90, p99;
};

template <typename F>
Result bench(const std::string& name, F launch, int warmup=80, int iters=500) {
  for (int i = 0; i < warmup; ++i) launch();
  CK(cudaDeviceSynchronize());
  std::vector<cudaEvent_t> ev(iters + 1);
  for (auto& e : ev) CK(cudaEventCreate(&e));
  CK(cudaEventRecord(ev[0]));
  for (int i = 0; i < iters; ++i) {
    launch();
    CK(cudaEventRecord(ev[i + 1]));
  }
  CK(cudaEventSynchronize(ev.back()));
  std::vector<float> ms(iters);
  for (int i = 0; i < iters; ++i) CK(cudaEventElapsedTime(&ms[i], ev[i], ev[i + 1]));
  for (auto e : ev) CK(cudaEventDestroy(e));
  float mean = std::accumulate(ms.begin(), ms.end(), 0.f) / ms.size();
  std::sort(ms.begin(), ms.end());
  auto q = [&](float p) { return ms[std::min((int)ms.size()-1, (int)std::floor(p * ms.size()))]; };
  return {name, mean, q(.50f), q(.90f), q(.99f)};
}

template <typename T>
std::vector<T> load_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(2); }
  size_t n = (size_t)f.tellg() / sizeof(T);
  std::vector<T> v(n); f.seekg(0); f.read((char*)v.data(), n * sizeof(T));
  return v;
}

template <typename T>
T* dev_copy(const std::vector<T>& h) {
  T* d = nullptr; CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// ------------------------------------------------------------------ FiLM
__global__ void k_film(const __half* in, const __half* gamma,
                       const __half* beta, __half* out, int hw) {
  int nc = blockIdx.x;                 // 8 cameras * 160 channels
  float g = __half2float(gamma[nc]), b = __half2float(beta[nc]);
  size_t base = (size_t)nc * hw;
  for (int p = threadIdx.x; p < hw; p += blockDim.x) {
    float x = __half2float(in[base + p]);
    out[base + p] = __float2half((1.f + g) * x + b);
  }
}

// ------------------------------------------------------- sparse 3D lifts
struct Taps { int i[4]; float w[4]; };

__device__ __forceinline__ Taps taps(float ix, float iy, int h, int w) {
  int x0 = (int)floorf(ix), y0 = (int)floorf(iy);
  float fx = ix - x0, fy = iy - y0;
  Taps t;
  int xs[2] = {x0, x0 + 1}, ys[2] = {y0, y0 + 1};
  float wx[2] = {1.f - fx, fx}, wy[2] = {1.f - fy, fy};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int xx = xs[j & 1], yy = ys[j >> 1];
    bool ok = xx >= 0 && xx < w && yy >= 0 && yy < h;
    t.i[j] = ok ? yy * w + xx : 0;
    t.w[j] = ok ? wx[j & 1] * wy[j >> 1] : 0.f;
  }
  return t;
}

__device__ __forceinline__ float dweight(const __half* dp, int cam, int b0,
                                          float fr, const Taps& t,
                                          int d, int hw) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half* p = dp + ((size_t)cam * hw + t.i[j]) * d + b0;
    a += t.w[j] * (__half2float(p[0]) * (1.f-fr) + __half2float(p[1]) * fr);
  }
  return a + .05f;
}

__device__ __forceinline__ float context(const __half* cx, int cam, int c,
                                          const Taps& t, int channels, int hw) {
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j)
    a += t.w[j] * __half2float(cx[((size_t)cam * hw + t.i[j]) * channels + c]);
  return a;
}

// Reuses the real rig CSR sparsity and evaluates alternate height planes by
// shifted image samples.  All planes are reduced directly into one output.
__global__ void k_multiplane_lift(
    const __half* dp, const __half* cx, const int* rowptr, const int* col,
    const int* cam, const float* ix, const float* iy, const int* b0,
    const float* fr, __half* out, int channels, int planes,
    int depth_bins, int hf, int wf) {
  int cell = blockIdx.x, c = threadIdx.x;
  int r0 = rowptr[cell], k = rowptr[cell+1] - r0;
  __shared__ float sw[24];
  __shared__ Taps st[24];
  __shared__ int scam[24];
  int qn = planes * k;
  if (c < qn) {
    int pl = c / k, j = c - pl * k, p = col[r0 + j];
    scam[c] = cam[p];
    // A small shift models the fact that a new z plane projects to a nearby
    // image coordinate while preserving the real rig's sparse cell pattern.
    float off = (pl - (planes - 1) * .5f) * .35f;
    st[c] = taps(ix[p], iy[p] + off, hf, wf);
    int db = max(0, min(depth_bins - 2, b0[p] + pl - planes / 2));
    sw[c] = dweight(dp, scam[c], db, fr[p], st[c], depth_bins, hf * wf);
  }
  __syncthreads();
  if (c >= channels) return;
  float fused = 0.f, post_den = 0.f;
  for (int pl = 0; pl < planes; ++pl) {
    float acc = 0.f, den = 0.f;
    for (int j = 0; j < k; ++j) {
      int q = pl * k + j;
      den += sw[q];
      acc += context(cx, scam[q], c, st[q], channels, hf * wf) * sw[q];
    }
    float pw = planes == 1 ? 1.f : (pl == planes / 2 ? .5f : .25f);
    fused += pw * acc / fmaxf(den, 1e-4f); post_den += pw;
  }
  out[(size_t)cell * channels + c] = __float2half(fused / post_den);
}

// ------------------------------------------------ low-resolution map memory
__global__ void k_warp_gate(const __half* cur, const __half* prev,
                            const __half* gate, __half* out,
                            int h, int w, int c, float dx, float dy) {
  size_t n = (size_t)h * w * c;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n; idx += (size_t)blockDim.x * gridDim.x) {
    int ch = idx % c;
    size_t cell = idx / c;
    int x = cell % w, y = cell / w;
    float sx = x + dx, sy = y + dy;
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    float fx = sx - x0, fy = sy - y0, pv = 0.f;
    for (int oy = 0; oy < 2; ++oy) for (int ox = 0; ox < 2; ++ox) {
      int xx = x0 + ox, yy = y0 + oy;
      if (xx >= 0 && xx < w && yy >= 0 && yy < h) {
        float ww = (ox ? fx : 1.f-fx) * (oy ? fy : 1.f-fy);
        pv += ww * __half2float(prev[((size_t)yy * w + xx) * c + ch]);
      }
    }
    float g = __half2float(gate[cell]);
    float cv = __half2float(cur[idx]);
    out[idx] = __float2half(g * cv + (1.f-g) * pv);
  }
}

// ---------------------------------------------- deployment rasterizer probe
__global__ void k_raster_points(const short* px, const short* py,
                                const unsigned char* cls, int* mask,
                                int n, int h, int w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int x = px[i], y = py[i], v = cls[i];
  for (int oy = -2; oy <= 2; ++oy) for (int ox = -2; ox <= 2; ++ox) {
    int xx = x + ox, yy = y + oy;
    if (xx >= 0 && xx < w && yy >= 0 && yy < h)
      atomicMax(&mask[yy * w + xx], v);
  }
}

__global__ void k_i32_to_u8(const int* in, unsigned char* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) out[i] = (unsigned char)in[i];
}

int main(int argc, char** argv) {
  std::string table = argc > 1 ? argv[1] : "../liftbench/tables_r64";
  std::ifstream jf(table + "/meta.json");
  if (!jf) { fprintf(stderr, "missing %s/meta.json\n", table.c_str()); return 2; }
  nlohmann::json meta; jf >> meta;
  int ncam=meta["N"], d=meta["D"], hf=meta["Hf"], wf=meta["Wf"];

  auto hr = load_bin<int>(table + "/csr_rowptr.bin");
  auto hc = load_bin<int>(table + "/csr_col.bin");
  auto hcam = load_bin<int>(table + "/pair_cam.bin");
  auto hix = load_bin<float>(table + "/pair_ix.bin");
  auto hiy = load_bin<float>(table + "/pair_iy.bin");
  auto hb0 = load_bin<int>(table + "/pair_b0.bin");
  auto hfr = load_bin<float>(table + "/pair_fr.bin");
  int cells = (int)hr.size() - 1;
  int *dr=dev_copy(hr), *dc=dev_copy(hc), *dcam=dev_copy(hcam), *db0=dev_copy(hb0);
  float *dix=dev_copy(hix), *diy=dev_copy(hiy), *dfr=dev_copy(hfr);

  std::vector<Result> results;

  // FiLM over the actual eight-camera stride-4 feature tensor.
  const int film_hw=108*192, film_nc=8*160;
  __half *fi, *fo, *fg, *fb;
  CK(cudaMalloc(&fi, (size_t)film_nc*film_hw*2));
  CK(cudaMalloc(&fo, (size_t)film_nc*film_hw*2));
  CK(cudaMalloc(&fg, (size_t)film_nc*2)); CK(cudaMalloc(&fb, (size_t)film_nc*2));
  CK(cudaMemset(fi, 1, (size_t)film_nc*film_hw*2)); CK(cudaMemset(fg,0,(size_t)film_nc*2)); CK(cudaMemset(fb,0,(size_t)film_nc*2));
  results.push_back(bench("film_8x160x108x192_fp16", [&]{ k_film<<<film_nc,256>>>(fi,fg,fb,fo,film_hw); }));
  const int depth_nc=8*64;
  results.push_back(bench("calib_depth_logits_8x64x108x192_fp16", [&]{
    k_film<<<depth_nc,256>>>(fi,fg,fb,fo,film_hw);
  }));

  // Real-rig sparse fused lift.  The delta from 1 to 3 planes is the deploy
  // price; no 3x output tensor or concat is ever allocated.
  __half *dp, *cx96, *cx32, *lift96, *lift32;
  CK(cudaMalloc(&dp, (size_t)ncam*hf*wf*d*2));
  CK(cudaMalloc(&cx96, (size_t)ncam*hf*wf*96*2));
  CK(cudaMalloc(&cx32, (size_t)ncam*hf*wf*32*2));
  CK(cudaMalloc(&lift96, (size_t)cells*96*2)); CK(cudaMalloc(&lift32, (size_t)cells*32*2));
  CK(cudaMemset(dp, 0, (size_t)ncam*hf*wf*d*2)); CK(cudaMemset(cx96,1,(size_t)ncam*hf*wf*96*2)); CK(cudaMemset(cx32,1,(size_t)ncam*hf*wf*32*2));
  auto lift = [&](int ch, int planes, __half* cx, __half* out) {
    k_multiplane_lift<<<cells,128>>>(dp,cx,dr,dc,dcam,dix,diy,db0,dfr,out,ch,planes,d,hf,wf);
  };
  results.push_back(bench("surface_lift_1plane_96ch_core", [&]{lift(96,1,cx96,lift96);}));
  results.push_back(bench("surface_lift_3plane_96ch_fused_core", [&]{lift(96,3,cx96,lift96);}));
  results.push_back(bench("object_lift_1plane_32ch_core", [&]{lift(32,1,cx32,lift32);}));
  results.push_back(bench("object_lift_3plane_32ch_fused_core", [&]{lift(32,3,cx32,lift32);}));

  // Task-specific one-slot memory candidates and the current full-size
  // primitive for a directly comparable bandwidth measurement.
  auto memory_case = [&](int h, int w, int c, const std::string& name) {
    size_t nc=(size_t)h*w*c, ng=(size_t)h*w;
    __half *cur,*prev,*gate,*out;
    CK(cudaMalloc(&cur,nc*2)); CK(cudaMalloc(&prev,nc*2)); CK(cudaMalloc(&gate,ng*2)); CK(cudaMalloc(&out,nc*2));
    CK(cudaMemset(cur,1,nc*2)); CK(cudaMemset(prev,1,nc*2)); CK(cudaMemset(gate,0,ng*2));
    int blocks=std::min(4096, (int)((nc+255)/256));
    results.push_back(bench(name,[&]{k_warp_gate<<<blocks,256>>>(cur,prev,gate,out,h,w,c,.35f,-.25f);}));
    CK(cudaFree(cur)); CK(cudaFree(prev)); CK(cudaFree(gate)); CK(cudaFree(out));
  };
  memory_case(400,250,16,"lane_memory_1slot_16ch_400x250");
  memory_case(400,250,24,"lane_memory_1slot_24ch_400x250");
  memory_case(400,250,32,"lane_memory_1slot_32ch_400x250");
  memory_case(800,500,96,"legacy_memory_1slot_96ch_800x500");

  // 256 chains x 12 key-points; deployment returns the existing uint8 map.
  const int chains=256, pts=12, np=chains*pts, rh=800, rw=500;
  std::vector<short> hpx(np), hpy(np); std::vector<unsigned char> hcl(np);
  for (int i=0;i<np;++i) { hpx[i]=(short)((i*37)%rw); hpy[i]=(short)((i*19)%rh); hcl[i]=(unsigned char)(4+(i%3)); }
  short *dpx=dev_copy(hpx), *dpy=dev_copy(hpy); unsigned char* dcl=dev_copy(hcl);
  int *mask; unsigned char *map; CK(cudaMalloc(&mask,(size_t)rh*rw*4)); CK(cudaMalloc(&map,(size_t)rh*rw));
  results.push_back(bench("vector_raster_256x12_to_u8_800x500", [&]{
    CK(cudaMemsetAsync(mask,0,(size_t)rh*rw*4));
    k_raster_points<<<(np+255)/256,256>>>(dpx,dpy,dcl,mask,np,rh,rw);
    k_i32_to_u8<<<512,256>>>(mask,map,rh*rw);
  }));

  CK(cudaDeviceSynchronize());
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,0));
  printf("# device=%s sm=%d.%d table=%s cells=%d pairs=%zu\n", prop.name, prop.major, prop.minor, table.c_str(), cells, hcam.size());
  printf("name,mean_ms,p50_ms,p90_ms,p99_ms\n");
  for (const auto& r:results) printf("%s,%.6f,%.6f,%.6f,%.6f\n",r.name.c_str(),r.mean,r.p50,r.p90,r.p99);
  return 0;
}
