# METEOR Architecture v2 Orin microbench

Standalone CUDA probes for the structural primitives proposed in
`docs/PLAN_architecture_v2_2026-08-23.md`.

The benchmark uses the deployed rig's real sparse lift CSR table and avoids
materializing separate height-plane BEVs. It measures:

- camera-aware FiLM on the eight-camera stride-4 feature tensor;
- one-plane versus fused three-plane SurfaceLift (96 channels);
- one-plane versus fused three-plane ObjectLift (32 channels);
- one-slot Lane memory at 16/24/32 channels and 400x250;
- the current full-resolution 96-channel memory primitive for comparison;
- a fixed key-point-to-uint8 vector Lane rasterizer.

Build on AGX Orin:

```bash
cmake -S . -B build
cmake --build build -j
./build/archv2_bench ../liftbench/tables_r64
```
