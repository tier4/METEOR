# METEOR C++ TensorRT runtime

Self-contained C++ deployment: engine build (from ONNX, cached fp16),
raw-t4dataset parsing, training-identical preprocessing, the 3-slot
temporal memory ring, box/lane/ego decoding, and a lightweight overlay
video. Optional inputs (lidar_bev / kin) are zero-filled — camera-only
mode is bit-equal to training by construction. The full 12-panel
visualisation remains in Python (`deploy/visualize.py`).

## Build
```bash
cd deploy/cpp && mkdir -p build && cd build
cmake .. -DTENSORRT_DIR=/path/to/TensorRT   # or export TENSORRT_DIR
make -j
```
Dependencies: TensorRT >= 8.6 (C++), CUDA, OpenCV 4, nlohmann-json.

## Run
```bash
./meteor_infer --onnx  ../../out/meteor_v36.onnx \
               --t4d   /path/to/t4dataset/<scene> \
               --video out.mp4 [--limit N]
# or with a prebuilt engine: --engine meteor_v36_fp16.engine
```
