# 推論クイックスタート (Hugging Face の公開物から)

このドキュメントだけで、公開済みのモデルとデモシーンを取得し、GPU 1 枚で 12 タスクの推論動画を得られます。
手順はワークステーション (データセンター向け NVIDIA GPU 1 枚、TensorRT 8.6) で実際に通したものです。英語版の詳細は
[REPRODUCE.md §8](REPRODUCE.md#8-deployment--the-verified-release-path)。

## 1. 公開物

| | 場所 | 内容 |
|---|---|---|
| モデル | <https://huggingface.co/AutowareFoundation/meteor> | `meteor_v157c3Z.onnx` (素の ONNX、カメラのみ、カスタム演算なし)、`meteor_v157.pt` (PyTorch チェックポイント)、`meteor_v157.param.yaml` (カメラ順・解像度・BEV 格子・出力一覧)、`lift_plugin_tables_r64/` (Orin 用リフト・プラグインのテーブル) |
| デモシーン | <https://huggingface.co/datasets/AutowareFoundation/meteor-demo-scenes> | 匿名化済み 6 シーン (高速・山道・幹線・市街地・夜間高架・ベンチ用)。各シーン `manifest.json` / `img/` (8 カメラ × 147 フレーム、768×432) / `ego_motion.npz` / `lidar_bev/`。GT なし |

ライセンス: モデル Apache-2.0、デモシーン CC-BY-4.0。TensorRT エンジンは GPU と TensorRT の版に依存するため配布せず、ONNX から各自ビルドします。

## 2. 必要なもの

| 項目 | 条件 |
|---|---|
| GPU | NVIDIA、fp16 TensorRT 推論で VRAM 4 GB 以上 (エンジン構築時に workspace 8 GB を指定) |
| Python | 3.10 |
| パッケージ | `huggingface_hub`, `onnxruntime`, `opencv-python`, `numpy`; TensorRT 推論には `tensorrt` (8.6 以上) と `pycuda`; 微調整・再 export には `torch` |
| その他 | 動画確認用に `ffmpeg` があると便利 |

## 3. 手順

```bash
# 0. コードと公開物
git clone https://github.com/tier4/METEOR && cd METEOR
pip install -U huggingface_hub onnxruntime opencv-python numpy
hf download AutowareFoundation/meteor --local-dir models
hf download AutowareFoundation/meteor-demo-scenes --repo-type dataset --local-dir data
sha256sum -c models/SHA256SUMS --ignore-missing && (cd data && sha256sum -c SHA256SUMS --quiet)

# 1. ONNX が動くか (CPU、1 フレーム約 4 秒)。全出力の形を表示し "SMOKE PASS" で終わる
python3 hf/onnx_smoke_test.py --onnx models/meteor_v157c3Z.onnx --root data/valday --frame 40

# 2. TensorRT fp16 エンジン (プラグイン・較正なし、約 10 分、194 MB)
pip install tensorrt pycuda
python3 deploy/build_engine_fp16.py models/meteor_v157c3Z.onnx out/meteor_v157_fp16.engine 8

# 3. 6 シーンのデモ動画 (Orin と同じ描画。ワークステーションで推論 30 ms、描画込み約 20 FPS)
mkdir -p out/demo6 && for r in highway_day mountain_day arterial_day valday valcurve fast; do \
  s=$(cat data/$r/scenes.txt); ln -sfn $PWD/data/$r/$s out/demo6/; echo $s >> out/demo6/scenes.txt; done
METEOR_TH2D=0.30 METEOR_SEG2D_OVERLAY=0 METEOR_OCC_PANEL=0 METEOR_2D_HIDE=7 PYTHONPATH=. \
python3 deploy/orin_realtime.py --engine out/meteor_v157_fp16.engine --root out/demo6 --out out/demo6.mp4
```

終了時に pycuda が「context stack was not empty」と出ることがありますが無害で、動画は完成しています。
1 シーンだけ試すなら `--root data/valday` のようにルートを直接指定します。

## 4. その先

- **Jetson AGX Orin (INT8、約 70 ms)**: `make_plugin_onnx.py --tables models/lift_plugin_tables_r64` でリフト・プラグインを挿入し、
  Orin 上で `deploy/orin_build_int8.py` (実フレーム較正) でビルド。手順と落とし穴は [deploy/README.md](../deploy/README.md) §6–§7。
- **再 export / 微調整**: `deploy/export_onnx.py --ckpt models/meteor_v157.pt --model v52 --n-cams 8 …` (フラグは deploy/README.md §1)。
  `--with-lidar` で LiDAR 任意入力版が出ます。学習は [TRAINING.md](TRAINING.md)。
- **入力の作り方**: 画像は RGB uint8 `[1,8,3,432,768]`、カメラ順は `meteor_v157.param.yaml` の `input.cameras`、
  `K` は 768×432 での内部パラメータ、`T_cam_ego` は ego→camera、`v0` は m/s。`hf/onnx_smoke_test.py` が実装例です。

## 5. 描画の環境変数 (主なもの)

| 変数 | 既定 | 意味 |
|---|---|---|
| `METEOR_TH2D` | 0.30 | 2D 検出の表示閾値 |
| `METEOR_SEG2D_OVERLAY` | 0 | 2D タイルへのセグ重畳 (1 で表示) |
| `METEOR_OCC_PANEL` | 0 | 3D 占有パネル (1 で表示、描画が重くなる) |
| `METEOR_2D_HIDE` | 7 | 描かない 2D クラス (7 = 路面ペイント) |
| `METEOR_CUDAGRAPH` | 0 | CUDA Graph でエンジンを叩く (Orin で使用) |
| `METEOR_LIDAR` | 0 | `lidar_bev/` を LiDAR 入力版エンジンに供給 |
