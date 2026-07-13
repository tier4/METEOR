#!/usr/bin/env python3
"""Comprehensive overview PPTX: algorithm, architecture, cost, how-to-run, eval."""
import cv2
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from lxml import etree

SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30); ACC = RGBColor(0x0E, 0x6E, 0xB8)
ORN = RGBColor(0xE8, 0x83, 0x3A); GRAY = RGBColor(0x60, 0x68, 0x70)
BB = RGBColor(0xE8, 0xF0, 0xF8); BO = RGBColor(0xFD, 0xEE, 0xDE)
AI_FILL = RGBColor(0xF7, 0xCE, 0x9C)     # learned NN blocks (AI計算)
OP_FILL = RGBColor(0xCF, 0xE2, 0xF3)     # fixed ops / geometry (no params)
GATE_FILL = RGBColor(0xF3, 0xC9, 0xC9)   # gating / fusion point
MONO = "Consolas"
prs = Presentation(); prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def add_slide(title):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.2), Inches(12.4), Inches(0.7))
    p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(23); p.font.bold = True; p.font.color.rgb = DARK
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(0.88), Inches(12.4), Emu(1))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACC; ln.line.fill.background()
    return s


def bullets(slide, items, x, y, w, h, size=14, mono=False):
    tb = slide.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple):
            lvl, it = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = it if mono else ("・" if lvl == 0 else "－ ") + it
        p.font.size = Pt(size if lvl == 0 else size - 1)
        p.font.color.rgb = DARK if lvl == 0 else GRAY
        if mono:
            p.font.name = MONO
        p.space_after = Pt(1 if mono else 4)


def caption(slide, text, x, y, w, size=11, align=PP_ALIGN.CENTER):
    tb = slide.shapes.add_textbox(x, y, w, Inches(0.35)); p = tb.text_frame.paragraphs[0]
    p.text = text; p.font.size = Pt(size); p.font.color.rgb = GRAY; p.alignment = align


def pic(slide, path, x, y, mw, mh, cap=None):
    im = cv2.imread(path)
    if im is None:
        return 0
    h, w = im.shape[:2]; sc = min(mw / w, mh / h); pw, ph = int(w * sc), int(h * sc)
    slide.shapes.add_picture(path, x + (mw - pw) // 2, y, width=pw, height=ph)
    if cap:
        caption(slide, cap, x, y + ph + Emu(20000), mw)
    return ph


def box(slide, x, y, w, h, title, sub="", fill=BB):
    b = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.color.rgb = GRAY; b.line.width = Pt(1)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(11); p.font.bold = True
    p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
    if sub:
        p2 = tf.add_paragraph(); p2.text = sub; p2.font.size = Pt(8.5)
        p2.font.color.rgb = GRAY; p2.alignment = PP_ALIGN.CENTER


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = DARK; c.line.width = Pt(1.6)
    le = c.line._get_or_add_ln()
    etree.SubElement(le, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd').set('type', 'arrow')


def table(slide, rows, widths, x, y, w, h, fs=11):
    t = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, h).table
    for c, wd in enumerate(widths):
        t.columns[c].width = Inches(wd)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(r, c); cell.text = v; pp = cell.text_frame.paragraphs[0]
            pp.font.size = Pt(fs); pp.font.bold = (r == 0 or (c == 0 and r == len(rows) - 1))
            pp.font.color.rgb = RGBColor(255, 255, 255) if r == 0 else DARK


# 1 title
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.7), Inches(2.0))
p = tb.text_frame.paragraphs[0]
p.text = "BEVLane 総合概要"
p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = DARK
p2 = tb.text_frame.add_paragraph()
p2.text = "アルゴリズム / モデル構造 / パラメータ・FLOPs / 動かし方(TRT含む) / 定性評価 / GTなしデモ"
p2.font.size = Pt(17); p2.font.color.rgb = ACC
p3 = tb.text_frame.add_paragraph()
p3.text = "6/8カメラ → ロングレンジBEV (前後±80m×左右±50m) 意味セグ + 深度"
p3.font.size = Pt(15); p3.font.color.rgb = GRAY
caption(s, "2026-07-08  現行モデル v12 (DepthGatedIPMNet, 8カメラ)", Inches(0.8), Inches(6.4), Inches(9), 14, PP_ALIGN.LEFT)

# 2 pipeline / algorithm
s = add_slide("アルゴリズム全体像 (Autolabel → 学習)")
bullets(s, [
    "【Autolabel (GT自動生成)】 t4dataset の 2D Panoptic + LiDAR + ego_pose から3D GTを合成",
    (1, "LiDAR点を全カメラへ投影しPanopticラベルを転写 → マップ座標で全フレーム累積 (0.1m)"),
    (1, "ラスタBEV → ベクタ化 (スケルトン/破線連結/polyfit) → nuScenes形式。1120シーン量産"),
    "【学習GT (ego中心)】 各キーフレームで ±80m×±50m (800×500@0.2m) に切り出し",
    (1, "面クラス=Segmentation, 線クラス=固定幅ライン (laneline/stopline/road_edge)"),
    (1, "road_edge = road面の外縁で非road領域が連続する箇所 / marking除去 / 対向車線除去"),
    (1, "深度GT: LiDAR+Panopticで稠密化 (stride4, 64bin@1.25m, sky=79.5m, 自車=2m)"),
    "【推論モデル】 6/8カメラRGB + キャリブ → 深度ゲートIPMでBEVへ → 9クラスセグ + 深度",
    "【損失】 CE(bg学習) + Lovász(IoU直接) + 境界重み + Tversky(線細線化) + 遠方重み + 深度CE",
], Inches(0.7), Inches(1.05), Inches(12.2), Inches(6.0), size=14)

# 3 architecture diagram (AI blocks orange, fixed-op/geometry blue)
s = add_slide("モデル構造: DepthGatedIPMNet (TRT対応 pull型)")
y = Inches(1.35)
box(s, Inches(0.4), y, Inches(1.5), Inches(0.85), "6/8カメラ画像", "768x432 (入力)", fill=RGBColor(0xEE, 0xEE, 0xEE))
box(s, Inches(2.2), y, Inches(1.9), Inches(0.85), "ResNet34 + FPN", "stride4 特徴 (共有)", fill=AI_FILL)
box(s, Inches(4.5), y - Inches(0.15), Inches(2.0), Inches(0.7), "深度分布 64bin", "画素毎softmax / LiDAR監督", fill=AI_FILL)
box(s, Inches(4.5), y + Inches(0.72), Inches(2.0), Inches(0.55), "Context特徴 96ch", "", fill=AI_FILL)
box(s, Inches(6.9), y, Inches(2.9), Inches(0.85), "深度ゲートIPM (pull型)",
    "投影+grid_sample+深度ゲート\n★学習パラメータ無し(幾何+固定演算)", fill=OP_FILL)
box(s, Inches(10.2), y, Inches(1.5), Inches(0.85), "BEV特徴", "800x500 @0.2m", fill=OP_FILL)
box(s, Inches(11.9), y, Inches(1.0), Inches(0.85), "デコーダ", "9クラス", fill=AI_FILL)
arrow(s, Inches(1.9), y + Inches(0.42), Inches(2.2), y + Inches(0.42))
arrow(s, Inches(4.1), y + Inches(0.42), Inches(4.5), y + Inches(0.3))
arrow(s, Inches(6.5), y + Inches(0.2), Inches(6.9), y + Inches(0.35))
arrow(s, Inches(6.5), y + Inches(1.0), Inches(6.9), y + Inches(0.6))
arrow(s, Inches(9.8), y + Inches(0.42), Inches(10.2), y + Inches(0.42))
arrow(s, Inches(11.7), y + Inches(0.42), Inches(11.9), y + Inches(0.42))
box(s, Inches(4.5), y + Inches(1.5), Inches(2.0), Inches(0.5), "カメラ校正 K/T", "モデル入力(幾何)", fill=OP_FILL)
arrow(s, Inches(5.5), y + Inches(1.5), Inches(6.9), y + Inches(0.75))
# legend
box(s, Inches(9.9), y + Inches(1.45), Inches(1.4), Inches(0.42), "AI計算", "学習パラメータ", fill=AI_FILL)
box(s, Inches(11.4), y + Inches(1.45), Inches(1.5), Inches(0.42), "幾何/固定演算", "パラメータ無", fill=OP_FILL)
bullets(s, [
    "設計方針: 深度で「持ち上げてscatter」ではなく、BEVセルから画像へ「引き込む(pull)」+深度でゲート",
    (1, "使う演算は conv / grid_sample / gather のみ → TensorRT完全対応 (scatter不要)"),
    "橙=AI計算(学習パラメータ有, GPU/TRTで実行) / 青=幾何・固定演算(パラメータ無し)。次頁で詳細",
    "深度は中間表現として学習 (LiDAR+Panoptic稠密GTで監督)。平面仮定に頼らず遠方・立体に頑健",
], Inches(0.6), Inches(3.7), Inches(12.4), Inches(3.2), size=13)

# 3b detailed depth-gated IPM block diagram
s = add_slide("深度ゲートIPM 詳細ブロックダイアグラム")
# why-box
wb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45), Inches(1.0),
                        Inches(12.45), Inches(1.15))
wb.fill.solid(); wb.fill.fore_color.rgb = RGBColor(0xF7, 0xF9, 0xFB); wb.line.color.rgb = GRAY
tf = wb.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.15)
for i, (txt, col, bold) in enumerate([
    ("なぜ深度でゲートするのか?", ACC, True),
    ("通常のIPMは「地面(z=0)」を仮定してBEVセルを画像へ逆投影し特徴を取る → 地面より高い立体物(建物・車・ガードレール)が地面に引き伸ばされて滲む。", DARK, False),
    ("深度ゲートIPM: 各画素の深度分布もAIで予測し、「そのセルのカメラからの距離 dist」と「その画素の予測深度」が一致する確率で特徴を重み付け(ゲート)。距離が合わない特徴を抑制 → 滲み除去・立体に頑健。", DARK, False),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = txt; p.font.size = Pt(12 if i == 0 else 11); p.font.bold = bold; p.font.color.rgb = col

by = Inches(2.45)
BH = Inches(0.62)
# Lane 1: Context (AI) -> sample
box(s, Inches(0.45), by, Inches(1.7), BH, "1カメラ画像", "768x432", fill=RGBColor(0xEE, 0xEE, 0xEE))
box(s, Inches(2.5), by, Inches(2.2), BH, "Backbone+Context head", "特徴 C=96ch", fill=AI_FILL)
box(s, Inches(5.05), by, Inches(2.15), BH, "grid_sample @(u,v)", "→ ctx_s (セルの特徴)", fill=OP_FILL)
# Lane 2: Depth (AI) -> sample -> gather
box(s, Inches(2.5), Inches(3.25), Inches(2.2), BH, "Depth head", "深度分布 64bin softmax", fill=AI_FILL)
box(s, Inches(5.05), Inches(3.25), Inches(2.15), BH, "grid_sample @(u,v)", "→ prob_s (深度分布)", fill=OP_FILL)
box(s, Inches(7.55), Inches(3.25), Inches(2.4), BH, "gather + 線形補間", "→ P(深度 = dist)", fill=OP_FILL)
# Lane 3: geometry
box(s, Inches(0.45), Inches(4.05), Inches(1.7), BH, "BEVセル (x,y,0)", "ego座標", fill=RGBColor(0xEE, 0xEE, 0xEE))
box(s, Inches(2.5), Inches(4.05), Inches(2.2), BH, "幾何投影 K・T", "→ 画素 (u,v)", fill=OP_FILL)
box(s, Inches(5.05), Inches(4.05), Inches(2.15), BH, "レイ距離 dist", "= ‖セル − カメラ‖", fill=OP_FILL)
# gate + aggregate
box(s, Inches(7.55), by, Inches(2.4), BH, "ゲート重み w", "= P(深度=dist) + 0.05", fill=GATE_FILL)
box(s, Inches(10.25), Inches(2.85), Inches(2.65), BH, "全カメラ加重平均", "Σ(ctx_s·w) / Σ w", fill=OP_FILL)
box(s, Inches(10.25), Inches(3.75), Inches(2.65), BH, "BEVデコーダ", "→ 9クラスBEV", fill=AI_FILL)
# arrows
arrow(s, Inches(2.15), by + Inches(0.31), Inches(2.5), by + Inches(0.31))
arrow(s, Inches(4.7), by + Inches(0.31), Inches(5.05), by + Inches(0.31))
arrow(s, Inches(7.2), by + Inches(0.31), Inches(7.55), by + Inches(0.31))
arrow(s, Inches(4.7), Inches(3.56), Inches(5.05), Inches(3.56))
arrow(s, Inches(7.2), Inches(3.56), Inches(7.55), Inches(3.56))
arrow(s, Inches(8.75), Inches(3.25), Inches(8.75), by + Inches(0.62))     # P -> gate
arrow(s, Inches(2.15), Inches(4.36), Inches(2.5), Inches(4.36))
arrow(s, Inches(4.7), Inches(4.36), Inches(5.05), Inches(4.36))
arrow(s, Inches(3.6), Inches(4.05), Inches(3.6), by + Inches(0.62))       # (u,v) feeds samples
arrow(s, Inches(6.1), Inches(4.05), Inches(6.1), by + Inches(0.62))
arrow(s, Inches(6.1), Inches(4.05), Inches(6.1), Inches(3.25) + Inches(0.62))
arrow(s, Inches(9.95), by + Inches(0.31), Inches(10.25), Inches(2.85) + Inches(0.31))  # w+ctx -> agg
arrow(s, Inches(11.55), Inches(3.47), Inches(11.55), Inches(3.75))
# legend + key point
box(s, Inches(0.45), Inches(5.0), Inches(2.0), Inches(0.45), "AI計算", "学習パラメータ有", fill=AI_FILL)
box(s, Inches(2.6), Inches(5.0), Inches(2.6), Inches(0.45), "幾何/固定演算", "パラメータ無(投影・sample・集約)", fill=OP_FILL)
box(s, Inches(5.35), Inches(5.0), Inches(2.0), Inches(0.45), "ゲート(融合)", "深度×距離", fill=GATE_FILL)
# glossary (right of legend)
gl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.55), Inches(4.9),
                        Inches(5.35), Inches(0.72))
gl.fill.solid(); gl.fill.fore_color.rgb = RGBColor(0xF7, 0xF9, 0xFB); gl.line.color.rgb = GRAY
tf = gl.text_frame; tf.word_wrap = True; tf.margin_top = Inches(0.03); tf.margin_left = Inches(0.1)
for i, (lab, txt) in enumerate([
    ("用語", ""),
    ("Context特徴 96ch", "= 各画素の見た目・意味を表す特徴(BEVへ運ぶ「中身」)。深度はそれを『どの距離に置くか』の重み"),
    ("カメラ校正 K/T", "= 各カメラのキャリブレーション入力(学習しない)。K:内部(焦点距離・主点) / T:外部(車両↔カメラの位置姿勢)"),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    r1 = p.add_run(); r1.text = (lab if i == 0 else lab + " ")
    r1.font.bold = True; r1.font.size = Pt(9.5)
    r1.font.color.rgb = ACC if i == 0 else DARK
    if txt:
        r2 = p.add_run(); r2.text = txt; r2.font.size = Pt(8.8); r2.font.color.rgb = GRAY
bullets(s, [
    "AI計算は3ブロックのみ: ①Backbone+Context ②Depth head ③BEVデコーダ (これらだけが学習・重みを持つ)",
    "残りは全て幾何・固定演算(パラメータ無し): カメラ投影 K·T / grid_sample / レイ距離 / gather補間 / 加重平均",
    "→ scatter を使わないため TensorRT で完全に実装可能。深度分布が「どの距離の特徴か」を選ぶ弁の役割",
], Inches(0.5), Inches(5.75), Inches(12.4), Inches(1.6), size=12.5)

# 4 params / flops
s = add_slide("パラメータ・計算量・レイテンシ")
table(s, [
    ("モジュール", "構成", "パラメータ", "MACs (GFLOPs=2xMACs)"),
    ("Backbone", "ResNet34 (共有, 8視点)", "21.3 M", "86 G"),
    ("FPN", "stride4 lateral+fuse", "0.4 M", "18 G"),
    ("深度ヘッド", "64bin 深度分布", "0.24 M", "18 G"),
    ("IPM射影", "grid_sample+gather (学習パラメータ無)", "0", "≈0"),
    ("BEVデコーダ", "800x500全解像度 ConvBlock x4", "1.16 M", "465 G"),
    ("合計", "8カメラ 512x288 → BEV 800x500x9 + 深度", "23.1 M", "588 GMACs = 1177 GFLOPs"),
], [2.0, 4.6, 1.8, 3.6], Inches(0.6), Inches(1.05), Inches(12.1), Inches(3.0), fs=12)
bullets(s, [
    "計算量はBEVデコーダ(800x500全解像度)が支配的。解像度を下げれば大幅削減可能",
    "TensorRT (L40S, 8カメラ ±80×±50m, ランダム重み検証): fp16 ≈ 38 ms (単独GPU)",
    (1, "※学習と同一GPUでの計測は競合で約2倍に膨張 (81ms) → 完了後に単独GPUで確定計測"),
    "6カメラ・±30m版 (旧構成) では TRT INT8 1.5ms 実測 — 構成で速度は大きく変わる",
    "車載SoC(Orin級)想定: 解像度/バックボーン調整 + INT8 で数十msが目標レンジ",
], Inches(0.7), Inches(4.3), Inches(12.2), Inches(2.6), size=13)

# 5 how to run
s = add_slide("動かし方 (学習・推論・TRT)")
cmds = [
    "# 1) Autolabel GT 量産 (ラスタ→ベクタ→nuScenes)",
    "python3 run_batch.py --scenes <list> --out out/production --workers 16",
    "python3 vectorize_bev.py out/production/<scene>        # ベクタGT",
    "",
    "# 2) 学習サンプル抽出 (ego中心BEV GT + 8カメラ画像 + 深度GT)",
    "python3 bevlane/extract_gt.py --stride 2 --workers 24",
    "python3 bevlane/add_narrow_cams.py --workers 24        # 望遠追加(8カメラ)",
    "python3 bevlane/render_vector_gt.py --stride 2 --workers 24   # 線GT",
    "python3 bevlane/extract_depth_dense.py --stride 2 --workers 32  # 深度GT",
    "",
    "# 3) 学習 (8GPU DDP)",
    "torchrun --nproc_per_node=8 bevlane/train.py --model v8 \\",
    "   --train-bg --aug --dice-w .5 --lovasz-w .4 --boundary-w 3 \\",
    "   --tversky-w .6 --far-w 1 --depth-w .3 --out out/ckpt",
    "",
    "# 4) 推論可視化 / GTなしデモ動画 (RGB+Depth+BEV)",
    "python3 bevlane/infer.py --ckpt out/ckpt/best.pt --model v8 --scene <s>",
    "python3 bevlane/demo_rgbd_bev.py --ckpt out/ckpt/best.pt --scenes <s...>",
    "",
    "# 5) TensorRT 化 (ONNX→fp16/INT8エンジン + ベンチ + パリティ)",
    "python3 bevlane/export_trt.py --ckpt out/ckpt/best.pt --model v8 \\",
    "   --out out/trt      # 8カメラ入力, INT8は実データでcalib",
]
box0 = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.05), Inches(12.3), Inches(5.9))
box0.fill.solid(); box0.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xF8); box0.line.color.rgb = GRAY
bullets(s, cmds, Inches(0.7), Inches(1.15), Inches(12.0), Inches(5.7), size=11.5, mono=True)

# 6 qualitative
s = add_slide("定性評価 (v12, 未見データ)")
pic(s, "out/v12_cmp/highway_378.jpg", Inches(0.3), Inches(1.05), Inches(12.8), Inches(2.4),
    "高速道路378: 6カメラ | GT | 予測 — road_edge(橙)がガードレール沿いに, markingノイズ無し")
pic(s, "out/v12_cmp/urban_156.jpg", Inches(0.3), Inches(3.65), Inches(12.8), Inches(2.4),
    "市街地156: 交差点構造・横断歩道・road_edge・レーンラインを再現")
caption(s, "val mIoU 0.286 (未見収録日) / road P0.55 R0.69 / road_edge 細線 (pred厚 1.7x GT) / 深度中央値誤差 ~0.4m",
        Inches(0.5), Inches(6.15), Inches(12.4), 12, PP_ALIGN.LEFT)

# 7 demo + status
s = add_slide("GTなしデモ・成果物・状況")
pic(s, "out/demo_rgbd_frame.png", Inches(0.5), Inches(1.05), Inches(7.4), Inches(4.6),
    "GTなしデモ: RGB入力(8カメラ) + 予測Depth(turbo) + 予測BEV")
bullets(s, [
    "デモ動画 (GTなし, 展開ビュー):",
    (1, "out/demo_rgbd_bev.mp4 — RGB + Depth + BEV"),
    "推論動画 (GT比較):",
    (1, "out/infer_video_v12_highway.mp4 (高速道路)"),
    "PPTX: bevlane_overview / v12_gt / v12_status ほか",
    "モデル: out/bevlane_ckpt_v12/best.pt",
    "データ: out/bevlane/ (画像+BEV GT+線GT+深度GT)",
    "",
    "状況: v12学習 ep6/14 (mIoU 0.286) 継続中",
    (1, "完了後に単独GPUでTRT fp16/INT8を確定計測"),
], Inches(8.1), Inches(1.15), Inches(4.9), Inches(5.8), size=13)

prs.save("out/bevlane_overview.pptx")
print(f"saved out/bevlane_overview.pptx ({len(prs.slides._sldIdLst)} slides)")
