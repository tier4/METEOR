#!/usr/bin/env python3
"""METEOR v18 architecture PPTX (overview + detailed blocks + losses + specs)."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from lxml import etree

SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30); ACC = RGBColor(0x0E, 0x6E, 0xB8)
GRAY = RGBColor(0x60, 0x68, 0x70)
AI = RGBColor(0xF7, 0xCE, 0x9C)      # learned blocks
OP = RGBColor(0xCF, 0xE2, 0xF3)      # fixed ops / geometry
OUTC = RGBColor(0xD9, 0xEF, 0xD9)    # task outputs
E2E = RGBColor(0xE8, 0xD5, 0xF2)     # E2E head (new)
prs = Presentation(); prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def add_slide(title):
    s = prs.slides.add_slide(BLANK)
    tb = s.shapes.add_textbox(Inches(0.45), Inches(0.18), Inches(12.4), Inches(0.6))
    p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(22); p.font.bold = True; p.font.color.rgb = DARK
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.45), Inches(0.8), Inches(12.4), Emu(1))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACC; ln.line.fill.background()
    return s


def box(s, x, y, w, h, title, sub="", fill=AI, fs=11):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    b.fill.solid(); b.fill.fore_color.rgb = fill; b.line.color.rgb = GRAY; b.line.width = Pt(1)
    tf = b.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(fs); p.font.bold = True
    p.font.color.rgb = DARK; p.alignment = PP_ALIGN.CENTER
    if sub:
        p2 = tf.add_paragraph(); p2.text = sub; p2.font.size = Pt(8.5)
        p2.font.color.rgb = GRAY; p2.alignment = PP_ALIGN.CENTER
    return b


def arrow(s, x1, y1, x2, y2):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = DARK; c.line.width = Pt(1.6)
    le = c.line._get_or_add_ln()
    etree.SubElement(le, '{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd').set('type', 'arrow')


def bullets(s, items, x, y, w, h, size=12.5):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 0
        if isinstance(it, tuple):
            lvl, it = it
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("・" if lvl == 0 else "－ ") + it
        p.font.size = Pt(size if lvl == 0 else size - 1)
        p.font.color.rgb = DARK if lvl == 0 else GRAY
        p.space_after = Pt(3)


# ================= slide 1: overview =================
s = add_slide("METEOR v18 — 6タスク マルチタスクBEVモデル（27.5M / 2,787 GFLOPs / TensorRT実装可能）")
y1 = Inches(2.0)
box(s, Inches(0.3), y1, Inches(1.2), Inches(0.8), "8カメラ画像", "3ch 432x768", fill=RGBColor(0xEE, 0xEE, 0xEE), fs=10)
box(s, Inches(6.0), Inches(3.15), Inches(1.6), Inches(0.5), "カメラ校正 K/T", "内部/外部パラメータ", fill=RGBColor(0xEE, 0xEE, 0xEE), fs=9)
arrow(s, Inches(6.8), Inches(3.15), Inches(7.2), y1 + Inches(0.8))
box(s, Inches(1.8), y1, Inches(1.7), Inches(0.8), "ResNet34+FPN", "21.7M / 475G\n→ 160ch 108x192 (stride4)", fill=AI, fs=10)
# image-side heads
box(s, Inches(3.9), Inches(1.0), Inches(1.95), Inches(0.55), "2D Segヘッド", "21クラス(CSV) 0.23M / 78G", fill=AI, fs=9.5)
box(s, Inches(3.9), Inches(1.66), Inches(1.95), Inches(0.55), "2D BBoxヘッド", "10クラス CenterNet 0.33M / 111G", fill=AI, fs=9.5)
box(s, Inches(3.9), Inches(2.32), Inches(1.95), Inches(0.55), "深度デコーダ", "64bin@1.25m 3.29M / 1093G", fill=AI, fs=9.5)
box(s, Inches(3.9), Inches(2.98), Inches(1.95), Inches(0.55), "Context 96ch", "0.02M / 5G", fill=AI, fs=9.5)
box(s, Inches(6.35), y1, Inches(2.1), Inches(0.8), "深度ゲートIPM", "K/T + grid_sample + gather\nパラメータ無 / TRT-safe", fill=OP, fs=9.5)
box(s, Inches(8.85), y1, Inches(1.25), Inches(0.8), "BEV特徴", "96ch 800x500\n@0.2m", fill=OP, fs=9.5)
# BEV-side heads
box(s, Inches(10.6), Inches(1.0), Inches(2.3), Inches(0.55), "BEVレーンSegデコーダ", "9クラス 1.16M / 933G", fill=AI, fs=9.5)
box(s, Inches(10.6), Inches(1.66), Inches(2.3), Inches(0.55), "3D BBoxヘッド", "CenterPoint式 0.41M / 82G", fill=AI, fs=9.5)
box(s, Inches(10.6), Inches(2.32), Inches(2.3), Inches(0.75), "E2Eヘッド ★新規", "軌跡/Steer/Accel/Brake\n+速度v0入力 0.37M / 11G", fill=E2E, fs=9.5)
# outputs row
oy = Inches(3.85)
box(s, Inches(0.9), oy, Inches(1.85), Inches(0.55), "2D Seg [8,21,108,192]", "Cityscapes系色", fill=OUTC, fs=8.5)
box(s, Inches(2.95), oy, Inches(1.85), Inches(0.55), "2D BBox [8cam]", "10クラス+スコア", fill=OUTC, fs=8.5)
box(s, Inches(5.0), oy, Inches(1.85), Inches(0.55), "Depth [8,108,192]", "MAE ~1.6m", fill=OUTC, fs=8.5)
box(s, Inches(7.05), oy, Inches(1.85), Inches(0.55), "BEVレーン [9,800,500]", "mIoU 0.286", fill=OUTC, fs=8.5)
box(s, Inches(9.1), oy, Inches(1.85), Inches(0.55), "3D BBox(向き付き)", "車両/VRU", fill=OUTC, fs=8.5)
box(s, Inches(11.15), oy, Inches(1.85), Inches(0.55), "軌跡6点+操作量", "3秒先まで", fill=E2E, fs=8.5)
# arrows
arrow(s, Inches(1.5), y1 + Inches(0.4), Inches(1.8), y1 + Inches(0.4))
for yy in (1.27, 1.93, 2.59, 3.25):
    arrow(s, Inches(3.5), y1 + Inches(0.4), Inches(3.9), Inches(yy))
arrow(s, Inches(5.85), Inches(2.59), Inches(6.35), y1 + Inches(0.35))
arrow(s, Inches(5.85), Inches(3.25), Inches(6.35), y1 + Inches(0.55))
arrow(s, Inches(8.45), y1 + Inches(0.4), Inches(8.85), y1 + Inches(0.4))
for yy in (1.27, 1.93, 2.65):
    arrow(s, Inches(10.1), y1 + Inches(0.4), Inches(10.6), Inches(yy))
# legend
box(s, Inches(0.3), Inches(4.7), Inches(1.2), Inches(0.38), "AI計算", "学習有り", fill=AI, fs=8.5)
box(s, Inches(1.65), Inches(4.7), Inches(1.4), Inches(0.38), "幾何/固定演算", "学習無し", fill=OP, fs=8.5)
box(s, Inches(3.2), Inches(4.7), Inches(1.2), Inches(0.38), "タスク出力", "", fill=OUTC, fs=8.5)
box(s, Inches(4.55), Inches(4.7), Inches(1.2), Inches(0.38), "E2E(新規)", "", fill=E2E, fs=8.5)
bullets(s, [
    "1つの共有Backbone+1つのBEV特徴から6タスクを同時推論（追加ヘッドは軽量: 2D BBox 0.33M / E2E 0.37M）",
    "E2Eヘッドは現在速度v0を条件入力とし、BEV特徴のプーリングから 3秒先軌跡(0.5s刻み6点)・操舵角・加減速・ブレーキ を回帰",
    "全演算 conv / grid_sample / gather / maxpool / avgpool / MLP → TensorRT実装可能",
], Inches(0.3), Inches(5.3), Inches(12.7), Inches(1.8), size=12.5)

# ================= slide 2: image branch detail =================
s = add_slide("詳細① 画像ブランチ — 共有特徴と画像系4ヘッド（テンソル形状）")
y = Inches(1.35)
box(s, Inches(0.35), y, Inches(1.5), Inches(1.0), "入力（画像のみ）", "imgs [B,8,3,432,768]\n※K/TはBackbone非入力\n（IPMのみで使用→詳細②）", fill=RGBColor(0xEE, 0xEE, 0xEE), fs=8.5)
box(s, Inches(2.25), y, Inches(1.9), Inches(1.4), "ResNet34", "stem 64ch 108x192 (s4)\nlayer1 64ch 108x192 (s4)\nlayer2 128ch 54x96 (s8)\nlayer3 256ch 27x48 (s16)\nlayer4 512ch 14x24 (s32)\n21.28M", fill=AI, fs=8)
box(s, Inches(4.55), y, Inches(1.9), Inches(1.4), "FPN (lat1-4 + fuse)", "各層1x1で160chへ →\n全て108x192にupsample+加算\n→ fuse 3x3\n出力 f: 160ch 108x192 (s4)\n0.38M", fill=AI, fs=8)
box(s, Inches(7.0), Inches(1.1), Inches(2.8), Inches(0.62), "seg_head: Conv1x1 160→21", "2D Seg [B,8,21,108,192] / 0.23M / 78G", fill=AI, fs=9)
box(s, Inches(7.0), Inches(1.85), Inches(2.8), Inches(0.75), "det2d_stem: ConvBlock 160→128\nhm2d 1x1→10 / reg2d 1x1→4", "2D BBox hm[B,8,10,108,192] reg[.,4,.]\nfocal init b=-2.19 / 0.33M / 111G", fill=AI, fs=8.5)
box(s, Inches(7.0), Inches(2.75), Inches(2.8), Inches(0.75), "depth_head: ConvBlock 160→256→256→192→128 + 1x1→64", "dlog [B,8,64,108,192] @s4 (64bin x 1.25m)\n3.29M / 1093G", fill=AI, fs=9)
box(s, Inches(7.0), Inches(3.65), Inches(2.8), Inches(0.62), "ctx: Conv 160→96", "Context [B*8,96,108,192] / 0.02M / 5G", fill=AI, fs=9)
for yy in (1.41, 2.22, 3.12, 3.96):
    arrow(s, Inches(6.45), y + Inches(0.5), Inches(7.0), Inches(yy))
arrow(s, Inches(1.85), y + Inches(0.5), Inches(2.25), y + Inches(0.5))
arrow(s, Inches(4.15), y + Inches(0.5), Inches(4.55), y + Inches(0.5))
bullets(s, [
    "2D Seg GT: comlops-21cls-autolabel-2504.csv 準拠21クラス。細線クラス(lane/marking/pole/標識/信号)は被覆率>12%でセル判定する縮小でGT化（最近傍縮小だと消失）",
    (1, "ego_vehicle は背景0として監督（ignoreにするとボンネットが無監督ノイズになる）。クラス重み: lane/marking x4, pole/標識/信号 x2, VRU x1.5, 背景 x0.4"),
    "2D BBox GT: fastlabel_2510_instance.csv 準拠10クラス。object_annのbboxを768x432系に変換、カメラ毎に面積上位32個。CenterNet式（Gaussian focal + 中心L1）",
    "深度 GT: LiDAR + road/ペイント統合セグメント線形補間 + 幾何地面補完（下向き50度制限）。loss = bin-CE(ラベル平滑0.05) + 0.1 x 期待値L1",
    "デコード（TRT-safe）: 2D BBox = hm sigmoid → 3x3 maxpool NMS → topK → 中心オフセット+サイズexp復元",
], Inches(0.35), Inches(4.6), Inches(12.6), Inches(2.6), size=11.5)

# ================= slide 3: BEV branch detail =================
s = add_slide("詳細② BEVブランチ — 深度ゲートIPMとBEV系3ヘッド")
y = Inches(1.3)
box(s, Inches(0.35), y, Inches(2.5), Inches(1.9),
    "深度ゲートIPM（学習パラメータ無 / 入力: Context 96ch + 深度確率 + K/T）",
    "1) BEV格子点(z=0, 800x500@0.2m)を各カメラへ投影 (K,T)\n"
    "2) Context/深度確率を grid_sample\n"
    "3) 点距離の深度binを gather → 重み w=P(depth=d)+0.05\n"
    "4) 8カメラを重み付き平均 → BEV特徴 [B,96,800,500]", fill=OP, fs=8.5)
box(s, Inches(3.35), Inches(1.0), Inches(3.1), Inches(0.8), "dec: 96→160→160→128→9 (全て800x500)", "出力 [B,9,800,500]\nbg/road/歩道/横断歩道/lane/停止線/edge/駐車\n1.16M / 933G", fill=AI, fs=8.5)
box(s, Inches(3.35), Inches(2.0), Inches(3.1), Inches(0.9), "det_stem: 96ch 800x500 → 128ch 400x250 (s2)\nhm 1x1→2 / reg 1x1→6 @400x250", "3D BBox @stride2 400x250\nreg=(off_r,off_c,log l,log w,sin,cos yaw)\n0.41M / 82G", fill=AI, fs=8.5)
box(s, Inches(3.35), Inches(3.1), Inches(3.1), Inches(1.0), "ego_stem: 96ch 800x500 → 64ch 200x125 (s4)\n→ 96ch 200x125 → 96ch 100x63 (s2)\n→ AvgPool → 96次元 / ego_mlp: [96+v0]→256→256→15", "E2E: wp12 + steer + accel + brake\n0.37M / 11G", fill=E2E, fs=8.5)
arrow(s, Inches(2.85), y + Inches(0.6), Inches(3.35), Inches(1.4))
arrow(s, Inches(2.85), y + Inches(0.95), Inches(3.35), Inches(2.45))
arrow(s, Inches(2.85), y + Inches(1.3), Inches(3.35), Inches(3.6))
box(s, Inches(6.95), Inches(1.0), Inches(2.0), Inches(0.62), "BEVレーン", "9クラス 160x100m", fill=OUTC, fs=9)
box(s, Inches(6.95), Inches(2.1), Inches(2.0), Inches(0.62), "向き付き3D BBox", "(cls,x,y,l,w,yaw)", fill=OUTC, fs=9)
box(s, Inches(6.95), Inches(3.25), Inches(2.0), Inches(0.72), "軌跡+操作量", "wp6点(3s) steer accel brake", fill=E2E, fs=9)
arrow(s, Inches(6.45), Inches(1.4), Inches(6.95), Inches(1.31))
arrow(s, Inches(6.45), Inches(2.45), Inches(6.95), Inches(2.41))
arrow(s, Inches(6.45), Inches(3.6), Inches(6.95), Inches(3.61))
box(s, Inches(9.4), Inches(1.0), Inches(3.5), Inches(3.0),
    "E2E GT（CAN無し → ego_pose導出）",
    "軌跡: 将来3秒を0.5秒刻みで現在ego座標系へ変換\n"
    "速度v0: 位置差分（±0.5s平滑）\n"
    "加速度: 速度微分（平滑）\n"
    "操舵角: bicycleモデル atan(2.8m x ヨーレート / v)\n"
    "  ※v<0.5m/sはマスク\n"
    "ブレーキ: 減速度 < -0.5 m/s2 の2値\n"
    "シーン末端（3秒先なし）はvalid=0で除外\n"
    "損失: L1(wp) + 2xL1(steer) + L1(accel)\n"
    "      + 0.5xBCE(brake)、validのみ", fill=RGBColor(0xF5, 0xF0, 0xFA), fs=9)
bullets(s, [
    "深度が IPM のゲート（可視性の弁）として働く: 距離の合う位置だけContextがBEVへ流入 → 遮蔽・浮遊物のにじみを抑制",
    "E2Eヘッドは知覚と同一のBEV特徴を消費 — 知覚タスクの表現学習がそのままプランニングの入力品質になる",
], Inches(0.35), Inches(4.35), Inches(12.6), Inches(1.2), size=11.5)

# ================= slide 4: losses / training =================
s = add_slide("学習構成 — 6タスク joint / GT品質フィルタ / 運用制約")
bullets(s, [
    "総損失 = BEVレーン(CE·bg0.5 + Lovász0.5 + 境界重みx3 + Tversky0.6 + Dice0.5 + 遠方重みx1)",
    (1, "+ 0.6 x Depth(bin-CE ラベル平滑 + 期待値L1) + 0.8 x 3D BBox(Gaussian focal + 中心L1)"),
    (1, "+ 0.4 x 2D Seg(クラス重み付きCE, ignore=255, 全ignoreバッチはゼロ返却でnan回避)"),
    (1, "+ 0.3 x 2D BBox(focal + L1) + 0.5 x E2E(L1軌跡 + 2xL1操舵 + L1加速度 + 0.5xBCEブレーキ)"),
    "教師データ: aisin 1,120 + DTSET 変換分をローリング追加（r7時点 625 / 全5,282シーン変換中）",
    "GT品質フィルタ: シーン端トリム(先頭3+末尾10) / 停車地点除外(gtcov: 前方<0.5%) / 交差点ガード(対向車線除去の30%復元)",
    "val計測: BEV mIoU / [val2d] 2D Seg IoU(lane/bg等) / [valE2E] 軌跡ADE·FDE + steer/accel MAE + brake精度",
    "運用制約: GPU1恒久ECC故障(常時除外) / depth入りDDPはworkers=0 / depth-w 0の長期学習は深度崩壊 / torchrunは絶対パス",
    "ラウンド系譜: r2(4task 12cls)=0.283 → r3(21cls)=0.285 → r4(+DTSET)=0.286 → r5(v17 +2D BBox)=0.286 → r6(GT修正版, val2d初計測 lane0.371) → r7(v18 +E2E) 学習中",
], Inches(0.5), Inches(1.05), Inches(12.4), Inches(5.9), size=12.5)

# ================= slide 5: specs =================
s = add_slide("諸元 — パラメータ / FLOPs 内訳（実測: 27.49M / 2,787 GFLOPs @8cam 768x432）")
HDR = RGBColor(0xC9, 0xD9, 0xEA)
rows = [("ステージ", "Params", "GFLOPs", "比率", "備考"),
        ("ResNet34 Backbone + FPN", "21.66M", "474.7", "17.0%", "共有・8カメラ分 stride4特徴"),
        ("2D Segヘッド", "0.23M", "77.8", "2.8%", "21クラス @stride4"),
        ("深度デコーダ", "3.29M", "1093.4", "39.2%", "64bin 大型デコーダ(v14d) — 最大コスト"),
        ("Context", "0.02M", "5.1", "0.2%", "IPMへ運ぶ96ch特徴"),
        ("2D BBoxヘッド", "0.33M", "111.0", "4.0%", "10クラス CenterNet @stride4"),
        ("深度ゲートIPM", "0", "~1", "~0%", "grid_sample+gather（パラメータ無）"),
        ("BEVレーンSegデコーダ", "1.16M", "932.8", "33.5%", "9クラス 800x500 — 第2のコスト"),
        ("3D BBoxヘッド", "0.41M", "81.6", "2.9%", "CenterPoint式 @BEV stride2"),
        ("E2Eヘッド", "0.37M", "10.8", "0.4%", "ego_stem + MLP(96+v0→15)"),
        ("合計", "27.49M", "2787.2", "100%", "= 1393.6 GMACs（thop実測）")]
yy = 1.1
for i, (a, b, c, d, e) in enumerate(rows):
    hdr = i == 0
    last = i == len(rows) - 1
    fill1 = HDR if hdr else (RGBColor(0xDE, 0xE8, 0xF3) if last else RGBColor(0xF2, 0xF2, 0xF2))
    fill2 = HDR if hdr else (RGBColor(0xDE, 0xE8, 0xF3) if last else RGBColor(0xFA, 0xFA, 0xFA))
    box(s, Inches(0.5), Inches(yy), Inches(3.3), Inches(0.42), a, fs=10.5, fill=fill1)
    box(s, Inches(3.8), Inches(yy), Inches(1.25), Inches(0.42), b, fs=10.5, fill=fill2)
    box(s, Inches(5.05), Inches(yy), Inches(1.25), Inches(0.42), c, fs=10.5, fill=fill2)
    box(s, Inches(6.3), Inches(yy), Inches(1.0), Inches(0.42), d, fs=10.5, fill=fill2)
    box(s, Inches(7.3), Inches(yy), Inches(5.4), Inches(0.42), e, fs=9.5, fill=fill2)
    yy += 0.47
bullets(s, [
    "入力: 8カメラ 768x432（WIDE/LEFT/RIGHT/NARROW x 前後） + キャリブレーション(K, T) + 現在速度v0。全6タスクを1回のforwardで同時出力",
    "コストの76%が 深度デコーダ + BEVレーンデコーダ に集中 — 高速化するならこの2段の解像度/ch削減が第一候補",
    "新規ヘッドは軽量: 2D BBox 4.0% / E2E 0.4% — タスク追加の限界コストが小さいアーキテクチャ",
], Inches(0.5), Inches(6.45), Inches(12.4), Inches(1.0), size=11)

prs.save("out/bevlane_v18_arch.pptx")
print("saved out/bevlane_v18_arch.pptx (5 slides)")
