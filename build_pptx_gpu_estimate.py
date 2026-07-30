#!/usr/bin/env python3
"""One-slide GPU sizing summary for METEOR large-scale training.

Native (editable) PPTX chart: training days for 1,000 h x 10 epochs vs
GPU count. Numbers derive from the measured r43-r45 throughput:
2.2 s/step at 8x L40S, 16 samples/step, ~1.6 fps sampling of 30 s scenes.
"""
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

C_NAVY = RGBColor(0x1B, 0x2A, 0x4A)
C_BLUE = RGBColor(0x1E, 0x6E, 0xB8)
C_AMBER = RGBColor(0xC8, 0x7A, 0x14)
C_GRAY = RGBColor(0x55, 0x5C, 0x63)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
s = prs.slides.add_slide(prs.slide_layouts[6])

tb = s.shapes.add_textbox(Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.7))
p = tb.text_frame.paragraphs[0]
p.text = "大規模学習に必要なGPU数 ── 1,000時間データ × 10エポックの学習日数"
p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = C_NAVY

sub = s.shapes.add_textbox(Inches(0.5), Inches(0.95), Inches(12.3), Inches(0.4))
p = sub.text_frame.paragraphs[0]
p.text = ("実測ベース：現行8×L40Sで2.2秒/step・16サンプル/step（r43〜r45学習の実測値から線形換算）")
p.font.size = Pt(14); p.font.color.rgb = C_GRAY

# ---- native editable chart ----
cd = CategoryChartData()
cd.categories = ["8 GPU\n(現行1ノード)", "32 GPU\n(4ノード)",
                 "64 GPU\n(8ノード)", "64 GPU\n(H100換算)"]
cd.add_series("学習日数", (92, 23, 12, 5.5))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,
                        Inches(0.5), Inches(1.45), Inches(7.6), Inches(5.4),
                        cd)
ch = gf.chart
ch.has_legend = False
plot = ch.plots[0]
plot.has_data_labels = True
plot.data_labels.number_format = '0.#"日"'
plot.data_labels.number_format_is_linked = False
plot.data_labels.font.size = Pt(16)
plot.data_labels.font.bold = True
ser = plot.series[0]
ser.format.fill.solid()
ser.format.fill.fore_color.rgb = C_BLUE
ch.value_axis.has_major_gridlines = True
ch.value_axis.tick_labels.font.size = Pt(12)
ch.category_axis.tick_labels.font.size = Pt(12)

# ---- right column: key facts ----
rx = Inches(8.4)
box = s.shapes.add_textbox(rx, Inches(1.5), Inches(4.4), Inches(5.5))
tf = box.text_frame; tf.word_wrap = True
items = [
    ("推奨：64 GPU（8GPUノード×8）", C_AMBER, 18, True),
    ("→ 約12日/ラウンド（2週間サイクルの開発が可能）", C_NAVY, 14, False),
    ("", C_NAVY, 8, False),
    ("・32 GPUなら約23日 ＝ 月1リリースなら成立", C_NAVY, 14, False),
    ("・現行の8 GPUでは約92日 ＝ 非現実的", C_NAVY, 14, False),
    ("・H100系なら同じ64枚で5〜6日（実効2〜2.5倍）", C_NAVY, 14, False),
    ("", C_NAVY, 8, False),
    ("メモリ要件：バッチ2/GPUで約40GB使用", C_NAVY, 14, True),
    ("→ 48GB級（L40S / A100 / H100）が必要", C_NAVY, 14, False),
    ("", C_NAVY, 8, False),
    ("自動ラベリングはCPU処理（GPU不要）：", C_GRAY, 13, False),
    ("1,000時間分 ≈ 64コアノード1台で10〜14日（学習と並走可）", C_GRAY, 13, False),
    ("", C_NAVY, 8, False),
    ("参考：現行規模（64時間分×8エポック）は", C_GRAY, 13, False),
    ("8 GPUで約3日/ラウンド", C_GRAY, 13, False),
]
first = True
for txt, col, size, bold in items:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    p.text = txt
    p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = col

foot = s.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.4))
p = foot.text_frame.paragraphs[0]
p.text = ("感度：フレーム間引き1/2で全日数が半分／7カメラ（J6）構成で約12%減／"
          "DDPスケーリングはほぼ線形（実測に基づく）")
p.font.size = Pt(12); p.font.color.rgb = C_GRAY

out = "out/GPU_estimate.pptx"
prs.save(out)
print("saved", out)
