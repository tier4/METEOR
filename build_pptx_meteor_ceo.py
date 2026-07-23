#!/usr/bin/env python3
"""METEOR CEO introduction deck (Japanese, ~15 min) -> out/METEOR_CEO.pptx.

First-time executive introduction. Selling points: Zero Human Code, Zero Human
Annotation, No Map, depth-based vehicle-robust perception, lightweight edge
deployment, self-guardrail. Also renders two diagrams to docs/media/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    try: _fm.fontManager.addfont(_f)
    except Exception: pass
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# palette
NAVY = "#152238"; BLUE = "#1E6EB8"; TEAL = "#0E9AA0"; AMBER = "#E08A1E"
GREEN = "#1B7837"; RED = "#C0392B"; GRAYc = "#606870"; LT = "#F2F4F6"
DK = "#20"

# ---------------------------------------------------------------- diagram 1: engine
fig, ax = plt.subplots(figsize=(15.5, 5.4), dpi=120)
ax.set_xlim(0, 160); ax.set_ylim(0, 54); ax.axis("off")


def dbox(x, y, w, h, t, s="", fc="#CFE2F3", fs=12, sfs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55",
                                fc=fc, ec="#4a5560", lw=1.3))
    ax.text(x + w / 2, y + h * 0.66, t, ha="center", va="center",
            fontsize=fs, fontweight="bold", color="#152238")
    if s:
        ax.text(x + w / 2, y + h * 0.3, s, ha="center", va="center",
                fontsize=sfs, color="#444")


def darr(x1, y1, x2, y2, col="#152238"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=2.2, color=col))


dbox(1, 20, 26, 16, "走行データ", "8カメラ + LiDAR\n(人手ラベル無し)", fc="#EEEEEE")
dbox(33, 20, 28, 16, "自動ラベル生成", "LiDAR蓄積 + 幾何整合\nコンセンサスGT", fc="#FDE7B5")
dbox(67, 20, 28, 16, "自己学習", "マルチタスクBEVモデル\n(全コードAIが記述)", fc="#CFE2F3")
dbox(101, 20, 26, 16, "自己改善", "Refinerが\n出力を自動補正", fc="#E8D5F2")
dbox(133, 20, 25, 16, "エッジ配備", "TensorRT / C++\nカメラのみで動作", fc="#D9EAD3")
darr(27, 28, 33, 28); darr(61, 28, 67, 28); darr(95, 28, 101, 28)
darr(127, 28, 133, 28)
ax.text(80, 46, "METEOR の自律エンジン — 人が触れるのはデータ収集だけ",
        ha="center", fontsize=17, fontweight="bold", color="#152238")
ax.text(80, 8, "収集 → ラベル → 学習 → 改善 → 配備 の全工程を自動化"
        "（人手のラベリングもコード記述も不要）",
        ha="center", fontsize=11, style="italic", color="#B9770E")
fig.savefig("docs/media/ceo_engine.png", bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- diagram 2: vs conventional
fig, ax = plt.subplots(figsize=(14, 6.2), dpi=120)
ax.set_xlim(0, 140); ax.set_ylim(0, 62); ax.axis("off")
rows = [
    ("アノテーション", "数万〜数十万枚を人手\n（高コスト・数ヶ月）", "人手ゼロ・自動生成\n（数時間で拡張）"),
    ("HDマップ", "事前作成・維持が必要", "地図不要（どこでも走行）"),
    ("開発コード", "大規模な人手開発", "全コードをAIが記述"),
    ("センサー", "高価なLiDAR依存も多い", "カメラのみで推論\n（深度をAI推定）"),
    ("安全担保", "モデル任せになりがち", "決定論的な自己ガードレール"),
    ("配備", "重く高価な計算機", "軽量・エッジ実装"),
]
ax.text(23, 58, "観点", ha="center", fontsize=13, fontweight="bold", color="#152238")
ax.text(63, 58, "従来の自動運転認識", ha="center", fontsize=13, fontweight="bold", color="#C0392B")
ax.text(110, 58, "METEOR", ha="center", fontsize=13, fontweight="bold", color="#1B7837")
for i, (k, a, b) in enumerate(rows):
    y = 50 - i * 8.4
    ax.add_patch(FancyBboxPatch((2, y - 3.4), 42, 7.2, boxstyle="round,pad=0.2",
                                fc="#EEEEEE", ec="none"))
    ax.text(23, y, k, ha="center", va="center", fontsize=11, fontweight="bold",
            color="#152238")
    ax.text(63, y, a, ha="center", va="center", fontsize=9.5, color="#7a3b33")
    ax.text(110, y, b, ha="center", va="center", fontsize=9.8, color="#1B5c2b",
            fontweight="bold")
fig.savefig("docs/media/ceo_vs.png", bbox_inches="tight", facecolor="white")

# ---------------------------------------------------------------- diagram 3: model architecture
fig, ax = plt.subplots(figsize=(15.5, 6.4), dpi=120)
ax.set_xlim(0, 168); ax.set_ylim(0, 64); ax.axis("off")


def abox(x, y, w, h, t, s="", fc="#CFE2F3", fs=11.5, sfs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                fc=fc, ec="#4a5560", lw=1.2))
    ax.text(x + w / 2, y + (h * 0.62 if s else h / 2), t, ha="center",
            va="center", fontsize=fs, fontweight="bold", color="#152238")
    if s:
        ax.text(x + w / 2, y + h * 0.26, s, ha="center", va="center",
                fontsize=sfs, color="#444")


def aarr(x1, y1, x2, y2, col="#152238", w=2.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=15, lw=w, color=col))


ax.text(84, 61, "モデルアーキテクチャ（カメラのみ・地図なし）", ha="center",
        fontsize=16, fontweight="bold", color="#152238")
abox(1, 30, 20, 16, "8台の\nサラウンド\nカメラ", fc="#EEEEEE")
abox(26, 30, 22, 16, "共有バックボーン", "画像特徴抽出", fc="#CFE2F3")
abox(53, 44, 24, 12, "深度推定", "ピクセル毎(AI)", fc="#FDE7B5")
abox(53, 26, 24, 12, "深度ゲート付き\nIPM投影", "幾何的に正確", fc="#FDE7B5")
abox(82, 30, 22, 16, "共有BEV\n(俯瞰空間)", "+ 時系列メモリ", fc="#D9EAD3")
# task heads fanned out
heads = ["BEVセグ", "3D物体", "経路計画E2E", "深度・占有", "信号・車線", "未知障害物"]
for i, hd in enumerate(heads):
    y = 51 - i * 7.0
    abox(110, y - 2.6, 24, 5.6, hd, fc="#CFE2F3", fs=10)
    aarr(104, 38, 110, y)
abox(140, 26, 26, 24, "Refiner\n(自己改善)", "出力を自動補正\n+ 自己ガードレール\n(決定論的な安全層)",
     fc="#E8D5F2", fs=12)
aarr(21, 38, 26, 38); aarr(48, 38, 53, 50); aarr(48, 38, 53, 32)
aarr(77, 50, 82, 40); aarr(77, 32, 82, 36); aarr(104, 38, 110, 38)
aarr(134, 38, 140, 38)
ax.text(84, 3, "入力：CoMET基盤で自動ラベルしたCo-MLOpsの走行データ  →  "
        "1つのモデルが多タスクを同時出力  →  Refinerが補正・ガードレールが安全担保",
        ha="center", fontsize=10.5, style="italic", color="#B9770E")
fig.savefig("docs/media/ceo_arch.png", bbox_inches="tight", facecolor="white")
print("saved 3 diagrams")

# ---------------------------------------------------------------- deck
SW, SH = Inches(13.333), Inches(7.5)
C_NAVY = RGBColor(0x15, 0x22, 0x38); C_BLUE = RGBColor(0x1E, 0x6E, 0xB8)
C_GRAY = RGBColor(0x60, 0x68, 0x70); C_GREEN = RGBColor(0x1B, 0x78, 0x37)
C_RED = RGBColor(0xC0, 0x39, 0x2B); C_AMBER = RGBColor(0xE0, 0x8A, 0x1E)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF); C_TEAL = RGBColor(0x0E, 0x9A, 0xA0)
prs = Presentation(); prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def _fill(shape, rgb):
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb
    shape.line.fill.background()


def slide(title=None, sub=None, band=C_BLUE):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.55), Inches(0.25), Inches(12.2),
                                  Inches(0.8))
        p = tb.text_frame.paragraphs[0]
        p.text = title; p.font.size = Pt(28); p.font.bold = True
        p.font.color.rgb = C_NAVY
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.02),
                                Inches(12.2), Pt(3))
        _fill(ln, band)
        if sub:
            tb2 = s.shapes.add_textbox(Inches(0.55), Inches(1.06), Inches(12.2),
                                       Inches(0.45))
            p2 = tb2.text_frame.paragraphs[0]
            p2.text = sub; p2.font.size = Pt(14); p2.font.italic = True
            p2.font.color.rgb = C_GRAY
    return s


def bullets(s, items, x=0.7, y=1.6, w=11.9, h=5.5, size=17):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.auto_size = MSO_AUTO_SIZE.NONE
    for i, it in enumerate(items):
        lvl, txt, col = 0, it, None
        if isinstance(it, tuple):
            lvl, txt = it[0], it[1]; col = it[2] if len(it) > 2 else None
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("■ " if lvl == 0 else "     ・") + txt
        p.font.size = Pt(size if lvl == 0 else size - 3)
        p.font.color.rgb = col or C_NAVY; p.font.bold = (lvl == 0 and col is not None)
        p.space_after = Pt(7)


def demo_slide(title, sub=None, band=C_BLUE):
    """Title-only placeholder page for a demo video (frame + '▶' hint)."""
    s = slide(title, sub, band=band)
    frame = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.3),
                               Inches(1.9), Inches(8.7), Inches(4.9))
    frame.fill.solid(); frame.fill.fore_color.rgb = RGBColor(0x0E, 0x14, 0x22)
    frame.line.color.rgb = C_BLUE; frame.line.width = Pt(2)
    tf = frame.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.text = "▶"; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(60); p.font.color.rgb = C_WHITE
    p2 = tf.add_paragraph(); p2.text = "（デモ動画をここに挿入）"
    p2.alignment = PP_ALIGN.CENTER; p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(0xB9, 0xCF, 0xE8)
    return s


def bignum(s, items, y=2.2):
    n = len(items)
    w = 12.0 / n
    for i, (num, lab) in enumerate(items):
        x = 0.7 + i * w
        card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                  Inches(y), Inches(w - 0.35), Inches(2.4))
        _fill(card, RGBColor(0xF2, 0xF4, 0xF6))
        t = card.text_frame; t.word_wrap = True
        t.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = t.paragraphs[0]; p.text = num; p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(34); p.font.bold = True; p.font.color.rgb = C_BLUE
        p2 = t.add_paragraph(); p2.text = lab; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(13); p2.font.color.rgb = C_NAVY


# 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH); _fill(bg, C_NAVY)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.3), Inches(2))
p = tb.text_frame.paragraphs[0]
p.text = "METEOR"; p.font.size = Pt(66); p.font.bold = True
p.font.color.rgb = C_WHITE; p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1.0), Inches(3.9), Inches(11.3), Inches(1.4))
for i, (t, sz, c) in enumerate([
        ("人手ゼロで作る、自動運転の「眼」と「脳」", 26, C_WHITE),
        ("ラベル無し・地図無し・人手コード無し ── カメラだけで走る", 16, RGBColor(0xB9, 0xCF, 0xE8))]):
    p = tb2.text_frame.paragraphs[0] if i == 0 else tb2.text_frame.add_paragraph()
    p.text = t; p.font.size = Pt(sz); p.font.color.rgb = c
    p.alignment = PP_ALIGN.CENTER; p.font.bold = (i == 0)

# 2 executive summary
s = slide("エグゼクティブ・サマリー", "3行で言うと")
bullets(s, [
    (0, "METEORは、人手のアノテーションもコード記述も無しに、走行データだけから"
        "自動運転の認識・計画AIを自律的に作り続けるエンジンです。", C_NAVY),
    (0, "カメラのみ（深度をAIが推定）で、地図に依存せず、軽量にエッジ配備。"
        "決定論的な自己ガードレールで安全に動きます。", C_NAVY),
    (0, "現時点で人手ラベル0枚のまま約60時間分・7,147シーンを自動学習し、"
        "車両検出・車線・信号・経路計画までを1モデルで出力しています。", C_GREEN),
], size=18)
bignum(s, [("0", "人手ラベル"), ("0", "人手コード行"), ("0", "HDマップ"),
           ("60h+", "自動学習データ")], y=4.6)

# 3 problem
s = slide("なぜ自動運転の「認識」は高コストなのか", "従来アプローチの3つの重荷", band=C_RED)
bullets(s, [
    (0, "アノテーション地獄", C_RED),
    (1, "数万〜数十万枚の画像を人手でラベル付け。1都市分でも数ヶ月・高コスト。"),
    (0, "HDマップ依存", C_RED),
    (1, "事前に精密地図を作成・維持。地図の無い場所は走れない。"),
    (0, "重い開発と重いハード", C_RED),
    (1, "大規模な人手開発、LiDAR等の高価センサー、データセンター級の計算機。"),
    (0, "→ 結果、スケール（新地域・新車種への展開）が遅く、コストが青天井。", C_NAVY),
])

# 4 what is meteor + engine diagram
s = slide("METEORとは ── 自律的に成長する認識エンジン")
bullets(s, [
    (0, "走行データを入れるだけで、ラベル生成→学習→自己改善→配備までを自動で回す"
        "「認識・計画の自己生成エンジン」。", C_NAVY),
    (0, "出力は1つのモデルで多タスク：BEVセグメンテーション・3D物体・信号・"
        "経路計画(E2E)・深度・占有・車線グラフ・未知障害物。", C_NAVY),
], y=1.5, size=16)
s.shapes.add_picture("docs/media/ceo_engine.png", Inches(0.6), Inches(3.4),
                     width=Inches(12.1))

# 4.5 model architecture (early)
s = slide("モデルアーキテクチャ", "8台のカメラ映像を1つのモデルが俯瞰(BEV)へ変換し、多タスクを同時出力")
s.shapes.add_picture("docs/media/ceo_arch.png", Inches(0.55), Inches(1.4),
                     width=Inches(12.2))
bullets(s, [
    (0, "データはCoMET（自動ラベル基盤）で教師付けしたCo-MLOpsの走行データを使用。",
        C_NAVY),
], y=6.55, size=13)

# 5 differentiators overview
s = slide("METEORの4+2の強み（＝競争優位）", "この組み合わせが他に無い")
bullets(s, [
    (0, "① Zero Human Annotation ── 人手ラベルゼロ（自動ラベル生成）", C_GREEN),
    (0, "② Zero Human Code ── 全コードをAI(Claude Fable 5)が記述", C_GREEN),
    (0, "③ No Map ── HDマップ不要（どこでも走行）", C_GREEN),
    (0, "④ Depthベースで車両ロバスト ── カメラのみ・深度推定で堅牢", C_GREEN),
    (0, "⑤ 軽量・エッジ実装 ── 車載計算機で動く", C_TEAL),
    (0, "⑥ 自己ガードレール ── 決定論的な安全層で配備", C_TEAL),
], size=18)

# 6 vs conventional
s = slide("従来手法との違い（一目で）")
s.shapes.add_picture("docs/media/ceo_vs.png", Inches(1.0), Inches(1.35),
                     width=Inches(11.3))

# 7 zero annotation
s = slide("① Zero Human Annotation ── 人手ラベルゼロ", "最大のコスト要因を消す", band=C_GREEN)
bullets(s, [
    (0, "データ基盤：Co-MLOpsの走行データを、自動ラベル基盤CoMETで教師付け。", C_NAVY),
    (0, "LiDARの点群を走行全体で蓄積し、幾何的に整合させて教師データを自動生成。"),
    (1, "2通りの生成の「一致」だけを採用するコンセンサス方式でラベルノイズを除去。"),
    (0, "人手ラベル0枚のまま、現在7,147シーン・約104万フレームを学習に使用。"),
    (1, "新しい地域・車種のデータが来ても、人手を介さず数時間で学習データに追加。"),
    (0, "→ アノテーション費用が実質ゼロ。スケールが人件費に縛られない。", C_GREEN),
], size=16)

# 8 zero code
s = slide("② Zero Human Code ── 全コードをAIが記述", "開発速度そのものが競争力", band=C_GREEN)
bullets(s, [
    (0, "モデル設計・学習・評価・自動ラベル・エッジ配備まで、全コードをAI"
        "(Claude Fable 5)が記述。"),
    (0, "24時間、人手を待たずに実験・改善を回し続ける。ラウンドを重ねるごとに自動で"
        "精度が積み上がる。"),
    (1, "クラッシュの自動復旧、新データの自動取り込み、失敗事例の自動採掘まで自律運用。"),
    (0, "→ 人的リソースに律速されない、圧倒的な開発スループット。", C_GREEN),
])

# 9 no map
s = slide("③ No Map ── HDマップ不要", "地図の無い場所でも、その場で認識", band=C_GREEN)
bullets(s, [
    (0, "事前地図に依存せず、その場のカメラ映像だけで走行空間(BEV)を再構成。"),
    (0, "車線・停止線・横断歩道・走行可能領域・信号までをリアルタイムに推定。"),
    (0, "→ 地図の作成・維持コストが不要。未整備地域や変化する道路にも即対応。", C_GREEN),
])

# 10 depth robust
s = slide("④ Depthベースで車両にロバスト", "高価なLiDARに頼らずカメラだけで", band=C_GREEN)
bullets(s, [
    (0, "8台のカメラ映像からAIがピクセル毎の深度を推定し、正確なBEV(俯瞰)を生成。"),
    (0, "深度に基づく投影で、遠近・オクルージョン・多様な車両形状に堅牢。"),
    (1, "LiDARは学習時の教師にのみ使い、推論はカメラのみ（センサー選択の自由度）。"),
    (0, "→ センサーコストを抑えつつ、実環境の車両・歩行者を安定して捉える。", C_GREEN),
])

# 11 edge
s = slide("⑤ 軽量・エッジ実装", "データセンター不要、車載で動く", band=C_TEAL)
bullets(s, [
    (0, "TensorRTエンジン + C++ランタイムに変換済み。車載GPUで単体動作。"),
    (0, "カメラのみで完結（LiDAR等はオプション）。追加センサー無しで配備可能。"),
    (0, "→ 量産車への搭載を見据えた、現実的なコストとフットプリント。", C_TEAL),
])

# 12 guardrail
s = slide("⑥ 自己ガードレール ── 安全に配備する", "AIの判断を決定論で二重化", band=C_TEAL)
bullets(s, [
    (0, "AIの経路計画を、ルールベースの決定論的チェックが常時監視。"),
    (1, "衝突予測・赤信号×停止線・曲率の実行可能性・走行可能領域からの逸脱を検査。"),
    (0, "危険と判定すれば計画を却下し、安全停止(MRM)へ。"),
    (0, "→ 「AIが暴走しない」ことを構造で担保。配備の信頼性を高める。", C_TEAL),
])

# 13 what it outputs (metrics)
s = slide("何ができるか ── 1モデルで多タスク", "カメラのみ・地図なしの実測値（検証データ）")
bullets(s, [
    (0, "BEVセグメンテーション（車線・停止線・横断歩道・走行領域）"),
    (0, "3D物体検出：車両 適合率 83%"),
    (0, "信号認識：全体 74%（青信号 94%）"),
    (0, "経路計画(E2E)：平均誤差 約0.4〜0.9m"),
    (0, "加えて：深度・占有・車線グラフ・未知障害物・他者挙動予測・リスク地図"),
    (1, "後段のRefinerが、遠方や細部を自動で補正しさらに高精度化。"),
], size=16)

# 13.5 METEOR demo video (placeholder)
demo_slide("METEOR デモ ── 実走行での認識・計画",
           "カメラのみ・地図なしで、BEV・3D物体・信号・経路計画をリアルタイム出力")

# 14 business value
s = slide("事業価値", "コスト構造を根本から変える", band=C_AMBER)
bullets(s, [
    (0, "アノテーション費 → 実質ゼロ（人手ラベル不要）。"),
    (0, "地図作成・維持費 → 不要（No Map）。"),
    (0, "開発人件費 → 大幅圧縮（コードをAIが記述、24時間稼働）。"),
    (0, "センサーBOM → 低減（カメラ中心、LiDAR任意）。"),
    (0, "→ 新地域・新車種への展開が「データを入れるだけ」でスケール。"
        "限界費用が劇的に低い。", C_AMBER),
])

# 15 roadmap
s = slide("今後の予定（ロードマップ）", band=C_AMBER)
bullets(s, [
    (0, "データのロバスト化", C_AMBER),
    (1, "NVIDIA Cosmosで生成した多様・希少シーンのデータで学習し、実環境への"
        "堅牢性を高める。"),
    (0, "エッジ最適化・車載SoC実装", C_AMBER),
    (1, "必要な機能に絞り込んだ上で最適化を実施。"),
    (1, "NVIDIA SoC「Orin」および Renesas「R-Car Gen5」への実装を進める。"),
    (0, "リファレンスAIとしてOSS公開", C_AMBER),
    (1, "Reference AI（オープンソース）として公開し、業界標準・エコシステム形成を狙う。"),
    (0, "並行して：遠方・未知障害物の精度向上、車種・地域の横展開、自己学習ループの"
        "完全自律化。", C_NAVY),
], size=16)

# 15.5 Cosmos robustification demo video (placeholder)
demo_slide("Cosmos生成データによるロバスト化 ── デモ",
           "NVIDIA Cosmosで生成した多様・希少シーンでの認識・計画の頑健性", band=C_AMBER)

# 16 closing
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH); _fill(bg, C_NAVY)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.5), Inches(11.3), Inches(2.6))
lines = [("人手ゼロで、自動運転の認識を作り続ける。", 30, C_WHITE, True),
         ("ラベル無し・地図無し・コード無し。カメラだけで、どこでも走る。",
          18, RGBColor(0xB9, 0xCF, 0xE8), False),
         ("", 10, C_WHITE, False),
         ("METEOR ── スケールする自律認識エンジン", 22, C_TEAL, True)]
for i, (t, sz, c, b) in enumerate(lines):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.text = t; p.font.size = Pt(sz); p.font.color.rgb = c
    p.font.bold = b; p.alignment = PP_ALIGN.CENTER

out = "out/METEOR_CEO.pptx"
prs.save(out)
print("saved", out, "-", len(prs.slides._sldIdLst), "slides")
