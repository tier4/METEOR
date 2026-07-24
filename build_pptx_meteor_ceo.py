#!/usr/bin/env python3
"""METEOR CEO introduction deck (Japanese, ~15 min) -> out/METEOR_CEO.pptx.

First-time executive introduction. Selling points: Zero Human Code, Zero Human
Annotation, No Map, depth-based vehicle-robust perception, lightweight edge
deployment, self-guardrail.

All block diagrams are built from NATIVE PowerPoint shapes + connectors (no
image insertion) so they can be edited afterwards.
"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

SW, SH = Inches(13.333), Inches(7.5)
C_NAVY = RGBColor(0x15, 0x22, 0x38); C_BLUE = RGBColor(0x1E, 0x6E, 0xB8)
C_GRAY = RGBColor(0x60, 0x68, 0x70); C_GREEN = RGBColor(0x1B, 0x78, 0x37)
C_RED = RGBColor(0xC0, 0x39, 0x2B); C_AMBER = RGBColor(0xE0, 0x8A, 0x1E)
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF); C_TEAL = RGBColor(0x0E, 0x9A, 0xA0)
C_EDGE = RGBColor(0x4A, 0x55, 0x60)
F_GRAY = RGBColor(0xEE, 0xEE, 0xEE); F_BLUE = RGBColor(0xCF, 0xE2, 0xF3)
F_AMBER = RGBColor(0xFD, 0xE7, 0xB5); F_GREEN = RGBColor(0xD9, 0xEA, 0xD3)
F_PURP = RGBColor(0xE8, 0xD5, 0xF2)
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
        p.font.color.rgb = col or C_NAVY
        p.font.bold = (lvl == 0 and col is not None)
        p.space_after = Pt(7)


# ---- native block-diagram helpers (editable shapes + connectors) ----
def nbox(s, x, y, w, h, title, sub="", fc=F_BLUE, fs=12.5, sfs=9, tc=C_NAVY):
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                           Inches(w), Inches(h))
    b.fill.solid(); b.fill.fore_color.rgb = fc
    b.line.color.rgb = C_EDGE; b.line.width = Pt(1.1)
    b.shadow.inherit = False
    tf = b.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]; p.text = title; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(fs); p.font.bold = True; p.font.color.rgb = tc
    if sub:
        p2 = tf.add_paragraph(); p2.text = sub; p2.alignment = PP_ALIGN.CENTER
        p2.font.size = Pt(sfs); p2.font.color.rgb = RGBColor(0x55, 0x5C, 0x63)
    return b


def _arrow(conn, col):
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"),
                          {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    conn.line.color.rgb = col; conn.line.width = Pt(2.0)


def link(s, a, b, col=C_BLUE, side="h"):
    """Connector glued between shapes a->b. side 'h' = a.right->b.left,
    'v' = a.bottom->b.top. Stays attached when boxes are moved (editable)."""
    if side == "h":
        x1, y1 = a.left + a.width, a.top + a.height // 2
        x2, y2 = b.left, b.top + b.height // 2
        pa, pb = 3, 1
    else:
        x1, y1 = a.left + a.width // 2, a.top + a.height
        x2, y2 = b.left + b.width // 2, b.top
        pa, pb = 2, 0
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    _arrow(c, col)
    try:
        c.begin_connect(a, pa); c.end_connect(b, pb)
    except Exception:
        pass
    return c


def caption(s, txt, y=6.9, col=C_AMBER, size=12):
    tb = s.shapes.add_textbox(Inches(0.6), Inches(y), Inches(12.1), Inches(0.4))
    p = tb.text_frame.paragraphs[0]; p.text = txt; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(size); p.font.italic = True; p.font.color.rgb = col


def table(s, rows, x, y, w, col_w, fs=13, head_fill=RGBColor(0x1E, 0x6E, 0xB8),
          zebra=RGBColor(0xF2, 0xF4, 0xF6)):
    nr, nc = len(rows), len(rows[0])
    t = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                           Inches(0.5 * nr)).table
    for ci, cw in enumerate(col_w):
        t.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = t.cell(ri, ci); c.text = str(val)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            pr = c.text_frame.paragraphs[0]
            pr.font.size = Pt(fs); pr.font.bold = (ri == 0 or ci == 0)
            if ri == 0:
                c.fill.solid(); c.fill.fore_color.rgb = head_fill
                pr.font.color.rgb = C_WHITE
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = zebra if ri % 2 else C_WHITE
                pr.font.color.rgb = (C_RED if ci == 1 else
                                     C_GREEN if ci == 2 else C_NAVY)
    return t


def bignum(s, items, y=2.2):
    n = len(items); w = 12.0 / n
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


def demo_slide(title, sub=None, band=C_BLUE):
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


# ---------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH); _fill(bg, C_NAVY)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.3), Inches(2))
p = tb.text_frame.paragraphs[0]
p.text = "METEOR"; p.font.size = Pt(66); p.font.bold = True
p.font.color.rgb = C_WHITE; p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1.0), Inches(3.9), Inches(11.3), Inches(1.4))
for i, (t, sz, c) in enumerate([
        ("人手ゼロで作る、自動運転の「眼」と「脳」", 26, C_WHITE),
        ("ラベル無し・地図無し・人手コード無し ── カメラだけで走る", 16,
         RGBColor(0xB9, 0xCF, 0xE8))]):
    p = tb2.text_frame.paragraphs[0] if i == 0 else tb2.text_frame.add_paragraph()
    p.text = t; p.font.size = Pt(sz); p.font.color.rgb = c
    p.alignment = PP_ALIGN.CENTER; p.font.bold = (i == 0)

# ---------------------------------------------------------------- 2 exec summary
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

# ---------------------------------------------------------------- 3 problem
s = slide("なぜ自動運転の「認識」は高コストなのか", "従来アプローチの3つの重荷",
          band=C_RED)
bullets(s, [
    (0, "アノテーション地獄", C_RED),
    (1, "数万〜数十万枚の画像を人手でラベル付け。1都市分でも数ヶ月・高コスト。"),
    (0, "HDマップ依存", C_RED),
    (1, "事前に精密地図を作成・維持。地図の無い場所は走れない。"),
    (0, "重い開発と重いハード", C_RED),
    (1, "大規模な人手開発、LiDAR等の高価センサー、データセンター級の計算機。"),
    (0, "→ 結果、スケール（新地域・新車種への展開）が遅く、コストが青天井。", C_NAVY),
])

# ---------------------------------------------------------------- 4 what is meteor
s = slide("METEORとは ── 自律的に成長する認識エンジン")
bullets(s, [
    (0, "METEORは、Co-MLOpsの自動ラベリング基盤CoMETが生み出すデータから"
        "作られた、自動運転の認識・計画AIです。", C_GREEN),
    (0, "そのデータで学習→自己改善→配備までを人手を介さず自動で回す"
        "「認識・計画の自己生成エンジン」。", C_NAVY),
], y=1.45, size=15)
# native pipeline diagram
py = 3.7; ph = 1.7
b1 = nbox(s, 0.55, py, 2.2, ph, "走行データ", "DRS収集・日本全国\n8カメラ+LiDAR", fc=F_GRAY)
b2 = nbox(s, 3.15, py, 2.35, ph, "自動ラベル(CoMET)", "Co-MLOpsの\n自動ラベル済みを再利用", fc=F_AMBER)
b3 = nbox(s, 5.9, py, 2.05, ph, "自己学習", "マルチタスク\nBEVモデル", fc=F_BLUE)
b4 = nbox(s, 8.35, py, 2.0, ph, "自己改善", "Refinerが\n出力を自動補正", fc=F_PURP)
b5 = nbox(s, 10.75, py, 2.0, ph, "エッジ配備", "TensorRT/C++\nカメラのみ", fc=F_GREEN)
for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
    link(s, a, b)
caption(s, "収集 → ラベル → 学習 → 改善 → 配備 の全工程を自動化"
        "（人手のラベリングもコード記述も不要）", y=5.8)

# ---------------------------------------------------------------- 4.5 dev method
s = slide("開発のやり方 ── 人間は「指示」、AIが「開発」",
          "コンセプト・機能は人間が指示。残りの全工程はAIが主導（Zero Human Code）")


def person_icon(sl, cx, cy, h, col):
    """simple editable person pictogram (head + shoulders)"""
    hd = h * 0.36
    head = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - hd / 2), Inches(cy),
                               Inches(hd), Inches(hd))
    _fill(head, col)
    bw, bh = h * 0.62, h * 0.5
    body = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(cx - bw / 2), Inches(cy + hd + 0.04),
                               Inches(bw), Inches(bh))
    _fill(body, col)


def robot_icon(sl, cx, cy, h, col):
    """simple editable robot pictogram (antenna + head + eyes + mouth)"""
    tip = sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - h * 0.07),
                              Inches(cy - h * 0.02), Inches(h * 0.14),
                              Inches(h * 0.14))
    _fill(tip, col)
    ant = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(cx - 0.015),
                              Inches(cy + h * 0.1), Inches(0.03),
                              Inches(h * 0.18))
    _fill(ant, col)
    hw = h * 0.92
    head = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                               Inches(cx - hw / 2), Inches(cy + h * 0.26),
                               Inches(hw), Inches(h * 0.72))
    _fill(head, col)
    for dx in (-hw * 0.22, hw * 0.22):
        eye = sl.shapes.add_shape(MSO_SHAPE.OVAL,
                                  Inches(cx + dx - h * 0.075),
                                  Inches(cy + h * 0.44), Inches(h * 0.15),
                                  Inches(h * 0.15))
        _fill(eye, C_WHITE)
    mo = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(cx - hw * 0.26), Inches(cy + h * 0.76),
                             Inches(hw * 0.52), Inches(h * 0.09))
    _fill(mo, C_WHITE)


# --- human zone (left, amber) ---
hb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.45),
                        Inches(1.8), Inches(2.5), Inches(3.55))
hb.fill.solid(); hb.fill.fore_color.rgb = F_AMBER
hb.line.color.rgb = C_AMBER; hb.line.width = Pt(1.75); hb.shadow.inherit = False
person_icon(s, 1.7, 2.0, 1.0, C_AMBER)
ht = s.shapes.add_textbox(Inches(0.5), Inches(3.6), Inches(2.4), Inches(0.95))
p = ht.text_frame.paragraphs[0]; p.text = "人間"; p.alignment = PP_ALIGN.CENTER
p.font.size = Pt(19); p.font.bold = True; p.font.color.rgb = C_NAVY
p2 = ht.text_frame.add_paragraph()
p2.text = "コンセプト・機能を\n指示するだけ"; p2.alignment = PP_ALIGN.CENTER
p2.font.size = Pt(11.5); p2.font.color.rgb = C_NAVY
qb = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.62),
                        Inches(4.55), Inches(2.16), Inches(0.62))
qb.fill.solid(); qb.fill.fore_color.rgb = C_WHITE
qb.line.color.rgb = C_AMBER; qb.line.width = Pt(1); qb.shadow.inherit = False
p = qb.text_frame.paragraphs[0]; p.text = "「◯◯できる機能が欲しい」"
p.alignment = PP_ALIGN.CENTER; p.font.size = Pt(9.5); p.font.italic = True
p.font.color.rgb = C_NAVY

# --- AI zone (right band, blue) ---
band = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.9),
                          Inches(1.8), Inches(8.9), Inches(3.55))
band.fill.solid(); band.fill.fore_color.rgb = RGBColor(0xF2, 0xF6, 0xFB)
band.line.color.rgb = C_BLUE; band.line.width = Pt(1.75)
band.shadow.inherit = False
robot_icon(s, 7.05, 1.95, 0.62, C_BLUE)
bh_ = s.shapes.add_textbox(Inches(7.45), Inches(2.05), Inches(3.6),
                           Inches(0.45))
p = bh_.text_frame.paragraphs[0]; p.text = "AIが主導（Claude Fable 5）"
p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = C_BLUE
# 5 steps, each with its own editable icon shape above the box
steps = [(MSO_SHAPE.GEAR_6, "モデル開発", "設計・学習・評価"),
         (MSO_SHAPE.CAN, "データセット\n読み込み", "新データ自動取込"),
         (MSO_SHAPE.LEFT_RIGHT_ARROW, "データ変換", "学習形式へ自動変換"),
         (MSO_SHAPE.FUNNEL, "データ\nクレンジング", "ノイズ自動除去"),
         (MSO_SHAPE.CUBE, "モデル管理", "配備・自動復旧")]
sb = []
for i, (icon, t, sub) in enumerate(steps):
    bx = 4.1 + i * 1.75
    if icon == MSO_SHAPE.LEFT_RIGHT_ARROW:      # wide so it reads as an arrow
        ic = s.shapes.add_shape(icon, Inches(bx + 0.44), Inches(2.8),
                                Inches(0.68), Inches(0.32))
    else:
        ic = s.shapes.add_shape(icon, Inches(bx + 0.55), Inches(2.72),
                                Inches(0.45), Inches(0.45))
    _fill(ic, C_BLUE)
    sb.append(nbox(s, bx, 3.3, 1.55, 1.3, t, sub, fc=F_BLUE,
                   fs=11, sfs=8.5))
for a, b in zip(sb, sb[1:]):
    link(s, a, b)
# 24h self-improvement loop inside the AI band
lo = s.shapes.add_shape(MSO_SHAPE.DONUT, Inches(5.45), Inches(4.72),
                        Inches(0.5), Inches(0.5))
_fill(lo, C_TEAL)
lt = s.shapes.add_textbox(Inches(6.1), Inches(4.78), Inches(6.0), Inches(0.4))
p = lt.text_frame.paragraphs[0]
p.text = "24時間 自己改善ループ（クラッシュ復旧・再学習も自動）"
p.font.size = Pt(11.5); p.font.bold = True; p.font.color.rgb = C_TEAL

# --- human <-> AI arrows ---
c1 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, hb.left + hb.width,
                            Inches(2.9), band.left, Inches(2.9))
_arrow(c1, C_AMBER)
c2 = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, band.left, Inches(4.35),
                            hb.left + hb.width, Inches(4.35))
_arrow(c2, C_TEAL)
for lx, ly, txt, col in ((2.87, 2.5, "指示", C_AMBER),
                         (2.87, 4.45, "報告・提案", C_TEAL)):
    lb = s.shapes.add_textbox(Inches(lx), Inches(ly), Inches(1.1), Inches(0.35))
    p = lb.text_frame.paragraphs[0]; p.text = txt; p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = col

# --- workload ratio bar (who does the work) ---
rt = s.shapes.add_textbox(Inches(0.45), Inches(5.62), Inches(2.0), Inches(0.35))
p = rt.text_frame.paragraphs[0]; p.text = "作業量の割合:"
p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = C_NAVY
seg_h = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.95), Inches(5.95),
                           Inches(0.55), Inches(0.45))
_fill(seg_h, C_AMBER)
seg_a = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.5), Inches(5.95),
                           Inches(10.35), Inches(0.45))
_fill(seg_a, C_BLUE)
p = seg_a.text_frame.paragraphs[0]
p.text = "AI ～95%（モデル開発・データ処理・運用のすべて）"
p.alignment = PP_ALIGN.CENTER; p.font.size = Pt(12); p.font.bold = True
p.font.color.rgb = C_WHITE
hl = s.shapes.add_textbox(Inches(1.35), Inches(6.45), Inches(1.9),
                          Inches(0.35))
p = hl.text_frame.paragraphs[0]; p.text = "人間 ～5%（指示）"
p.font.size = Pt(10.5); p.font.bold = True; p.font.color.rgb = C_AMBER
caption(s, "人間は「何を作るか」を決めるだけ ── "
        "モデル開発からデータ処理・運用まで、作り方はすべてAIが自律実行",
        y=6.95, col=C_GREEN, size=13)

# ---------------------------------------------------------------- 5 model architecture
s = slide("モデルアーキテクチャ",
          "8台のカメラ映像を1つのモデルが俯瞰(BEV)へ変換し、多タスクを同時出力")
ay = 2.15; ah = 1.7
a1 = nbox(s, 0.55, ay, 1.9, ah, "8台の\nカメラ", fc=F_GRAY)
a2 = nbox(s, 2.75, ay, 2.15, ah, "共有\nバックボーン", "画像特徴抽出", fc=F_BLUE)
a3 = nbox(s, 5.2, ay - 0.95, 2.2, 1.5, "深度推定", "ピクセル毎(AI)", fc=F_AMBER)
a3b = nbox(s, 5.2, ay + 1.05, 2.2, 1.5, "深度ゲート\nIPM投影", "幾何的に正確", fc=F_AMBER)
a4 = nbox(s, 7.7, ay, 2.1, ah, "共有BEV\n(俯瞰空間)", "+時系列メモリ", fc=F_GREEN)
a5 = nbox(s, 10.05, ay - 0.35, 2.7, 2.4,
          "マルチタスクヘッド",
          "BEVセグ / 3D物体 / 信号\n経路計画E2E / 深度・占有\n車線グラフ / 未知障害物",
          fc=F_BLUE, fs=12, sfs=9.5)
a6 = nbox(s, 5.6, ay + 3.05, 4.6, 1.35, "Refiner + 自己ガードレール",
          "出力を自動補正し、決定論的な安全層で配備", fc=F_PURP, fs=13, sfs=10)
link(s, a1, a2); link(s, a2, a3); link(s, a2, a3b)
link(s, a3, a4); link(s, a3b, a4); link(s, a4, a5)
link(s, a5, a6, side="v", col=C_TEAL)
bullets(s, [
    (0, "現行の車載SoC（NVIDIA Orin）で動作可能な、現実的な構成のCNNベース"
        "効率重視アーキテクチャを採用（LLM・世界モデル(WM)ベースにも拡張可能"
        "だが、現行ハードでは実行が難しいため現時点では未着手）。", C_NAVY),
    (0, "学習データはDRSで収集した日本全国（九州〜北海道）の走行データで"
        "バリエーションが高い。Co-MLOpsプロジェクトでCoMET（自動ラベル基盤）"
        "により自動ラベリング済みのものを再利用。", C_NAVY),
], y=6.3, size=11.5)

# ---------------------------------------------------------------- 6 METEOR demo (early hook)
demo_slide("METEOR デモ ── 実走行での認識・計画",
           "カメラのみ・地図なしで、BEV・3D物体・信号・経路計画をリアルタイム出力")

# ---------------------------------------------------------------- 7 differentiators
s = slide("METEORの4+2の強み（＝競争優位）", "この組み合わせが他に無い")
bullets(s, [
    (0, "① Zero Human Annotation ── 人手ラベルゼロ（自動ラベル生成）", C_GREEN),
    (0, "② Zero Human Code ── 全コードをAI(Claude Fable 5)が記述", C_GREEN),
    (0, "③ No Map ── HDマップ不要（どこでも走行）", C_GREEN),
    (0, "④ Depthベースで車両ロバスト ── カメラのみ・深度推定で堅牢", C_GREEN),
    (0, "⑤ 軽量・エッジ実装 ── 車載計算機で動く", C_TEAL),
    (0, "⑥ 自己ガードレール ── 決定論的な安全層で配備", C_TEAL),
], size=18)

# ---------------------------------------------------------------- 8 vs conventional (native table)
s = slide("従来手法との違い（一目で）")
table(s, [
    ["観点", "従来の自動運転認識", "METEOR"],
    ["アノテーション", "数万〜数十万枚を人手（数ヶ月）", "人手ゼロ・自動生成（数時間で拡張）"],
    ["HDマップ", "事前作成・維持が必要", "地図不要（どこでも走行）"],
    ["開発コード", "大規模な人手開発", "全コードをAIが記述"],
    ["センサー", "高価なLiDAR依存も多い", "カメラのみで推論（深度をAI推定）"],
    ["安全担保", "モデル任せになりがち", "決定論的な自己ガードレール"],
    ["配備", "重く高価な計算機", "軽量・エッジ実装"],
], 0.9, 1.5, 11.5, [2.6, 4.6, 4.3], fs=14)

# ---------------------------------------------------------------- 9 zero annotation
s = slide("① Zero Human Annotation ── 人手ラベルゼロ", "最大のコスト要因を消す",
          band=C_GREEN)
bullets(s, [
    (0, "Co-MLOpsプロジェクトで自動ラベル基盤CoMETにより既にラベル付け済みの"
        "データを再利用（新規の人手ラベルは一切不要）。", C_NAVY),
    (0, "学習データはDRSで収集した日本全国（九州から北海道まで）の走行データ。"
        "地域・道路環境のバリエーションが高い。", C_NAVY),
    (0, "LiDARの点群を走行全体で蓄積し、幾何的に整合させて教師データを自動生成。"),
    (1, "2通りの生成の「一致」だけを採用するコンセンサス方式でラベルノイズを除去。"),
    (0, "人手ラベル0枚のまま、現在7,147シーン・約104万フレームを学習に使用。"),
    (1, "新しい地域・車種のデータが来ても、人手を介さず数時間で学習データに追加。"),
    (0, "→ アノテーション費用が実質ゼロ。スケールが人件費に縛られない。", C_GREEN),
], size=16)

# ---------------------------------------------------------------- 10 zero code
s = slide("② Zero Human Code ── 全コードをAIが記述", "開発速度そのものが競争力",
          band=C_GREEN)
bullets(s, [
    (0, "モデル設計・学習・評価・自動ラベル・エッジ配備まで、全コードをAI"
        "(Claude Fable 5)が記述。"),
    (0, "24時間、人手を待たずに実験・改善を回し続ける。ラウンドを重ねるごとに自動で"
        "精度が積み上がる。"),
    (1, "クラッシュの自動復旧、新データの自動取り込み、失敗事例の自動採掘まで自律運用。"),
    (0, "→ 人的リソースに律速されない、圧倒的な開発スループット。", C_GREEN),
])

# ---------------------------------------------------------------- 10.5 dev history
s = slide("開発の歩み ── 約3週間で多タスク統合まで", "全コードAI記述による開発スピード")
# native horizontal timeline (editable shapes)
tl_y = 3.7
line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(tl_y),
                          Inches(10.9), Pt(4))
_fill(line, C_BLUE)
mile = [
    (1.7, "7/10", "開発スタート", "BEV Segmentation・\n3D BBox 推論", C_BLUE, True),
    (6.3, "7/17", "機能追加", "E2E（経路計画）\n機能を追加", C_TEAL, False),
    (11.3, "現在", "統合・自己改善", "多タスク統合・Refiner\n・エッジ配備まで", C_GREEN, True),
]
for x, date, head, desc, col, above in mile:
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.16), Inches(tl_y - 0.13),
                             Inches(0.34), Inches(0.34))
    _fill(dot, col)
    # date label on the line
    db = s.shapes.add_textbox(Inches(x - 0.7), Inches(tl_y + 0.25), Inches(1.4),
                              Inches(0.4))
    dp = db.text_frame.paragraphs[0]; dp.text = date; dp.alignment = PP_ALIGN.CENTER
    dp.font.size = Pt(16); dp.font.bold = True; dp.font.color.rgb = col
    # description card above or below
    cy = tl_y - 1.85 if above else tl_y + 0.75
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x - 1.5),
                              Inches(cy), Inches(3.0), Inches(1.5))
    _fill(card, RGBColor(0xF2, 0xF4, 0xF6))
    tf = card.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    hp = tf.paragraphs[0]; hp.text = head; hp.alignment = PP_ALIGN.CENTER
    hp.font.size = Pt(14); hp.font.bold = True; hp.font.color.rgb = col
    dp2 = tf.add_paragraph(); dp2.text = desc; dp2.alignment = PP_ALIGN.CENTER
    dp2.font.size = Pt(11); dp2.font.color.rgb = C_NAVY
caption(s, "人手を待たずに24時間開発を継続 ── わずか約3週間で認識から計画まで1モデルに統合",
        y=6.4, col=C_GREEN, size=14)

# ---------------------------------------------------------------- 11 no map
s = slide("③ No Map ── HDマップ不要", "地図の無い場所でも、その場で認識",
          band=C_GREEN)
bullets(s, [
    (0, "事前地図に依存せず、その場のカメラ映像だけで走行空間(BEV)を再構成。"),
    (0, "車線・停止線・横断歩道・走行可能領域・信号までをリアルタイムに推定。"),
    (0, "→ 地図の作成・維持コストが不要。未整備地域や変化する道路にも即対応。", C_GREEN),
])

# ---------------------------------------------------------------- 12 depth robust
s = slide("④ Depthベースで車両にロバスト", "高価なLiDARに頼らずカメラだけで",
          band=C_GREEN)
bullets(s, [
    (0, "8台のカメラ映像からAIがピクセル毎の深度を推定し、正確なBEV(俯瞰)を生成。"),
    (0, "深度に基づく投影で、遠近・オクルージョン・多様な車両形状に堅牢。"),
    (1, "LiDARは学習時の教師にのみ使い、推論はカメラのみ（センサー選択の自由度）。"),
    (0, "→ センサーコストを抑えつつ、実環境の車両・歩行者を安定して捉える。", C_GREEN),
])

# ---------------------------------------------------------------- 13 edge
s = slide("⑤ 軽量・エッジ実装", "データセンター不要、車載で動く", band=C_TEAL)
bullets(s, [
    (0, "現行SoC（NVIDIA Orin）で動作可能な、現実的な構成のCNNベース"
        "効率重視アーキテクチャ。"),
    (1, "LLM・世界モデル（WM）ベースにもできるが、現行ハードウェアでの実行が"
        "難しいため現時点では未着手（将来の拡張余地）。"),
    (0, "TensorRTエンジン + C++ランタイムに変換済み。車載GPUで単体動作。"),
    (0, "カメラのみで完結（LiDAR等はオプション）。追加センサー無しで配備可能。"),
    (0, "→ 量産車への搭載を見据えた、現実的なコストとフットプリント。", C_TEAL),
])

# ---------------------------------------------------------------- 14 guardrail
s = slide("⑥ 自己ガードレール ── 安全に配備する", "AIの判断を決定論で二重化",
          band=C_TEAL)
bullets(s, [
    (0, "AIの経路計画を、ルールベースの決定論的チェックが常時監視。"),
    (1, "衝突予測・赤信号×停止線・曲率の実行可能性・走行可能領域からの逸脱を検査。"),
    (0, "危険と判定すれば計画を却下し、安全停止(MRM)へ。"),
    (0, "→ 「AIが暴走しない」ことを構造で担保。配備の信頼性を高める。", C_TEAL),
])

# ---------------------------------------------------------------- 15 capabilities
s = slide("何ができるか ── 1モデルで多タスク", "カメラのみ・地図なしの実測値（検証データ）")
bullets(s, [
    (0, "BEVセグメンテーション（車線・停止線・横断歩道・走行領域）"),
    (0, "3D物体検出：車両 適合率 83%"),
    (0, "信号認識：全体 74%（青信号 94%）"),
    (0, "経路計画(E2E)：平均誤差 約0.4〜0.9m"),
    (0, "加えて：深度・占有・車線グラフ・未知障害物・他者挙動予測・リスク地図"),
    (1, "後段のRefinerが、遠方や細部を自動で補正しさらに高精度化。"),
], size=16)

# ---------------------------------------------------------------- 16 business value
s = slide("事業価値", "コスト構造を根本から変える", band=C_AMBER)
bullets(s, [
    (0, "アノテーション費 → 実質ゼロ（人手ラベル不要）。"),
    (0, "地図作成・維持費 → 不要（No Map）。"),
    (0, "開発人件費 → 大幅圧縮（コードをAIが記述、24時間稼働）。"),
    (0, "センサーBOM → 低減（カメラ中心、LiDAR任意）。"),
    (0, "→ 新地域・新車種への展開が「データを入れるだけ」でスケール。"
        "限界費用が劇的に低い。", C_AMBER),
])

# ---------------------------------------------------------------- 17 roadmap
s = slide("今後の予定（ロードマップ）", band=C_AMBER)
bullets(s, [
    (0, "データのロバスト化", C_AMBER),
    (1, "NVIDIA Cosmosで生成した多様・希少シーンのデータで学習し、実環境への"
        "堅牢性を高める。"),
    (0, "エッジ最適化・車載SoC実装", C_AMBER),
    (1, "必要な機能に絞り込んだ上で最適化を実施。"),
    (1, "NVIDIA SoC「Orin」および Renesas「R-Car Gen5」への実装を進める。"),
    (0, "リファレンスAI化とWhite Paper公開", C_AMBER),
    (1, "Reference AI（オープンソース）として公開し、業界標準・エコシステム形成を狙う。"),
    (1, "技術詳細をWhite Paperとして公開："
        "github.com/tier4/METEOR/blob/main/paper/main.pdf"),
    (0, "並行して：遠方・未知障害物の精度向上、車種・地域の横展開、自己学習ループの"
        "完全自律化。", C_NAVY),
], size=15)

# ---------------------------------------------------------------- 18 Cosmos demo
demo_slide("Cosmos生成データによるロバスト化 ── デモ",
           "NVIDIA Cosmosで生成した多様・希少シーンでの認識・計画の頑健性",
           band=C_AMBER)

# ---------------------------------------------------------------- 19 closing
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH); _fill(bg, C_NAVY)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.3), Inches(11.3), Inches(3.2))
lines = [("人手ゼロで、自動運転の認識を作り続ける。", 30, C_WHITE, True),
         ("ラベル無し・地図無し・コード無し。カメラだけで、どこでも走る。",
          18, RGBColor(0xB9, 0xCF, 0xE8), False),
         ("", 10, C_WHITE, False),
         ("METEOR ── スケールする自律認識エンジン", 22, C_TEAL, True),
         ("", 12, C_WHITE, False),
         ("Repository:  github.com/tier4/METEOR", 15,
          RGBColor(0xB9, 0xCF, 0xE8), False),
         ("White Paper:  github.com/tier4/METEOR/blob/main/paper/main.pdf", 14,
          RGBColor(0x9F, 0xB8, 0xD4), False)]
for i, (t, sz, c, b) in enumerate(lines):
    p = tb.text_frame.paragraphs[0] if i == 0 else tb.text_frame.add_paragraph()
    p.text = t; p.font.size = Pt(sz); p.font.color.rgb = c
    p.font.bold = b; p.alignment = PP_ALIGN.CENTER

out = "out/METEOR_CEO.pptx"
prs.save(out)
print("saved", out, "-", len(prs.slides._sldIdLst), "slides")
