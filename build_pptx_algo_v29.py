#!/usr/bin/env python3
"""METEOR algorithm-design deck (v29/v30 era) -> out/METEOR_algorithm_v29.pptx.

発表用: 最新のアルゴリズムデザイン全体 — BEV生成、時系列メモリ、12タスク、
マルチモーダルE2E、GT工場、学習インフラ、デプロイ、汎化。"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Emu, Inches, Pt

SW, SH = Inches(13.333), Inches(7.5)
DARK = RGBColor(0x20, 0x28, 0x30)
ACC = RGBColor(0x0E, 0x6E, 0xB8)
GRAY = RGBColor(0x60, 0x68, 0x70)
GREEN = RGBColor(0x1B, 0x78, 0x37)
RED = RGBColor(0xC0, 0x39, 0x2B)
AMBER = RGBColor(0xB9, 0x77, 0x0E)
BG1 = RGBColor(0xF2, 0xF4, 0xF6)
prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide(title=None, sub=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        tb = s.shapes.add_textbox(Inches(0.5), Inches(0.22), Inches(12.3), Inches(0.7))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(27)
        p.font.bold = True
        p.font.color.rgb = DARK
        ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(0.95),
                                Inches(12.3), Emu(1))
        ln.fill.solid()
        ln.fill.fore_color.rgb = ACC
        ln.line.fill.background()
        if sub:
            tb2 = s.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.3), Inches(0.4))
            p2 = tb2.text_frame.paragraphs[0]
            p2.text = sub
            p2.font.size = Pt(14)
            p2.font.italic = True
            p2.font.color.rgb = GRAY
    return s


def bullets(s, items, x=0.6, y=1.5, w=12.1, h=5.6, size=17):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    for i, it in enumerate(items):
        lvl, txt, col, sz = 0, "", None, None
        stack = list(it) if isinstance(it, tuple) else [it]
        while stack:
            e = stack.pop(0)
            if isinstance(e, RGBColor):
                col = e
            elif isinstance(e, str):
                txt = e
            elif isinstance(e, tuple):
                stack = list(e) + stack
            elif isinstance(e, int):
                if e in (0, 1):
                    lvl = e
                else:
                    sz = e
        base = size if lvl == 0 else size - 3
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.font.size = Pt(sz if sz else base)
        p.font.color.rgb = col if col else DARK
        p.font.bold = sz is not None and lvl == 0
        p.space_after = Pt(6)


def big(s, lines, y=2.4):
    tb = s.shapes.add_textbox(Inches(0.8), Inches(y), Inches(11.7), Inches(3.4))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, sz, bold, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = txt
        p.font.size = Pt(sz)
        p.font.bold = bold
        p.font.color.rgb = col
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(10)


def table(s, rows, x, y, w, col_w, fs=13, hdr_fill=RGBColor(0xC9, 0xD9, 0xEA)):
    from pptx.util import Inches as In
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, In(x), In(y), In(w), In(0.34 * nr)).table
    for ci, cw in enumerate(col_w):
        gt.columns[ci].width = In(cw)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            c = gt.cell(ri, ci)
            c.text = str(val)
            pr = c.text_frame.paragraphs[0]
            pr.font.size = Pt(fs)
            pr.font.bold = ri == 0
            pr.font.color.rgb = DARK
            if ri == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = hdr_fill
    return gt


def note(s, txt, y=6.85):
    tb = s.shapes.add_textbox(Inches(0.55), Inches(y), Inches(12.2), Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    p.text = txt
    p.font.size = Pt(12)
    p.font.italic = True
    p.font.color.rgb = GRAY


def pic(s, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(path, Inches(x), Inches(y), **kw)


# 1 title --------------------------------------------------------------
s = slide()
big(s, [
    ("METEOR", 54, True, ACC),
    ("マルチタスク BEV 認識・計画モデル — アルゴリズムデザイン v29/v30/v31", 28, True, DARK),
    ("カメラオンリー運用(LiDARはオプション入力・同一重み) / 人手ラベル0・人手コード0(全コードをClaudeが作成)", 16, False, GRAY),
    ("12タスク同時学習 | TensorRT実機デプロイ | 2026-07", 15, False, GRAY),
], y=2.1)

# 2 design principles ---------------------------------------------------
s = slide("設計原則", "なぜこの形か — 3つの柱")
bullets(s, [
    (ACC, 22, "1. 幾何は計算で、曖昧さは学習で"),
    (1, "画像→BEV投影は深度ゲートIPM(厳密な幾何)。学習するのは深度分布のみ — BEVFormer型の学習投影は採らない"),
    (1, "利点: 学習が安定・データ効率が高い・国や車種が変わっても投影は壊れない(→米国ゼロショットで実証)"),
    (ACC, 22, "2. GTは自動工場で無限に作る"),
    (1, "t4dataset(LiDAR+8cam)から17ステージのautolabel工場でBEVレーン/3D箱/OCC/リスク/レーングラフGTを全自動生成"),
    (1, "怪しいラベルは捨てずに ignore(255) — 「無いものをでっち上げない」が全ステージ共通の規約"),
    (ACC, 22, "3. 全タスクを1つのBEVで — ただし時系列は選択的に"),
    (1, "幾何タスク(レーン/OCC)は現在フレームBEV、動的タスク(検出/予測/E2E)は時系列融合BEVを使う task routing"),
], y=1.45, size=18)

# 3 architecture --------------------------------------------------------
s = slide("全体アーキテクチャ (v38/v39)", "8 cam → depth-gated IPM → BEV 96ch → 3-slot memory → 12 heads + optional LiDAR/intent")
pic(s, "docs/media/architecture.png", 0.65, 1.5, w=12.0)

# 3.1-3.4 task detail diagrams -------------------------------------------
for ttl, sub, img in (
    ("詳細: BEV生成", "depth-gated IPM + オプションLiDAR (C6a/C6b) — ゼロ入力=カメラオンリーbit一致", "detail_bev.png"),
    ("詳細: 時系列メモリとtask routing", "3スロット+B2ゲート・幾何=RAW/動的=FUSED・運動残差", "detail_temporal.png"),
    ("詳細: E2E計画スタック+ガードレール", "意図・運動学・B3・リスク統合選択・v39デカップリング・C7", "detail_e2e.png"),
    ("詳細: 知覚ヘッド群", "3D検出・unknown・予測・B1レーングラフ・OCC+flow", "detail_heads.png")):
    s = slide(ttl, sub)
    pic(s, f"docs/media/{img}", 0.9, 1.6, w=11.5)

# 4 BEV generation ------------------------------------------------------
s = slide("BEV生成: 深度ゲート IPM", "学習するのは「どの画素を信じるか」だけ")
bullets(s, [
    ("8カメラ 768×432 → 共有2D backbone → 画素毎の深度分布(softmax, 0-80m)を予測"),
    ("各BEVセル(0.2m格子, ±40×±25m→800×500)へ厳密な幾何投影し、深度確率で重み付け(= depth gating)"),
    ("誤マッチの抑制: 深度が合わない画素は自動的にBEVへ寄与しない — 「持ち上げ」を学習しない"),
    (GREEN, "強み: キャリブが正しければ投影は常に正しい。汎化・安定性・データ効率で有利"),
    (AMBER, "コスト: IPM ForeignNode が推論時間の60.8%(後述のOrin最適化ポイント)"),
    ("2D側の副生成物: 21クラス2D Seg・2D検出も同じbackboneから出力(マルチタスクの正則化にも寄与)"),
], size=17)

# 5 temporal memory ------------------------------------------------------
s = slide("時系列メモリキュー", "3スロット (t−0.4 / −1.2 / −2.8s) + ego-motion warp + task routing")
bullets(s, [
    ("過去BEV特徴を3スロット保持し、自車運動で現在座標系へアフィンwarpして tfuse3(1×1 conv)で融合"),
    ("Task routing: 幾何ヘッド(レーンseg・OCC)= 現在BEVのみ / 動的ヘッド(3D検出・軌跡・E2E・flow)= 融合BEV"),
    (1, "理由: 動物体のゴースト(過去位置の残像)が幾何タスクを汚すため"),
    (RED, 18, "学習で踏んだ地雷: BN統計汚染"),
    (1, "履歴フレームのforward(3回)が現在フレーム(1回)よりBN running statsを支配、無効スロットの全零画像(~16%)も混入"),
    (1, "→ 訓練lossは健全なのにeval性能が50stepで崩壊。model.eval()で履歴パスを包んで解決(mIoU 0.213→0.265@step50)"),
    ("スロット初期化: v28の学習済みtfuseをslot0に移植し新スロットは零初期化 — 初期状態でv28と出力bit一致を保証"),
], size=17)

# 6 task heads ----------------------------------------------------------
s = slide("12タスクヘッド", "1つのBEV特徴から全て — v30でunknown検出を追加")
table(s, [
    ["ヘッド", "出力", "GTソース", "備考"],
    ["BEVレーンseg", "11cls 800×500 (0.2m)", "LiDAR強度+HDマップレス自動抽出", "最優先タスク"],
    ["2D seg / 2D det", "21cls / 箱", "オープンボキャブラリ蒸留", "backbone正則化"],
    ["BEV 3D検出", "veh/VRU 中心+サイズ+向き", "LiDARクラスタ+カメラ確認", "最優先タスク"],
    ["unknown検出 (v30)", "固定0.4m箱 (コーン等)", "OCC小ブロブ抽出 (≤2.4m², ≤1.6m)", "ヘッド分離"],
    ["エージェント軌跡", "3s先 6点×K", "追跡スムージング", "クラス条件付き(後述)"],
    ["E2E自車計画", "K=3経路+信頼度+制御", "実走行ログ", "最優先タスク"],
    ["信号/リスク/静止", "TL状態・リスク場・静止flag", "画像分類蒸留・将来占有", ""],
    ["OCC 3D / flow", "16層ボクセル10cls・速度場", "LiDAR累積+動的±1frame", "影フィルタ(後述)"],
    ["レーングラフ", "24スロット折れ線+隣接", "レーンseg骨格化", "P/R 0.01 — B1で刷新予定"],
], x=0.5, y=1.5, w=12.35, col_w=[2.5, 3.1, 3.55, 3.2], fs=12)
note(s, "全ヘッドが同時学習。GT欠損は255-ignoreでスキップ — タスク間でデータ量が不揃いでも成立")

# 7 E2E multimodal -------------------------------------------------------
s = slide("マルチモーダル E2E (K=3)", "モード崩壊とその解 — eps-WTA + 多様初期化")
bullets(s, [
    ("K=3の経路仮説(直進/左/右バイアス初期化 ±0.8m@3s)+ softmax信頼度、学習は Winner-Takes-All"),
    (RED, "観測された failure: 3本の経路がほぼ同一方向に収束(ユーザー指摘で発覚)"),
    (1, "純WTAでは初期に僅かに良い1モードが全勝ち→他モードに勾配が流れず複製のまま固定"),
    (GREEN, "解1: eps-WTA (ε=0.1) — 敗者モードにも ε 分の勾配を常に流す"),
    (GREEN, "解2: 学習済み42ch headの再多様化(平均バイアス+横方向スプレッド注入)"),
    (GREEN, "解3: データ側 — 旋回フレーム(|lat@3s|>4m, 走行中の7.5%)を3倍オーバーサンプル(5%→12%)"),
    ("モード選択CE重み 0.3→0.6 (r22): 交差点で正しい分岐に高信頼度を割り当てる圧を強化"),
], size=17)

# 8 class-aware traj ------------------------------------------------------
s = slide("クラス条件付きエージェント軌跡", "「歩行者が車の向きに歩く」バグの根治")
bullets(s, [
    (RED, "症状: 歩行者の予測進行方向が常に車両と同じ向き(heading誤差 75.9°)"),
    ("原因: 軌跡ヘッドの入力にクラス情報が無く、多数派(車両)の運動事前分布に引きずられていた"),
    (GREEN, "修正: traj_feat = concat(traj_stem(融合BEV), det_feat.detach()) — 検出特徴を切り離して注入"),
    (1, "detach が要点: 軌跡lossが検出ヘッドを劣化させない(BEV Seg・3D検出の精度を守る優先則)"),
    (GREEN, "VRUセルのloss重み×2.5 + クラス別評価指標 vehADE/vruADE/vehHead/vruHead を新設し回帰を常時監視"),
], size=18)

# 8.5 oncoming flip diagnosis --------------------------------------------
s = slide("対向車の向き反転 — 定量診断と修正 (r23)", "「対向車が自車方向を向く」の正体は二峰性のflip")
table(s, [
    ["相対方向", "自車状態", "flip率 (>120°)", "中央値誤差"],
    ["同方向", "全状態", "1〜2.5%", "2〜5°"],
    ["対向", "走行中/発進", "21〜23%", "2〜9°"],
    ["対向", "停止中", "51%", "122°"],
    ["横断", "停止中", "46%", "104°"],
], x=0.7, y=1.5, w=11.9, col_w=[2.6, 2.9, 3.4, 3.0], fs=14)
bullets(s, [
    ("中央値は低い=大半は正しいが、5台に1台(停止中は2台に1台)が自車進行方向へ反転"),
    ("原因: 3D検出は現在フレームのみ(task routing)で向きは見た目依存。運動方向の手がかりは融合BEVのスミアを浅いstemが暗黙復号するだけ → 曖昧なとき多数派(先行車=自車と同方向)の事前分布に落ちる"),
    (GREEN, "修正1: 運動残差の明示入力 — traj入力に「現在BEV − ego-warp済みt−0.4sスロット」を追加(96→192ch)。対向車は真の運動方向の符号付きダイポールとして線形に読める。TRT-safe"),
    (GREEN, "修正2: 対向車セルのloss重み×2.5(|相対yaw|>135°、VRUと同格の少数派扱い)"),
    ("グラフト: 新チャネル零初期化で初期挙動は旧モデルとbit一致 → 精度を落とさず載せ替え"),
], x=0.6, y=3.6, w=12.1, h=3.4, size=15)

# 9 GT factory ----------------------------------------------------------
s = slide("GT自動工場の最新改良", "OCC影フィルタ / unknown物体GT")
bullets(s, [
    (ACC, 19, "OCC dynamic-shadow フィルタ (stage 16)"),
    (1, "動的クラスは±1フレーム累積のため移動車が~4mスミア。近傍ego車両voxelの18%がGT箱の外=幻の教師"),
    (1, "3D GT箱footprint(0.6m膨張)の外の veh/2輪/歩行者 voxel → 255 ignore(freeにはしない: 未確認実車の可能性)"),
    (ACC, 19, "unknown物体GT (stage 17) — v30で検出ヘッド化"),
    (1, "OCC obstacleクラスの小ブロブ(≤2.4m², 高さ≤1.6m)の重心を抽出 → コーン/ガイドポスト等を固定0.4m箱で検出"),
    (1, "既存アノテーションに無いクラスをOCC GT経由で「発明」— ラベル体系の外の物体に対応"),
], x=0.6, y=1.4, w=6.3, h=5.6, size=15)
pic(s, "out/assets_v29_occgt_still.png", 7.1, 1.7, w=5.8)
note(s, "右: OCC GT (LiDAR累積・LUT 10クラス・±40m 0.4m格子・16層 z∈[−1.0, 5.4))")

# 10 training infra -------------------------------------------------------
s = slide("学習インフラ", "データ量最大化と90秒フィードバックループ")
bullets(s, [
    (ACC, 19, "EpochSubsetSampler — コーパス利用率 14% → 100%"),
    (1, "従来: 固定シードで全シーンの14%を毎epoch再利用 → epoch毎に新しいランダム部分集合を引き直す方式に変更"),
    (1, "計算コスト増ゼロで実効データ量を7倍化。重み付きサンプリング対応(旋回×3のオーバーサンプルに使用)"),
    (ACC, 19, "--val-every ステッププローブ"),
    (1, "N step毎にrank0がBEV mIoU+3D det指標を測定(~90秒)。75分/epochを待たず崩壊を検知 — BN汚染の発見に直結"),
    (ACC, 19, "ラウンド運用"),
    (1, "r20: BN修正+全量データ / r21: eps-WTA+クラス条件軌跡 / r22: 影フィルタOCC+旋回×3 / r23: v30 unknown(自動起動待機)"),
], size=16)

# 10.5 optional LiDAR (v31 / C6a) -----------------------------------------
s = slide("オプションLiDAR入力 (v31)", "同一重みでカメラオンリー/LiDAR併用の両推論 — r23で学習・両モードのデモ動画あり")
bullets(s, [
    (ACC, 19, "設計: 深度分布のシャープ化(不確かさの源泉に最小介入)"),
    (1, "LiDAR点群を各カメラへ投影した疎な深度マップを入力に追加(GT工場のdepth4と同形式=学習側の追加コストゼロ)"),
    (1, "実測がある画素だけ、予測深度softmaxを実測ビンへブレンド: dprob' = (1−α·m)·dprob + α·m·tri(d)  (α学習可能)"),
    (1, "全て要素演算 → TRT-safe。推論時間への影響ほぼゼロ(実測 it/s 同等)"),
    (ACC, 19, "「同一重みで両モード」の仕組み"),
    (1, "LiDAR=ゼロ入力 ⇔ カメラオンリーとbit一致(検証済み)→ ONNX/エンジンは1つ、点群を入れるかゼロを入れるかだけ"),
    (1, "modality dropout(学習中50%でLiDARを隠す)→ BN統計が両モードで較正され、LiDAR併用学習の勾配がカメラオンリー性能も押し上げる設計"),
    (1, "probeがカメラのみ/+lidarのmIoUを併記 — 両モード差を90秒で常時監視"),
    (GRAY, "拡張余地(C6b): LiDAR BEV特徴ブランチの残差融合 — 3D検出/OCCの上積み狙い、Orinコストと相談"),
], size=15)

# 11 results -------------------------------------------------------------
s = slide("現状の精度", "held-out 走行日で評価(訓練と完全分離)— r22完走時点")
table(s, [
    ["指標", "r20 (基準)", "r22 (最新完走)", "傾向"],
    ["BEVレーン mIoU", "0.308", "0.302", "→ (維持)"],
    ["2D seg mIoU (21cls)", "0.539", "0.535", "→"],
    ["3D det 車両 P / Rn", "0.83 / 0.72", "0.85 / 0.72", "↑"],
    ["3D det yaw誤差 / 向き反転", "6.4° / —", "5.8° / 10%", "↑"],
    ["3D det VRU P / Rn", "0.75 / 0.45", "0.73 / 0.44", "→ (A4課題)"],
    ["E2E ADE / ADEc", "0.71 / 0.78 m", "0.73 / 0.72 m", "ADEc改善"],
    ["TL精度", "0.28 (退行)", "0.86", "回復 (tl-w 0.6)"],
    ["agent軌跡 vehHead / vruHead", "— / 75.9°", "34° / 74°", "対向flipはr23で対処"],
    ["レーングラフ P/R", "0.01", "0.01", "構造課題 → B1"],
], x=0.7, y=1.45, w=11.9, col_w=[3.9, 2.7, 3.0, 2.3], fs=13)
note(s, "優先則: BEV Seg・BEV 3D・E2Eの精度を下げる変更は入れない。r23(v31)はep0でmIoU 0.309とr22超え、カメラ/+lidar両モードを並行監視中")

# 12 deployment ----------------------------------------------------------
s = slide("デプロイ: TensorRT フレームワーク", "生t4dataset → ONNX 18出力 → fp16エンジン → 動画/npz")
bullets(s, [
    ("export_onnx.py: 18出力(lane/det/traj/E2E/TL/risk/occ/flow/レーングラフ/unknown)+ 履歴BEV/theta入力。ORT一致 ~1e-5"),
    ("runtime.py: 3スロット履歴リング(実時間オフセット2/6/14フレーム)をホスト側で管理 — エンジンは純粋なfeed-forward"),
    ("infer_t4dataset.py: 生のt4datasetを直接推論(変換不要)→ フレーム毎npz + オーバレイ動画"),
    ("L40S fp16 実測 68.7ms/frame。内訳: depth-gated IPM 41.8ms (60.8%)、12ヘッド合計 ~4%"),
    (AMBER, "Orin見積り: ボトルネックはメモリ帯域(×4.2)。ヘッドのTransformer化は速度に無関係 — IPMセクターマスキングが本命、次にINT8"),
], size=16)

# 13 US generalization ----------------------------------------------------
s = slide("ゼロショット汎化: 米国データ", "日本のみで学習 → 車種・国・右側通行が違うデータへ無変更適用")
pic(s, "out/assets_us_zeroshot.png", 0.9, 1.45, w=9.2)
bullets(s, [
    (GREEN, "動作: 複数車線再構成"),
    (GREEN, "米国トラック3D検出"),
    (GREEN, "横型信号の認識"),
    (GREEN, "114km/h(未学習速度域)"),
    (RED, "弱点: 信号なし道路でTL誤出力"),
    (RED, "路面クラスの過拡張"),
], x=10.3, y=1.6, w=2.9, h=5.4, size=13)
note(s, "幾何ベースIPMの汎化上の利点が実証された形 — 学習した投影ではないため座標系は壊れない")

# 14 roadmap -------------------------------------------------------------
s = slide("今後の候補(優先順)", "ルール: 10分probeで測定できない変更はラウンドにしない")
table(s, [
    ["#", "項目", "内容", "状態"],
    ["1", "TL重み回復 (A3)", "loss配分の飢餓が原因 → tl-w 0.6", "✅ r22で0.86回復"],
    ["2", "対向車軌跡flip", "運動残差入力+対向×2.5重み", "✅ 実装済 r23検証中"],
    ["3", "LiDARオプション入力 (C6a)", "深度シャープ化・同一重み両モード", "✅ 実装済 r23学習中"],
    ["4", "時系列ゲート (B2)", "セル毎slot attention。ゴースト抑制→task routing廃止の可能性", "S"],
    ["5", "レーングラフ刷新 (B1)", "DETR型query decoder(24 query×352 token, TRT-safe)。唯一の全損ヘッド", "M"],
    ["6", "リスク場×E2E統合 (C1)", "K=3経路をリスク場で線積分→信頼度×安全性で選択。再学習不要", "S"],
    ["7", "E2E attentionプール (B3) / INT8 (C4) / LiDAR BEVブランチ (C6b)", "上記の結果を見て順次", "M"],
], x=0.6, y=1.5, w=12.1, col_w=[0.6, 3.4, 5.6, 2.5], fs=13)
note(s, "却下済み: BEVFormer型 image→BEV cross-attention(400k query×165k key で非現実的+deformableはTRT plugin)")

# 15 takeaways -----------------------------------------------------------
s = slide()
big(s, [
    ("まとめ", 40, True, ACC),
    ("幾何は計算・曖昧さは学習 — depth-gated IPM が安定性と汎化の土台", 20, False, DARK),
    ("GT工場が全て — 人手ラベル0のまま12タスク・2,800+シーンを供給し続ける", 20, False, DARK),
    ("90秒probe + ラウンド運用 — 崩壊を数分で検知し、優先タスクの精度を守りながら前進", 20, False, DARK),
    ("実機まで一直線 — 生データ→TRT fp16 68.7ms、米国データにもゼロショットで動作", 20, False, DARK),
    ("センサ構成に自由度 — v31: LiDARは同一重みのオプション入力(入れても外しても動く)", 20, False, DARK),
], y=2.0)

import os
os.makedirs("out", exist_ok=True)
prs.save("out/METEOR_algorithm_v29.pptx")
print("saved out/METEOR_algorithm_v29.pptx", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
