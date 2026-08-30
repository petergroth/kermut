# Kermut を FLIP の GB1 で動かす

Kermut は ProteinGym のベンチマークに合わせて作られている。これを FLIP
（<https://github.com/J-SNACKKB/FLIP>）の GB1 landscape で動かすために何をしたかの記録。

**結論から言うと、モデル本体（カーネル・GP）は一切変えていない。** 変えたのはデータの
入り口だけ。Kermut のカーネルは「配列 + 構造 + 埋め込み」があれば動くので、FLIP の
データをその形に整える変換層を足せば済む。

---

## 1. なぜ変換が必要なのか

Kermut は 1 データセットにつき、以下のファイルが決まった場所に決まった形式であることを
前提にしている（`src/data/data_utils.py` を読むと分かる）。

| 要素 | パス | 何に使うか |
|---|---|---|
| 参照メタデータ | `data/DMS_substitutions.csv` | DMS_id・WT配列・UniProt_ID を引く |
| アッセイCSV | `data/substitutions_singles/{DMS_id}.csv` | 変異・配列・目的値・fold列 |
| ESM-2 埋め込み | `data/embeddings/substitutions_singles/ESM2/{DMS_id}.h5` | global カーネルの入力 |
| ProteinMPNN 分布 | `data/conditional_probs/ProteinMPNN/{DMS_id}.npy` | 変異カーネルの入力 |
| CA 座標 | `data/structures/coords/{DMS_id}.npy` | 距離カーネルの入力 |
| ゼロショット | `data/zero_shot_fitness_predictions/ESM2/650M/{DMS_id}.csv` | GP の平均関数 |

FLIP はこのどれも満たさない。特に食い違うのが次の4点。

### (a) 変異の表記

- ProteinGym: `V39A`（WT残基・位置・変異後残基）、多重変異は `V39A:D40E`
- FLIP GB1: `ADGV` という4文字の短縮記法（4サイトに入るアミノ酸を並べただけ）

### (b) 配列の中身

FLIP の `sequence` 列は **265残基** ある。これは PDB 5LDE 由来の融合タンパクで、

```
1-56    GB1 本体          ← 変異が入るのはここだけ
57-     リンカー + Viral FLICE + His-tag
```

という構成。57残基目以降は全変異体で完全に同一。

### (c) split の意味

- ProteinGym: `fold_random_5` 等の **5-fold 交差検証**（fold列に0-4）
- FLIP: **単一の train/test split**（`one_vs_rest` 等の列に `train`/`test` の文字列）

Kermut のベンチは「各 fold を順に test にする」ループなので、そのままでは形が合わない。

### (d) 評価対象の行

`four_mutations_full_data.csv` は全 **149,361行** あるが、FLIP の公式 split で
train/test が割り当てられているのは **`keep=True` の 8,733行だけ**。残りは空欄。

---

## 2. どう解決したか

### (a) 変異表記 → 変換する

`ADGV` を `V39A` に変換する。変異位置は 1-indexed で **39, 40, 41, 54**、
WT の4残基は **VDGV**（WT配列上で検証済み）。WT と同じ残基の位置は記法に含めない。

WT行（`VDGV`、変異なし）だけは記法が空文字になる。これを放置すると CSV 経由で NaN に
なり、埋め込みのラベル照合（`assert mutants == all_labels`）とゼロショットの merge を
壊すので、**明示的に `"WT"` という名前を付けている**。

### (b) 配列 → 先頭56残基に切り詰める

57残基目以降が全変異体で同一なことをスクリプト内で検証したうえで、GB1 本体だけ残す。

265残基のまま渡しても結果は変わらないが、one-hot が `265*20` 次元、距離行列が
`265x265`、ESM-2 も毎回265残基分…と、**情報が増えないのに計算量だけ増える**。

### (c) split → fold列の形に写す

ベンチ側を大きく書き換えたくないので、アダプタ側で写像する。

```
FLIP の test  -> fold 0
FLIP の train -> fold 1
```

そのうえでベンチ側に「`fold_flip_` で始まる split のときは fold 0 を test にする回だけ
回す」という分岐を1つ足した。するとループが1回だけ回り、

```
train_idx = (fold != 0) = fold 1 = FLIP の train
test_idx  = (fold == 0) = fold 0 = FLIP の test
```

となって FLIP の分割がそのまま再現される。

> **注意**: この対応を逆（train=0）にすると train を予測してしまう。また分岐を消すと
> fold 1 を test にする回も回り、無意味な結果が混ざる。

### (d) 評価対象 → `keep=True` で絞る

ここを間違えると「FLIP の split で評価した」と言えなくなり、他手法のスコアと比較できない。
内訳は HD=0:1 / HD=1:27 / HD=2:396 / HD=3:2566 / HD=4:5743 = 8,733。

---

## 3. 構造をどうしたか（5LDE ではなく 1PGA を使う）

**これが今回いちばん判断を要した箇所。**

FLIP の配列は 5LDE 由来なので、素直に考えれば構造も 5LDE を使う。しかし 5LDE を調べると
**GB1領域(1-56)に13残基の欠損がある**（chain A: 残基 1, 9-19, 55 が欠損）。

これが問題になるのは、Kermut の座標抽出（`src/data/extract_3d_coords.py`）が
**「PDB に出てくる CA を順に並べれば WT配列と1対1で対応する」前提**で書かれているため。
欠損があると 43個の座標が 56残基分の配列に詰められ、**インデックスが全部ずれる**。
しかもエラーは出ず、静かに間違った距離行列ができる。

そこで **1PGA**（GB1 単体の構造）に差し替えた。

| | 5LDE chain A | 1PGA |
|---|---|---|
| GB1領域(1-56)の欠損 | 13残基（1, 9-19, 55） | **なし** |
| FLIP配列との差 | 0箇所 | 位置2のみ（T vs Q） |
| 変異4サイトの座標 | あり | あり |

1PGA は位置2が `T`、FLIP配列は `Q` で1残基だけ違う。ただし
**変異4サイト(39/40/41/54)から遠く、距離カーネルへの実質的な影響はない**。
「13残基が欠けてインデックスがずれる」ことと比べれば、こちらの方がはるかに安全。

このため座標抽出と ProteinMPNN 変換は Kermut 本体のものを使わず、GB1 用に切り出した。
本体は `assert wt_seq_from_toks == wt_sequence` で**完全一致を要求する**ので、
位置2の差で止まってしまうため。切り出した版は許容ミスマッチ数を
`--max-mismatch`（既定1）で明示的に持たせ、**それを超えたら止める**ようにしてある。
「1残基だけ違うのは想定内、それ以上は異常」を機械的に区別するため。

---

## 4. 変更したファイル

### 新規追加

| ファイル | 役割 |
|---|---|
| `scripts/flip/prepare_flip_gb1.py` | FLIP → ProteinGym 形式変換（上記 a〜d 全部）。前提を assert で検証 |
| `scripts/flip/extract_gb1_coords.py` | 1PGA から CA 座標抽出。欠損は NaN 埋め、ミスマッチ数を検査 |
| `scripts/flip/extract_gb1_conditional_probs.py` | ProteinMPNN 出力の変換（完全一致 assert を緩和した版） |
| `example_scripts/flip_gb1_preprocess.sh` | 前処理一式 |
| `example_scripts/benchmark_flip_gb1.sh` | 5 split の実行 |

### 既存ファイルへの変更

**`src/experiments/proteingym_benchmark.py`** — 実質1箇所（+説明コメント）

- `unique_folds` を `fold_flip_*` のとき fold 0 のみに絞る分岐を追加
- `use_multiples` は変更していないが、「多重変異なのに singles を読んでいて大丈夫か？」
  と後で混乱しやすいので理由をコメントで明示

ProteinGym の split_method は `fold_flip_` で始まらないので、**従来の挙動は一切変わらない**。

**`pyproject.toml`** — マージ衝突の解決に伴い `packages.find` を `kermut` → `src` に修正
（このリポジトリは `src/` 版をベースにしているため）

**`.gitignore`** — `scripts/` の一括除外を `scripts/*` + `!scripts/flip/` に変更し、
今回書いたスクリプトが追跡されるようにした

---

## 5. モデルの入手と配置（ProteinMPNN / ESM-2）

Kermut 本体の README は「公式リポジトリからインストールして環境変数を設定」としか
書いていないので、実際に何をどこに置くかをここに補う。

### 5.1 ProteinMPNN

**HuggingFace ではない。** GitHub リポジトリで、しかも **モデル重みがリポジトリに
同梱されている**（各 6.7MB の `.pt` ファイル）。`pip install` も重みの別途ダウンロードも
不要で、clone するだけで完結する。

| 項目 | 内容 |
|---|---|
| 入手元 | <https://github.com/dauparas/ProteinMPNN>（MIT ライセンス） |
| サイズ | 約 120MB（重み込み） |
| 重みの場所 | `vanilla_model_weights/` `soluble_model_weights/` `ca_model_weights/` |
| Kermut が使う重み | `vanilla_model_weights/v_48_020.pt` |
| 依存 | torch, numpy のみ |

```bash
# clone するだけ。重みも一緒に入ってくる
git clone https://github.com/dauparas/ProteinMPNN.git ~/tools/ProteinMPNN

# Kermut / 前処理スクリプトはこの環境変数を見る
export PROTEINMPNN_DIR=~/tools/ProteinMPNN
# 毎回設定するのが面倒なら ~/.bashrc に追記しておく
```

**なぜ「clone したディレクトリ」を指す必要があるのか。**
`protein_mpnn_run.py` は重みのパスを *スクリプト自身の位置から* 相対的に解決する:

```python
file_path = os.path.realpath(__file__)
k = file_path.rfind("/")
model_folder_path = file_path[:k] + '/vanilla_model_weights/'
```

つまり `protein_mpnn_run.py` だけを別の場所にコピーすると重みを見失う。
リポジトリごと置いて、そのパスを `PROTEINMPNN_DIR` に入れること。

使う重みは `--model_name` の既定値 `v_48_020`（48エッジ・ノイズ0.20Å）。
Kermut 本家の実行例も既定のままなので、こちらも変えていない。

### 5.2 GB1 で特に注意する点（PDBのファイル名）

ProteinMPNN の出力 `.npz` の **ファイル名は入力 PDB のファイル名から決まる**。
`protein_mpnn_utils.py` の `parse_PDB` が

```python
my_dict['name'] = biounit[(fi+1):-4]   # ディレクトリと ".pdb" を除いた部分
```

として名前を決め、これがそのまま出力ファイル名になる。

一方 Kermut 側は、参照ファイルの `UniProt_ID` を使って

```
data/conditional_probs/raw_ProteinMPNN_outputs/{UniProt_ID}/proteinmpnn/conditional_probs_only/{UniProt_ID}.npz
```

を探す。GB1 では `UniProt_ID = GB1_1PGA` としているので、
**PDB は `GB1_1PGA.pdb` という名前でなければならない**。
`1pga.pdb` のまま置くと `1pga.npz` が出力され、変換スクリプトが見つけられない。

配置:

```
data/structures/pdbs/GB1_1PGA.pdb     ← RCSB の 1PGA をこの名前で置く
```

```bash
curl -o data/structures/pdbs/GB1_1PGA.pdb https://files.rcsb.org/download/1PGA.pdb
```

なお 1PGA は **チェーンA単独・56残基・HETATM は水のみ**なので、
チェーン指定のオプション類は一切不要（`--pdb_path` だけでよい）。

実行コマンド（`example_scripts/flip_gb1_preprocess.sh` の手順3が中でやっていること）:

```bash
DATASET=GB1_1PGA
mkdir -p data/conditional_probs/raw_ProteinMPNN_outputs/${DATASET}/proteinmpnn

python $PROTEINMPNN_DIR/protein_mpnn_run.py \
    --pdb_path data/structures/pdbs/${DATASET}.pdb \
    --save_score 1 \
    --conditional_probs_only 1 \
    --num_seq_per_target 10 \
    --batch_size 1 \
    --out_folder data/conditional_probs/raw_ProteinMPNN_outputs/${DATASET}/proteinmpnn \
    --seed 37
```

- `--conditional_probs_only 1`: 配列を設計させるのではなく、各残基の条件付き
  アミノ酸分布だけを出させるモード。Kermut の変異カーネルが必要とするのはこれ。
- `--num_seq_per_target 10`: デコード順をランダムに変えて10回計算する。
  変換スクリプト側で `log_p.mean(axis=0)` して平均を取っている。
- `--seed 37`: Kermut 本家の例に合わせた固定値（再現性のため）。

GB1 は56残基と小さいので **CPU でも数秒〜数十秒で終わる**。
GPU が要るのは ESM-2 の方だけ。

### 5.3 ESM-2 (650M)

こちらも HuggingFace ではなく、**Meta の配布URLから直接ダウンロード**して
`models/` に置く（Kermut 本体の README に従う）。`esm` パッケージ自体は
`environment.yml` 経由で入る。

```bash
curl -o models/esm2_t33_650M_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
curl -o models/esm2_t33_650M_UR50D-contact-regression.pt \
    https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

配置:

```
models/esm2_t33_650M_UR50D.pt                      （約 2.5GB）
models/esm2_t33_650M_UR50D-contact-regression.pt
```

埋め込み抽出とゼロショット計算は **GPU 環境（`gpu-a03`, conda `sample_pytorch`）で回す**。
FLIP GB1 は 8,733 配列 × 5 split なので、ローカル CPU で走らせないこと。

---

## 6. 実行手順

事前準備（詳細は5章）:

```bash
# a. FLIP のデータ
#    https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1 から取得し
#    four_mutations_full_data.csv を data/flip/ に置く

# b. GB1 の構造（ファイル名は GB1_1PGA.pdb でなければならない。5.2 参照）
curl -o data/structures/pdbs/GB1_1PGA.pdb https://files.rcsb.org/download/1PGA.pdb

# c. ProteinMPNN（clone するだけ。重みは同梱されている）
git clone https://github.com/dauparas/ProteinMPNN.git ~/tools/ProteinMPNN
export PROTEINMPNN_DIR=~/tools/ProteinMPNN

# d. ESM-2 650M の重みを models/ に置く
curl -o models/esm2_t33_650M_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
curl -o models/esm2_t33_650M_UR50D-contact-regression.pt \
    https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

実行:

```bash
# 前処理（変換 → 座標 → ProteinMPNN → ESM-2）
bash example_scripts/flip_gb1_preprocess.sh

# 5 split で評価
bash example_scripts/benchmark_flip_gb1.sh
```

**どこまでが CPU で回るか**:

| 手順 | 環境 | 所要 |
|---|---|---|
| FLIP変換・座標抽出 | CPU で十分 | 数十秒 |
| ProteinMPNN | CPU で十分（GB1 は56残基） | 数秒〜数十秒 |
| ESM-2 埋め込み・ゼロショット | **GPU 必須**（`gpu-a03`, conda `sample_pytorch`） | 8,733配列 × 5 split |
| ベンチマーク | GPU 推奨 | split による |

ProteinMPNN は GPU 必須と思われがちだが、GB1 程度の小さいタンパク質なら CPU で済む。
重いのは ESM-2 の方だけ。

---

## 7. どこまで検証したか（重要）

**検証したのは「配線が正しいこと」だけで、予測精度は一切検証していない。**

CPU 環境には `esm` が入っておらず CUDA も無効なので、埋め込み・ゼロショット・
条件付き分布を**ダミーの乱数で埋めて**、train/test 各150件のスモークテストを実施した。

確認できたこと:

- フル Kermut 構成（変異カーネル + 距離 + ProteinMPNN分布 + 埋め込み + ゼロショット）が完走する
- 予測対象が **FLIP の test セットと厳密に一致**（mutant集合が完全一致）
- **fold 0 のみ**が評価される（fold 1 が混ざっていない）
- NaN なし、予測分散がすべて正

確認していないこと:

- **予測精度・Spearman 相関**。入力がダミー乱数なので、この時点の予測値に意味はない。
  実際の性能は GPU で本物の埋め込みを作ってから測る。

スモークテスト用の一時ファイルは削除済み。

---

## 8. 補足・今後の検討事項

- **埋め込みの重複計算**: 現状は split ごとに埋め込みを作るので `8,733配列 × 5 split`
  分になる。split 間で配列集合はほぼ重複しているため、1回だけ計算して使い回す最適化の
  余地がある（未実装）。
- **`low_vs_high` の解釈**: train が「WT以下の fitness」なので、外挿性能を見る split。
  他の4つとは性質が違うので、結果をまとめるときは分けて論じた方がよい。
- **`one_vs_rest` は train が28件しかない**。GP のハイパラ推定が不安定になりうるので、
  結果が荒れたらこの点を疑う。
