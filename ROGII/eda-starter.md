# ROGII - Wellbore Geology Prediction EDA 要約

このノートブックは、Kaggle コンペティション **ROGII - Wellbore Geology Prediction** 向けのスターター EDA です。目的は、水平坑井の既知区間から先にある隠し評価区間について、`TVT`（True Vertical Thickness）を予測することです。

評価指標は、予測した `tvt` と正解 `TVT` の RMSE です。

## タスク概要

- 各水平坑井は `MD`（Measured Depth）に沿った系列データとして与えられる。
- 予測開始前は `TVT_input` に既知の `TVT` が入っている。
- 予測開始後は `TVT_input` が欠損し、その区間の `TVT` を推定する必要がある。
- 水平坑井の `XYZ` 座標、`GR`（Gamma Ray）、既知区間の `TVT_input`、対応する Typewell の `TVT` と `GR` が重要な手掛かりになる。
- `TVT` は `MD` に対して単調増加とは限らず、増加・減少・ほぼ一定のどれも起こり得る。

## データ構成

学習データには、各井戸について次のファイルがある。

- `{WELLNAME}__horizontal_well.csv`
  - 水平坑井の軌跡、ログ、正解 `TVT`、既知入力 `TVT_input`、地層面関連カラムを含む。
- `{WELLNAME}__typewell.csv`
  - 対応する垂直参照井戸。主に `TVT`、`GR`、`Geology` を含む。
- `{WELLNAME}.png`
  - 井戸軌跡や地質断面を確認するための可視化画像。

テストデータには水平坑井 CSV と Typewell CSV があり、隠し区間の正解 `TVT` は含まれない。可視の `test/` はサンプル用で、本番採点時には Kaggle 側で hidden test に差し替えられる。

ファイル数の確認結果は次の通り。

- `train` の horizontal well CSV: 773 件
- `train` の typewell CSV: 773 件
- `train` の PNG: 773 件
- visible `test` の horizontal well CSV: 3 件
- visible `test` の typewell CSV: 3 件
- `sample_submission.csv`: 1 件

## ノートブックの処理内容

### 1. セットアップ

`Path`、`numpy`、`pandas`、`matplotlib` などを読み込み、Kaggle Notebook とローカル展開済みデータの両方で動くように `DATA_ROOT` を探索している。`train/`、`test/`、`sample_submission.csv` がある場所をデータルートとして扱う。

### 2. タスク資料の確認

同梱の `AI_wellbore_geology_prediction_task_en.pptx` からテキストを抽出し、タスクの重要点を整理している。特に、水平坑井の `GR` を Typewell の `GR` と TVT スケール上で対応付ける考え方、近隣井戸の地質傾向を使える可能性、`TVT` が単調ではない点が強調されている。

### 3. サンプル井戸の読み込み

`000d7d20`、`00bbac68`、`00e12e8b` などをサンプルとして読み込み、水平坑井と Typewell の形状・カラム・先頭行を確認している。

例として `000d7d20` は次の構造を持つ。

- Horizontal shape: `(5278, 13)`
- Typewell shape: `(1296, 3)`
- Horizontal columns: `MD`, `X`, `Y`, `Z`, `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`, `TVT`, `GR`, `TVT_input`
- Typewell columns: `TVT`, `GR`, `Geology`

### 4. 学習 PNG と井戸ごとの可視化

学習 PNG を表示し、井戸パスと地質断面の雰囲気を確認している。さらに各サンプル井戸について、次の 3 つの観点を横並びで可視化している。

- `MD` に沿った `TVT` と `TVT_input`
- 水平坑井の `GR` ログ
- Typewell の `GR` と、既知区間の水平坑井 `GR` を TVT スケールに投影したもの

この可視化により、予測開始位置、既知区間の TVT トレンド、Typewell と水平坑井 GR の対応関係を把握できる。

### 5. `TVT_input` と予測開始位置

`TVT_input` は予測開始前だけ値があり、予測開始後は欠損する。ノートブックでは最初に `TVT_input` が欠損する行を `prediction_start_index` として扱い、そこから先を評価対象区間として整理している。

学習データでは真の `TVT` が全区間に存在するため、テストと同じ欠損状況を模した検証を作れる。

## 主な EDA 結果

### TVT の方向性

`dTVT / dMD` を使って、学習井戸の TVT 変化方向を確認している。結果は次の通り。

- nearly flat: 約 52.1%
- increasing: 約 32.5%
- decreasing: 約 15.4%

この結果から、`TVT` が常に `MD` とともに増えると仮定するモデルは危険である。地層の傾きや水平坑井の掘削方向により、同じ井戸内でも上昇・下降・ほぼ一定の動きが発生する。

### 学習井戸ごとの統計

全 773 件の学習水平坑井について、井戸単位の概要統計を作成している。

代表的な統計は次の通り。

- 1 井戸あたりの行数
  - 平均: 約 6,588 行
  - 中央値: 6,576 行
  - 最小: 2,058 行
  - 最大: 12,141 行
- `TVT_range`
  - 平均: 約 734 ft
  - 中央値: 約 758 ft
  - 最大: 約 1,257 ft
- `GR_missing_pct`
  - 平均: 約 29.4%
  - 中央値: 約 27.7%
  - 最大: 約 80.1%
- `TVT_input` 既知行数
  - 平均: 約 1,692 行
  - 中央値: 1,703 行
- 評価対象行数
  - 平均: 約 4,895 行
  - 中央値: 4,840 行
- 評価対象比率
  - 平均: 約 73.3%
  - 中央値: 約 74.0%

多くの井戸で、予測対象区間が全体の大部分を占める。

### 近隣井戸の位置関係

`X` と `Y` を使って井戸軌跡を地図上に描き、近隣井戸の文脈を確認している。タスク資料でも、近隣井戸の地質傾向や dip が予測に役立つ可能性が示されている。

### 欠損

水平坑井では `GR` に欠損が多く、井戸ごとに欠損率も大きく異なる。`TVT_input` の欠損は評価区間を表す意図的な欠損であり、通常の欠損値とは意味が違う。

学習データ全体の欠損確認では、地層面カラムの一部にもわずかな欠損が見られる。例として `ANCC` は約 0.9% の欠損がある一方、`BUDA` や `ASTNU` は確認範囲では欠損がない。

### 数値特徴量の相関

全行を一括で扱うのではなく、各井戸からサンプリングして数値特徴量の相関を確認している。サンプルは `(154600, 13)` の形状で、`MD`、`X`、`Y`、`Z`、地層面カラム、`TVT`、`GR`、`TVT_input` を含む。

ただし、井戸データは時系列・空間系列であり、行同士が独立同分布ではない。そのため、相関は大まかな関係を見るための参考情報として扱うべきである。

### Typewell の探索

Typewell は垂直参照ログで、水平坑井の `GR` を TVT スケールへ対応付ける基準になる。全 773 件の Typewell について概要統計を作成している。

主な結果は次の通り。

- 1 Typewell あたりの行数
  - 平均: 約 2,027 行
  - 中央値: 約 2,011 行
- `TVT_range`
  - 平均: 約 884 ft
  - 中央値: 約 907 ft
  - 最大: 1,399 ft
- `GR_range`
  - 平均: 約 174
  - 中央値: 約 176
  - 最大: 約 400
- `TVT` と `GR` は Typewell では欠損なし。
- `Geology` は平均で約 30.6% 欠損している。

Typewell の `Geology` ラベル頻度も集計しており、`ANCC`、`BUDA`、`ASTNU` などが主要ラベルとして確認されている。

## Sample Submission と Test

`sample_submission.csv` は visible test で `(14151, 4)` の形状になっている。`id` は `{well_id}_{row_index}` 形式で、どの井戸のどの行に対して `tvt` を提出するかを示す。

この構造により、テスト水平坑井の `TVT_input` が欠損している行と提出対象行を対応付けられる。

## モデリング上の示唆

ノートブックは、強い解法に向けた実践的な方針として次を挙げている。

- 井戸データは独立した行ではなく、`MD` に沿った系列として扱う。
- `TVT_input` の既知区間から、予測開始直前の局所的な TVT トレンドを推定する。
- 水平坑井の `GR` と Typewell の `GR` を TVT スケール上でアラインメントする。
- `X`、`Y`、`Z` と近隣井戸を使い、地質 dip や局所的な地質傾向を推定する。
- `GR` 欠損を明示的に扱う。欠損率そのものも井戸や区間の特徴になり得る。
- 検証はランダム行分割ではなく、井戸単位または時系列構造を尊重した方法にする。

## まとめ

この EDA は、ROGII コンペのデータ構造と予測課題の性質を把握するためのスターターである。特に重要なのは、`TVT_input` が予測開始位置を定義すること、`TVT` が単調とは限らないこと、水平坑井 `GR` と Typewell `GR` の対応付けが中心的な手掛かりになること、そして近隣井戸や空間座標が地質トレンド推定に役立つ可能性があること。

今後のモデル構築では、単純な行単位回帰よりも、系列アラインメント、空間特徴量、井戸単位検証を組み合わせる方針が有望と考えられる。

## 追加ノートブックとの差分

`rogii-eda-leakage-aware-submission-pipeline.ipynb` は、`eda-starter.ipynb` の EDA を発展させ、リーク管理を意識した検証・モデル・提出ファイル作成まで含むパイプラインになっている。以下は `eda-starter.ipynb` には明示的に含まれていなかった追加内容である。

### リーク境界と情報ポリシー

新しいノートブックでは、特徴量を **Strict drilling-time** と **Offline batch** の 2 系統に分けている。

- Strict drilling-time は、既知 prefix と現在行または過去行だけを使う保守的な検証設定。
- Offline batch は、Kaggle の提出時に利用できる full test CSV の target-free な tail 側 covariate も使う設定。
- tail の真の `TVT`、tail の `TVT` 由来統計、ランダム行分割、validation 対象井戸を含む近隣井戸参照などをリーク要因として明示している。
- `GroupKFold` による井戸単位分割を推奨し、同一井戸内の自己相関によるリークを避ける方針を取っている。

### `TVT_input` の整合性チェック

`eda-starter.ipynb` では `TVT_input` が予測開始位置を定義することを確認していたが、新しいノートブックではさらに、既知 prefix において `TVT_input` が `TVT` と完全一致するかを検査している。結果として、既知 prefix の `max_abs(TVT - TVT_input)` は `0.0` と確認されている。

これにより、`TVT_input` は prefix 内では正解 `TVT` のコピーとして安全に anchor や残差特徴量へ使える、という前提が明確になっている。

### last-known TVT anchor とベースライン評価

新しいノートブックでは、最後に既知だった `TVT_input`、つまり `last_known_TVT` を強い flat anchor として扱う。tail 区間の予測は、絶対 `TVT` を直接当てるよりも、`last_known_TVT` からの residual movement を推定する設計になっている。

主な追加結果は次の通り。

- 学習 tail 行数は 3,783,989 行。
- last-known constant baseline の row-level RMSE は約 15.9099。
- 井戸単位の `constant_tail_rmse` は平均約 12.81、中央値約 7.10、最大約 70.64。
- prefix 全体や直近 200 点からの単純な線形外挿も評価しているが、`MD` 方向の単純外挿は非常に不安定で、blind な slope extrapolation は危険と整理されている。

### より詳細な well-level 診断

`eda-starter.ipynb` の井戸単位統計に加えて、次のような tail 難易度や geometry を表す診断量を作っている。

- `known_rows`、`tail_rows`、`tail_frac`
- `last_known_tvt`
- `tail_end_delta_from_last_known`
- `tail_tvt_range`
- `tail_median_abs_step`
- `constant_tail_rmse`
- prediction start 付近の `X`、`Y`、`Z`
- `xy_span`、`z_delta`、`azimuth_deg`
- prefix と tail それぞれの `GR` 欠損率

ただし、`tail_rows` や `tail_tvt_range`、`constant_tail_rmse` などは学習時の診断用であり、Strict な drilling-time feature にはしない、と明示している。

### Typewell alignment の拡張

`eda-starter.ipynb` では Typewell の `GR` を水平坑井 `GR` と対応付ける重要性を説明していた。新しいノートブックでは、それを特徴量化する方向に進めている。

- prefix の `TVT_input` に基づき、Typewell の `GR` と水平坑井の `GR` を比較する。
- prefix だけで affine calibration を推定し、tail の真の `TVT` は使わない。
- Typewell 上の candidate TVT path、boundary phase、近傍 offset search、beam/DTW 的な path alignment を検討している。
- candidate-path typewell features は full tail の位置や GR texture を使うため、Strict ではなく Offline feature として扱っている。

### 特徴量テーブルと残差モデル

新しいノートブックは、EDA だけでなく row-level feature table を構築し、`target_delta_from_last_known = TVT - last_known_TVT` を予測する残差モデルを設計している。

特徴量群は大きく次に分かれる。

- prefix TVT の slope、range、step などの anchor/prefix 特徴量
- trajectory step、`MD`、`X`、`Y`、`Z` 由来の幾何特徴量
- prefix と tail の `GR` 欠損・分布・texture 特徴量
- Typewell alignment 特徴量
- candidate-path 特徴量
- formation-plane / KNN reference 由来の空間地質特徴量

モデル側では `HistGradientBoostingRegressor` による HGB residual pipeline を用意し、anchor-aware な shrinkage や clipping により、flat な井戸を壊しすぎないようにしている。

### GroupKFold CV と stored evidence

`eda-starter.ipynb` には実モデルの CV はなかったが、新しいノートブックでは `well_id` 単位の `GroupKFold` を使い、row-weighted RMSE で評価している。

保存済みの grouped-CV evidence として、次の値が示されている。

- Constant anchor: RMSE 約 15.9099
- Offline `offline_candidate_path_alignment`: RMSE 約 13.6172

この RMSE は HGB residual pipeline の local validation evidence であり、all-row LightGBM や外部 artifact stack のスコアを直接表すものではない、と注意書きされている。

### 近隣井戸と formation-plane の扱い

新しいノートブックでは、近隣井戸の空間情報をさらに進め、train-only の formation surface カラムを補助ラベルとして使う spatial reference model を検討している。

- `ANCC` などの formation top を、`X`、`Y` から推定する fold-safe imputer を構築する。
- grouped CV では validation wells を imputer fit から除外する。
- test に直接存在しない train-only surface をそのまま使うのではなく、target-free な座標から再現可能な reference feature として扱う。

この方針は、単なる地図可視化に留まっていた `eda-starter.ipynb` よりも、空間地質 prior をモデル特徴量へ落とし込む内容になっている。

### 提出ファイル作成パイプライン

`eda-starter.ipynb` では `sample_submission.csv` の構造確認までだったが、新しいノートブックは実際に複数の submission candidate を作成する。

主な候補は次の通り。

- `submission_constant_fallback_v2.csv`
- `submission_lgbm_raw_v2.csv`
- `submission_lgbm_mild_v2.csv`
- `submission_lgbm_hgb_policy_v2.csv`
- `submission_formation_plane_formula_private_safe_v2.csv`
- `submission_lgbm_plane_blend_private_safe_v2.csv`
- `submission_formation_plane_formula_public_aggressive_v2.csv`
- `submission_lgbm_plane_blend_public_aggressive_v2.csv`
- `submission_lgbm_plane6_blend_public_aggressive_v2.csv`
- `submission_lgbm_plane_rowancc_blend_public_aggressive_v2.csv`

LightGBM は memory-safe な compact feature set を使い、GPU 実行を基本としつつ、OpenCL が使えない場合の CPU fallback も考慮している。最終候補としては `plane_blend_public_aggressive` が設定されているが、public-aggressive は public score を狙う性質があるため、private-safe や mild candidate と比較して解釈する注意が書かれている。

### 外部 artifact stack blend

さらに、別ノートブックで作った重い physics/spatial branch の OOF residual predictions と test residual predictions を読み込み、OOF 上で blend weight を学習して test に適用する artifact stack branch も含まれる。

- OOF residual と `target_delta` で blend weight を学習する。
- test label は使わない。
- flat residual blend、branch-level blend、safe gate blend など複数の blend family を比較する。
- 利用できる artifact が弱い場合は、built-in LGBM/plane candidate へ fallback できるよう quality gate を設けている。

### 最終フォーマットガード

最後に、`submission.csv` が `sample_submission` と同じ `id,tvt` 形式であること、行数・ID・欠損・順序が正しいことを検証する final contract guard がある。`eda-starter.ipynb` の sample submission 確認よりも一段進んで、提出直前の安全確認まで自動化している。

### 差分のまとめ

`eda-starter.ipynb` は、データ理解、可視化、基本統計、Typewell の役割、モデリング方針を整理するスターター EDA である。一方、`rogii-eda-leakage-aware-submission-pipeline.ipynb` は、その EDA をもとに、リークを避けた検証設計、anchor-residual 型のモデリング、candidate-path / formation-plane / artifact stack の特徴量、複数 submission candidate、最終フォーマット検証までを含む実戦用パイプラインである。
