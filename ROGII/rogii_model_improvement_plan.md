---
name: ROGII Model Improvement
overview: "`ROGII/based.ipynb` を土台に、`ROGII/eda-starter.md` の示唆とコンペ概要を踏まえて、モデル選択・改善課題・前処理/後処理の方針を整理します。特に、last-known anchor、Typewell GR alignment、空間地質 prior、井戸単位 CV、リーク管理を中心に改善します。"
todos:
  - id: audit-validation
    content: 現行 `based.ipynb` の特徴量生成が fold-safe / private-safe か点検する
    status: completed
  - id: error-analysis
    content: OOF 残差を井戸単位で分析し、失敗パターンを分類する
    status: completed
  - id: feature-cleanup
    content: リーク疑い特徴・弱い特徴・高コスト特徴を整理する
    status: completed
  - id: adaptive-postprocess
    content: confidence に応じた後処理ブレンドと smoothing を設計する
    status: completed
  - id: branch-experiments
    content: 追加モデルブランチを OOF 比較できる形で試す
    status: completed
isProject: false
---

# ROGII モデル改善方針

## 前提

- 参照元: [ROGII/based.ipynb](ROGII/based.ipynb)、[ROGII/based.md](ROGII/based.md)、[ROGII/eda-starter.md](ROGII/eda-starter.md)
- 予測対象: `TVT_input` が欠損している tail 区間の `tvt`
- 評価: predicted `tvt` の RMSE
- 重要な制約: `TVT` は `MD` に対して単調ではなく、井戸単位の系列・Typewell GR・空間地質を同時に扱う必要がある

## 推奨モデル方針

### 1. 現行の residual tabular ensemble を主軸にする

`based.ipynb` の `last_known_tvt` anchor からの差分 `target = TVT - last_known_tvt` を予測する設計は維持する。

- 単体では `catboost-3` が最良: CV RMSE 10.4742
- `catboost-3` + `lightgbm-3` の hill climbing で 10.429
- `pf_ancc` を混ぜる後処理で 10.4046

この結果から、まずは CatBoost / LightGBM の gradient boosting ensemble を基準線にし、特徴量と validation の改善で伸ばすのが現実的。

### 2. 物理/地質モデルを直接提出モデルにせず、強い補助特徴量として使う

`based.ipynb` の particle filter、beam search、NCC、DTW、formation-plane は、それ単体モデルではなく GBDT に入れる候補パス特徴量として扱うのがよい。

- Typewell GR alignment: DTW / beam / NCC / affine calibration
- 空間地質 prior: formation-plane / dense ANCC imputation
- trajectory prior: `Z`, `dzdmd`, `dx`, `dy`, `dxy`, `azimuth`
- uncertainty: `pf_ancc_std`, `dtw_stoch_std`, `sig_std`, `beam_std_d`

### 3. 系列モデルは補助ブランチとして試す

行単位 GBDT は強いが、tail は系列なので、追加ブランチとして sequence model を検証する価値がある。

候補:

- 1D CNN / TCN: `GR`, `Z`, `MD`, candidate path deltas を tail sequence として入力
- LightGBM/CatBoost の井戸単位集約特徴を追加した GBDT
- Kalman / particle smoother 型の後処理

ただし最初から deep sequence model を主軸にするより、現行 GBDT の OOF に対する追加ブランチとして比較する。

## 改善すべき問題

### 1. CV と public LB の乖離リスク

現行は `GroupKFold(well)` で妥当だが、test は hidden wells に差し替わるため、近隣井戸や formation surface を使う特徴量は fold-safe にしないと CV が楽観的になる。

対応:

- formation imputer は validation well を除外して fit
- 近隣井戸特徴量も fold ごとに train wells のみで構築
- public-aggressive と private-safe の candidate を分ける

### 2. `GR` 欠損と tail 長のばらつき

`GR_missing_pct` は平均約 29.4%、最大約 80.1%。tail は平均 4,895 行、評価対象比率も平均 73.3% と大きい。

対応:

- `GR` 欠損率、連続欠損長、欠損 mask を特徴量化
- `GR` が弱い井戸では spatial / formation / anchor を強める gating
- tail 長や `md_since`, `frac` ごとの残差傾向を別々に確認

### 3. 単純 slope extrapolation の危険

EDA では `TVT` が increasing / decreasing / nearly flat を取り、単調仮定は危険。`based.md` でも naive linear extrapolation は不安定。

対応:

- slope は直接外挿ではなく、clipping / shrinkage した特徴量として使う
- flat anchor を壊さない後処理を維持
- 井戸ごとに drift confidence を推定し、動かす井戸と動かさない井戸を分ける

### 4. 後処理の過学習リスク

Optuna の `alpha`, `tau`, `w_pf` 後処理は OOF RMSE を改善しているが、global parameter だけでは井戸タイプ差を吸収しきれない可能性がある。

対応:

- 井戸単位 diagnostics で後処理効果を分解
- `GR` alignment confidence / `pf_ancc_std` / `sig_std` に応じた adaptive blending を試す
- smoothing window は tail 長・曲率・予測変動量で変える

## 必要な前処理

- `TVT_input` と `TVT` の prefix 一致を検証する。
- `TVT_input` の最初の欠損行を prediction start として厳密に管理する。
- `GR` は interpolation だけでなく、欠損 mask・欠損率・連続欠損長も保持する。
- Typewell は `TVT` 順に sort し、`GR` を平滑化・標準化して alignment に使う。
- `MD`, `X`, `Y`, `Z` から step、slope、azimuth、curvature を作る。
- formation-plane / dense ANCC は fold-safe に fit し、validation/test へ project する。
- train-only surface columns は直接 test feature とみなさず、補助ラベルとして空間参照モデルに使う。

## 必要な後処理

- `last_known_tvt` anchor からの差分を予測し、最後に `tvt = last_known_tvt + delta` へ戻す。
- prediction start 直後は `tau` 型の ramp-up で急なジャンプを抑える。
- `pf_ancc` / formation / DTW / model prediction を confidence に応じて混ぜる。
- 井戸ごとに Savitzky-Golay などで滑らかにするが、過度な平滑化で本当の地層変化を消さないよう tail 長・局所変動で調整する。
- `sample_submission.csv` に対して `id`、行数、順序、欠損、重複を最終確認する。

## 実装優先度

1. 現行 `based.ipynb` を private-safe / fold-safe に整理し、CV が信用できる形にする。
2. OOF を井戸単位で分析し、悪い井戸を `GR欠損`, `tail長`, `constant_tail_rmse`, `alignment confidence`, `spatial distance` で分類する。
3. CatBoost / LightGBM の特徴量重要度と SHAP で、効いている特徴・怪しいリーク特徴を分ける。
4. adaptive postprocess を追加し、global `w_pf` から confidence-based blend へ拡張する。
5. 追加ブランチとして TCN/1D CNN または physics-heavy artifact を OOF stack する。
6. 最終的に mild / private-safe / public-aggressive の submission を分けて比較する。

## 期待する方向性

現時点の最有力方針は、`based.ipynb` の CatBoost + LightGBM residual ensemble を核にし、Typewell alignment と formation/spatial prior を fold-safe に強化すること。深層系列モデルをいきなり主役にするより、まず OOF 上で補助ブランチとして評価し、GBDT ensemble に stack するのが堅実。
