# ROGII モデル改善メモ

このメモは `based.ipynb` を改善していくための実装方針である。`eda-starter.md` の EDA と Kaggle コンペの目的を踏まえ、現行の CatBoost / LightGBM residual ensemble を核にしつつ、CV の信頼性、特徴量の安全性、前処理・後処理、追加ブランチを整理する。

## 結論

最初に大きく変えるべきなのはモデル種類ではなく、validation と特徴量生成の安全性である。現行の `based.ipynb` は OOF RMSE が強く、`last_known_tvt` からの residual prediction、Typewell GR alignment、particle filter、DTW、formation-plane、CatBoost / LightGBM ensemble という方向性は妥当である。

一方で、formation / dense ANCC 系の spatial reference は `GroupKFold` の fold 外 validation に対して完全には fold-safe ではない可能性が高い。ここを整理しないまま特徴量や後処理を増やすと、CV と hidden test の乖離が大きくなる。

優先順位は次の通り。

1. fold-safe な特徴量生成に直す。
2. OOF 残差を井戸単位で分析する。
3. リーク疑い・低価値・高コスト特徴量を分類する。
4. confidence-based な後処理へ拡張する。
5. 追加モデルは OOF branch として stack する。

## 1. Validation 監査

### 現行で良い点

- 目的変数を `target = TVT - last_known_tvt` にしており、flat anchor を中心にした残差予測になっている。
- `GroupKFold(n_splits=5)` を使い、同じ井戸の行が train / valid にまたがらないようにしている。
- LightGBM / CatBoost の OOF 予測を保存し、hill climbing で OOF blend している。
- `sample_submission.csv` の `id` に合わせて最終 `submission.csv` を作っている。

### 問題になり得る点

`FormationPlaneKNN` と `DenseANCCImputer` は、ノートブック冒頭で全 training wells から構築されている。

```python
FI = FormationPlaneKNN(train_wids, CFG.dataset_path / "train")
DI = DenseANCCImputer(train_wids, CFG.dataset_path / "train")
```

`build_well()` 内では train の場合に `self_wid = wid` として同一井戸は除外している。

```python
swid = wid if is_train else None
form_ev, knn_d = _FI.impute(xy_ev, self_wid=swid)
d_ancc, d_std, d_dist = _DI.impute(xy_ev, self_wid=swid)
```

ただし、`GroupKFold` の validation fold に含まれる別井戸は除外されない。つまり validation fold の井戸 A の特徴量を作るとき、同じ validation fold の井戸 B の formation surface を spatial reference として参照できる。これは target `TVT` を直接使うリークではないが、hidden test で再現できる保証がない validation-only 情報を使うため、CV が楽観的になる可能性がある。

### 改善方針

fold ごとに reference builder を作り直す。

```text
for fold:
  train_wells = wells not in validation fold
  valid_wells = wells in validation fold
  FI_fold = FormationPlaneKNN(train_wells)
  DI_fold = DenseANCCImputer(train_wells)
  build train features with FI_fold / DI_fold
  build valid features with FI_fold / DI_fold
```

最終 test 用だけは全 train wells から reference を作ってよい。

この変更により、CV は少し悪化する可能性があるが、hidden test への信頼性は上がる。

## 2. OOF 残差分析

現時点のワークスペースには `models/*/oof_preds.pkl` や `train.csv` が見つからないため、実 OOF 残差の再集計はまだ実行していない。ノートブック出力から分かる範囲では、fold 間の RMSE 差がある。

例:

- `catboost-3`: 9.194 から 11.444
- `lightgbm-3`: 9.457 から 11.836
- `hill-climbing`: 10.429
- `hill-climbing (pp)`: 10.4046

fold 3 が相対的に難しい傾向があり、井戸ごとの地質変化、GR 欠損、tail 長、空間分布の偏りを確認する必要がある。

### 作るべき診断テーブル

OOF が利用できる状態になったら、1 行 1 井戸で次を集計する。

- `well`
- `n_tail`
- `rmse_model`
- `rmse_anchor`
- `rmse_pf_ancc`
- `rmse_dtw`
- `rmse_formation`
- `gain_vs_anchor`
- `gr_missing_tail_rate`
- `gr_missing_prefix_rate`
- `known_len`
- `eval_len`
- `ktvt_range`
- `tail_tvt_range`
- `constant_tail_rmse`
- `pfx_rmse`
- `spatial_knn_dist`
- `dense_rmse`
- `sig_std`
- `dtw_stoch_std`

### 失敗パターン分類

OOF 残差は次のカテゴリに分ける。

- `GR_missing_high`: tail の `GR` 欠損が多い井戸。
- `flat_anchor_better`: anchor の方がモデルより良い井戸。
- `drift_underfit`: tail の drift が大きく、モデルが動かし足りない井戸。
- `over_moved`: flat に近いのにモデルが動かしすぎた井戸。
- `alignment_bad`: `pfx_rmse` や DTW uncertainty が高い井戸。
- `spatial_far`: `spatial_knn_dist` が大きく、近隣井戸 prior が弱い井戸。
- `formation_mismatch`: formation / dense ANCC 系と実測 prefix の整合が悪い井戸。

この分類ができると、後処理やブレンドを井戸タイプごとに変えられる。

## 3. 特徴量整理

### 維持したい特徴量

現行で強い可能性が高く、モデルの主軸に残す。

- `last_known_tvt`
- `md_since`, `frac`, `sqrt_frac`
- `slp_all`, `slp_50`, `slp_z`
- `pf_ancc`, `pf_ancc_delta`, `pf_ancc_std`
- `beam_*_d`, `beam_mean_d`, `beam_std_d`, `beam_med_d`
- `sc*_d`, `sc*_sc`, `sc_ens_d`, `hyb_d`
- `dtw_ens_d`, `dtw_stoch_mean_d`, `dtw_stoch_std`, `dtw_cost_min`, `dtw_cost_range`
- `tvtF_*`, `form_mean_d`, `form_std_d`, `dense_*`
- `gr`, `gr_d1`, `gr_d2`, `grm*`, `grs*`, `glag*`, `glead*`
- `dx`, `dy`, `dz`, `dxy`, `dzdmd`, `dxdmd`, `dydmd`

### リーク疑いとして監査する特徴量

直接 target leak ではないが、CV の作り方によって楽観的になりやすい。

- `tvtF_*`
- `tvtFw_*`
- `tvtF50_*`
- `frm_rmse_*`
- `form_mean_d`
- `form_std_d`
- `form_rng_d`
- `dense_ancc`
- `tvt_dense_*`
- `dense_rmse`
- `dense_bias`
- `dense_nb_std`
- `spatial_knn_dist`
- `dense_dist`

これらは fold-safe builder に置き換えた上で、CV 悪化幅と hidden/public への安定性を確認する。

### 高コスト特徴量

計算時間が長く、ablation 対象にする。

- particle filter 系
- beam search 複数設定
- DTW multiscale
- stochastic DTW
- dense ANCC imputation

高コスト特徴量は「全て入りモデル」だけでなく、family ごとの OOF ablation を行う。

```text
base geometry + GR
base + PF
base + beam
base + NCC
base + DTW
base + formation
base + all
```

## 4. 前処理改善

### `TVT_input` と prediction start

- prefix で `TVT_input == TVT` を検証する。
- `TVT_input` の最初の欠損行を prediction start として保存する。
- 欠損が途中で途切れる井戸がないか確認する。

### `GR` 欠損

現行は `interpolate(limit_direction='both')` で補完しているが、欠損の情報も特徴量として残す。

追加したい特徴量:

- `gr_isna`
- `gr_missing_prefix_rate`
- `gr_missing_tail_rate`
- `gr_missing_run_len`
- `gr_missing_run_frac`
- `dist_to_nearest_valid_gr`
- `valid_gr_density_21`, `valid_gr_density_101`

`GR` 欠損が多い井戸では Typewell alignment より spatial / anchor を強くする gating に使う。

### Typewell 前処理

- `TVT` で sort する。
- 重複 `TVT` があれば集約する。
- `GR` を軽く平滑化した版と raw 版の両方を用意する。
- prefix の水平坑井 `GR` と Typewell `GR` で affine calibration を fit する。
- calibration の `a`, `b`, `pfx_rmse`, `pfx_mae`, correlation を特徴量にする。

### 空間特徴量

- `X`, `Y` から近隣井戸距離を作る。
- `azimuth`, `xy_span`, `z_delta`, local curvature を作る。
- formation-plane は fold-safe / final-test 用を明確に分ける。

## 5. 後処理改善

現行の後処理は global な `alpha`, `tau`, `w_pf` で、OOF 上では有効。

```text
delta = alpha * ramp(md_since, tau) * ((1 - w_pf) * model_delta + w_pf * pf_delta)
```

次はこれを adaptive にする。

### confidence-based blend

`w_pf` を固定値ではなく、井戸・行ごとの confidence から決める。

候補:

- `pf_ancc_std` が小さいほど PF を強める。
- `sig_std` が大きいほど model delta を縮める。
- `dtw_stoch_std` が小さいほど DTW / alignment 系を強める。
- `pfx_rmse` が高いほど Typewell alignment を弱める。
- `gr_missing_tail_rate` が高いほど GR-based path を弱める。
- `spatial_knn_dist` が小さいほど formation / dense ANCC を強める。

例:

```text
w_pf = sigmoid(a0
               - a1 * pf_ancc_std
               - a2 * pfx_rmse
               - a3 * gr_missing_tail_rate
               + a4 * spatial_confidence)
```

この gating は直接 LB で合わせず、OOF で学習または Optuna で最適化する。

### anchor-preserving clipping

flat な井戸を壊さないため、予測差分に井戸別 clip を入れる。

候補:

- `clip_abs = q95(|OOF target_delta|)` を type ごとに設定
- `sig_std` が大きい行ほど clip を強くする
- prediction start 直後は clip を強くし、tail 後半は緩める

### smoothing

現行は全井戸で Savitzky-Golay をかけている。改善案:

- tail 長に応じて window を変える。
- `GR` や candidate path が急変する箇所では smoothing を弱める。
- `pred_delta` と `pf_delta` の乖離が大きい箇所では smoothing 前後の差を制限する。

## 6. 追加モデルブランチ

### まず試すべき branch

1. `fold-safe GBDT`
   - 現行特徴量を fold-safe に作り直した CatBoost / LightGBM。

2. `private-safe spatial-light`
   - formation を fold-safe に限定し、public-aggressive な same-well 参照を避ける。

3. `GR sequence TCN`
   - 入力: `GR`, `Z`, `MD`, `pf_delta`, `beam_delta`, `dtw_delta`, mask
   - 出力: `target_delta` sequence
   - 使い方: 単独提出ではなく OOF stack の 1 branch。

4. `anchor classifier + regressor`
   - まず「動かすべき井戸か」を分類し、その後 delta magnitude を予測する。

5. `postprocess gating model`
   - OOF 上で `model_delta`, `pf_delta`, `dtw_delta`, `formation_delta` の blending weight を学習する。

### stack のルール

- 各 branch は OOF と test prediction を必ず出す。
- stack は OOF だけで重みを決める。
- test label や public score に合わせた weight 調整はしない。
- `mild`, `private-safe`, `public-aggressive` を分けて提出候補を作る。

## 7. 実装ロードマップ

### Phase 1: 信頼できる CV の再構築

- fold ごとに `FormationPlaneKNN` / `DenseANCCImputer` を作る。
- train / valid / test feature generation を分離する。
- 現行 CV 10.4046 と fold-safe CV の差を測る。

### Phase 2: OOF error dashboard

- 井戸単位 RMSE を出す。
- anchor / PF / DTW / model の勝敗を井戸ごとに見る。
- `GR` 欠損、tail 長、alignment confidence、spatial distance で失敗分類する。

### Phase 3: feature ablation

- family ごとに ablation。
- 高コスト特徴量の価値を測る。
- CatBoost / LightGBM の重要度を比較する。

### Phase 4: adaptive postprocess

- fixed `w_pf=0.07` から confidence-based blending へ。
- smoothing window を tail 長・uncertainty で変える。
- anchor-preserving clip を追加する。

### Phase 5: branch stacking

- TCN / smoother / gating model を OOF branch として追加。
- OOF stack で採用可否を決める。
- final submission guard を追加する。

## 最終方針

現時点で最も良い方針は、`based.ipynb` の CatBoost + LightGBM residual ensemble を主軸に維持し、物理・地質系の候補パスを特徴量として強化すること。ただし、formation / spatial 系は fold-safe に直すのが先である。

深層系列モデルや追加物理モデルは、現行 GBDT を置き換える主役ではなく、OOF stack に入れる branch として試す。改善の中心は「モデルを増やすこと」よりも、「どの井戸で何が信頼できるか」を推定して、anchor / PF / DTW / formation / GBDT を適切に混ぜることに置く。
