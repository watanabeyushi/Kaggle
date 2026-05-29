# based.ipynb 要約

`based.ipynb` は、ROGII Wellbore Geology Prediction 向けの実戦的な予測パイプラインである。単なる EDA ではなく、特徴量生成、外部 artifact の再利用、LightGBM / CatBoost 学習、Ridge stacking によるアンサンブル、Optuna 後処理、`submission.csv` 作成までを一通り実行する。

予測対象は、各水平坑井で `TVT_input` が欠損している tail 区間の `TVT` である。モデルは絶対 `TVT` を直接予測するのではなく、最後に既知だった `TVT_input`、つまり `last_known_tvt` からの差分 `target = TVT - last_known_tvt` を学習する。

## 1. Imports and configs

主なライブラリは次の通り。

- `sklearn`
  - `GroupKFold`
  - `root_mean_squared_error`
  - `Ridge`
- `lightgbm`
- `CatBoostRegressor`
- `scipy`
  - `cKDTree`
  - `savgol_filter`
- `numba`
- `joblib`
- `optuna`

設定クラス `CFG` では、Kaggle competition data と外部 artifact dataset のパス、seed、5-fold の `GroupKFold`、評価指標 RMSE を定義している。

```text
dataset_path = /kaggle/input/competitions/rogii-wellbore-geology-prediction
artifacts_path = /kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts
n_splits = 5
```

## 2. Data loading and preprocessing

この章がノートブックの大部分を占め、水平坑井と Typewell から大量の特徴量を作る。

### 主要な考え方

- `TVT_input` が存在する prefix 区間を既知区間として使う。
- `TVT_input` が欠損する tail 区間だけを学習・予測対象にする。
- 学習時の目的変数は `TVT - last_known_tvt`。
- 推論時は、予測差分を `last_known_tvt` に足し戻して `tvt` にする。

### 生成される主な特徴量

#### Particle filter 系

`run_pf_ancc` と `run_pf_z` により、Typewell の `GR`、水平坑井の `GR`、`Z`、既知 TVT を使って TVT の候補パスを推定する。

主な特徴量:

- `pf_ancc`
- `pf_ancc_std`
- `pf_ancc_delta`
- `pf_z`
- `pf_z_delta`
- `pf_vs_z`

`pf_ancc` は後処理でも利用され、最終予測を補正する重要な補助信号になっている。

#### Beam search 系

複数の beam 設定で Typewell GR と水平坑井 GR の整合パスを探索する。

設定例:

- `cons`
- `loose`
- `vcons`
- `sm5`
- `vloose`
- `mid`
- `stiff`

主な特徴量:

- `beam_{tag}_d`
- `beam_mean_d`
- `beam_std_d`
- `beam_med_d`

`cons` と `sm5` の平均は `beam_ref` として使われる。

#### Multi-scale NCC 系

既知 prefix の水平坑井 `GR` と `TVT_input` を使い、複数 window 幅で normalized cross-correlation を計算する。

使用 window:

- 8
- 15
- 25

主な特徴量:

- `sc8_d`
- `sc15_d`
- `sc25_d`
- `sc_cons_d`
- `sc_ens_d`
- `sc_trust`
- `hyb_d`

`sc_trust` は既知 prefix の長さに応じて決まり、beam 系と NCC 系を混ぜた `hyb_ref` に使われる。

#### DTW 系

Typewell GR と水平坑井 GR を Dynamic Time Warping で対応付ける。Sakoe-Chiba band 付きの constrained DTW と、Gumbel noise を加える stochastic DTW がある。

使用 radius:

- 20
- 50
- 100
- 200

主な特徴量:

- `dtw_ens_d`
- `dtw_stoch_mean_d`
- `dtw_stoch_std`
- `dtw_stoch_cv`
- `dtw_slope_mean`
- `dtw_r{radius}_d`
- `dtw_slope_r{radius}`
- `dtw_cost_min`
- `dtw_cost_range`
- `dtw_vs_beam`
- `dtw_vs_pf`
- `dtw_vs_sc`

DTW は Typewell 上の GR signature と水平坑井 GR の対応を直接モデル化するため、地質的な位置推定に強い信号として扱われている。

#### Formation / spatial 系

学習井戸の formation surface カラムを使って、空間的な地質面を補間する。

対象 formation:

- `ANCC`
- `ASTNU`
- `ASTNL`
- `EGFDU`
- `EGFDL`
- `BUDA`

`FormationPlaneKNN` は井戸単位の代表点から formation 面を KNN + 局所平面で推定する。`DenseANCCImputer` は `ANCC` をより細かい空間サンプルから補間する。

主な特徴量:

- `tvtF_{formation}`
- `tvtFw_{formation}`
- `tvtF50_{formation}`
- `frm_rmse_{formation}`
- `form_mean_d`
- `form_std_d`
- `form_rng_d`
- `dense_ancc`
- `dense_std`
- `dense_dist`
- `tvt_dense_d`
- `tvt_densew_d`
- `tvt_dense50_d`
- `dense_rmse`
- `dense_bias`
- `dense_nb_std`

#### GR texture / trajectory 系

水平坑井の tail 区間に対して、位置・勾配・GR の局所統計を作る。

主な特徴量:

- `md_since`
- `frac`
- `frac2`
- `sqrt_frac`
- `z`
- `dx`
- `dy`
- `dz`
- `dxy`
- `dzdmd`
- `dxdmd`
- `dydmd`
- `gr`
- `gr_d1`
- `gr_d2`
- `gr_env`
- `gr_nrg`
- `grm5`, `grm21`, `grm51`, `grm101`
- `grs5`, `grs21`, `grs51`, `grs101`
- `glag1`, `glag5`, `glag15`, `glag30`
- `glead1`, `glead5`, `glead15`, `glead30`

### データセット構築

`build_well()` が 1 井戸単位で特徴量を作り、`build_dataset()` が `joblib.Parallel` で全井戸を並列処理する。

学習データは外部 artifact に `data/train.csv` が存在すればそれを読み込み、存在しなければ `train/` から作成して `train.csv` に保存する。テストデータは `test/` から毎回作成する。

最終的に次のように学習行列を作る。

```python
features = [c for c in train_df.columns if c not in {'well', 'id', 'target'}]
X = train_df[features]
y = train_df['target']
g = train_df['well']
X_test = test_df[features]
```

## 3. Training

学習は `GroupKFold(n_splits=5)` を使い、`well` 単位で fold を分ける。これにより、同じ井戸の行が train と valid にまたがるリークを避けている。

### LightGBM

3 つの LightGBM モデルを学習または外部 artifact から読み込む。GPU を使う設定で、early stopping は 250 round。

LightGBM の CV 結果:

- `lightgbm-1`
  - Fold RMSE: 9.805, 11.186, 9.340, 12.158, 11.238
  - Overall RMSE: 10.7947
- `lightgbm-2`
  - Fold RMSE: 9.869, 11.196, 9.323, 11.911, 11.282
  - Overall RMSE: 10.7595
- `lightgbm-3`
  - Fold RMSE: 9.827, 11.091, 9.457, 11.836, 11.325
  - Overall RMSE: 10.7457

### CatBoost

3 つの CatBoost モデルを学習または外部 artifact から読み込む。GPU 設定で、`od_wait=300` の overfitting detector を使う。

CatBoost の CV 結果:

- `catboost-1`
  - Fold RMSE: 10.061, 10.674, 9.206, 11.613, 11.238
  - Overall RMSE: 10.5930
- `catboost-2`
  - Fold RMSE: 10.037, 10.670, 9.237, 11.628, 11.184
  - Overall RMSE: 10.5849
- `catboost-3`
  - Fold RMSE: 10.014, 10.533, 9.194, 11.444, 11.038
  - Overall RMSE: 10.4742

単体モデルでは `catboost-3` が最良である。

## 4. Ridge stacking

LightGBM 3 本と CatBoost 3 本の OOF 予測を `sklearn.linear_model.Ridge` でブレンドする。Kaggle 標準環境にない `hill_climbing` 依存を避けるため、`positive=True` により非負係数の stacking として扱う。

元の hill climbing では次のような結果だった。

- 対象モデル数: 6
- 開始モデル: best
- 最初の best: `catboost-3`、RMSE 10.474
- 追加されたモデル: `lightgbm-3`、weight 0.272
- 最終アンサンブル数: 2
- Final score: 10.429
- 改善幅: +0.045、約 +0.43%

修正版では Ridge の OOF 予測を `hc_oof_preds`、テスト予測を `hc_test_preds` として保持し、後続の後処理コードはそのまま使う。

## 5. Postprocessing

後処理では、Ridge stacking の予測差分と `pf_ancc` 由来の差分を混ぜる。

```python
d = md * (1 - w_pf) + pd_ * w_pf
```

ここで:

- `md`: Ridge stacking 予測差分
- `pd_`: `pf_ancc - last_known_tvt`
- `w_pf`: particle filter 側の重み

さらに `tau` を使い、予測開始直後の差分を滑らかに立ち上げる。

```python
d *= (1 - exp(-md_since / tau))
```

Optuna で `alpha`、`tau`、`w_pf` を最適化している。

最良値:

- RMSE: 10.40456136218748
- `alpha`: 1.0
- `tau`: 60
- `w_pf`: 0.07

元の notebook 出力では、この後処理により hill-climbing の 10.429 から 10.4046 まで改善していた。修正版では Ridge stacking の OOF 結果に対して同じ後処理を最適化する。

また、井戸ごとに `savgol_filter` を使って最終予測 `pred` を平滑化する。

## 6. Inference

テストデータに対して、次の順序で推論する。

1. `hc_test_preds` を取得する。
2. `pf_test = pf_ancc - last_known_tvt` を計算する。
3. Optuna で得た `alpha`、`tau`、`w_pf` を使って補正差分を計算する。
4. `last_known_tvt` に補正差分を足して `pred` を作る。
5. 井戸ごとに Savitzky-Golay 平滑化を適用する。
6. `sample_submission.csv` の `id` に合わせて `tvt` を merge する。
7. 欠損が残った場合は、`train_df['last_known_tvt'].mean() + train_df['target'].mean()` で埋める。
8. `submission.csv` を保存する。

出力された submission は 14,151 行、2 列の `id`, `tvt` 形式である。

例:

- `000d7d20_1442`: 11747.366412
- `000d7d20_1443`: 11747.370182
- `00e12e8b_6383`: 11596.554635

## 7. Results

最後に、各モデルとブレンドの fold RMSE および overall RMSE を可視化している。

全体の順位感は次の通り。

1. `ridge-stack (pp)`: Ridge stacking に後処理を適用した結果
2. `ridge-stack`: Ridge stacking の結果
3. `catboost-3`: 10.4742
4. `catboost-2`: 10.5849
5. `catboost-1`: 10.5930
6. `lightgbm-3`: 10.7457
7. `lightgbm-2`: 10.7595
8. `lightgbm-1`: 10.7947

## まとめ

`based.ipynb` は、`last_known_tvt` を anchor とした residual prediction を中心に、Typewell GR alignment、particle filter、beam search、NCC、DTW、formation surface 補間、GR texture、trajectory を大量に特徴量化する高密度なモデルノートブックである。

単体モデルでは CatBoost が LightGBM より強く、特に `catboost-3` が最良だった。修正版では Kaggle 環境で動かすため、外部の `hill_climbing` パッケージではなく `Ridge` でブレンドし、Optuna による particle-filter 混合の後処理を加える構成にしている。

最終成果物は `submission.csv` で、`sample_submission.csv` に合わせた `id,tvt` 形式の提出ファイルになっている。
