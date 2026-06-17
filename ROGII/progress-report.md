# ROGII 進捗報告資料

## 1. 目的

ROGII Wellbore Geology Prediction では、水平坑井の未知区間に対する `TVT` を予測する。  
今回の実験では、`based.ipynb` を baseline として、モデル本体・stacking・後処理を段階的に変更し、Kaggle Public LB の RMSE がどう変化するかを確認した。

中心にした予測設計は、絶対 `TVT` を直接予測するのではなく、最後に既知だった `TVT_input` を anchor として、そこからの差分を予測する residual prediction である。

```text
target_delta = TVT - last_known_tvt
prediction  = last_known_tvt + predicted_delta
```

## 2. 実験結果

| 実験 | Public LB RMSE | baseline 差分 | 評価 |
|---|---:|---:|---|
| Confidence w_pf Only | 9.839 | -0.275 | 最大改善。ただし再実行で揺れあり |
| Confidence w_pf Fixed Reproduce | 9.932 | -0.182 | 固定 params で再現できた安定候補 |
| Adaptive Postprocess Only | 9.984 | -0.130 | 後処理の自由度追加が有効 |
| Adaptive Smoothing Only | 10.011 | -0.103 | smoothing 単独でも小幅改善 |
| Confidence w_pf Only | 10.046 | -0.068 | confidence weighting は概ね有効 |
| baseline | 10.114 | 0.000 | 比較基準 |
| Confidence w_pf Fixed Params | 10.112 | -0.002 | Kaggle version 差の疑い |
| Confidence w_pf 再実行 | 10.137 | +0.023 | Optuna 探索のブレを確認 |
| v1 fold-safe/adaptive 全部入り | 10.212 | +0.098 | 一括投入は悪化 |
| Ridge Stack Only | 10.275 | +0.161 | stack 単体は悪化 |

## 3. 何が効いたか

### 最も効果があったもの: confidence-based w_pf

固定の `w_pf` ではなく、井戸・行ごとの信頼度に応じて `model_delta` と `pf_delta` の混ぜ方を変える方針が最も良い結果を出した。

使った主な confidence 指標:

- `pf_ancc_std`: particle filter の不確実性
- `pfx_rmse`: prefix 区間での Typewell alignment の悪さ
- `gr_missing_scale`: `GR` 欠損の影響
- `spatial_scale`: 空間 prior の信頼度

9.839 を出した run の固定パラメータ:

```python
best_pp_params = {
    "alpha": 0.99,
    "tau": 115,
    "w_pf": 0.02664085453513666,
    "pf_std_scale": 0.001022713827327233,
    "pfx_rmse_scale": 0.3576194933500251,
    "gr_missing_scale": 1.4527449645308703,
    "spatial_scale": 0.48385285948248385,
}
```

この設定は `Confidence w_pf Fixed Reproduce` で 9.932 まで再現できており、現時点で最も有望な安定候補である。

### 有効だったもの: adaptive postprocess

`Adaptive Postprocess Only` は 9.984 まで改善した。  
これは、単純な固定 blend よりも、後処理側で井戸や行ごとの状態を見た方が Public LB に効くことを示している。

ただし、全部入り v1 は 10.212 まで悪化したため、adaptive postprocess は単独では有効でも、他の変更と混ぜると悪化する可能性がある。

### 小幅に有効だったもの: adaptive smoothing

`Adaptive Smoothing Only` は 10.011 で、baseline より 0.103 改善した。  
井戸ごとに予測系列を滑らかにする処理は有効だが、confidence-based w_pf と組み合わせた時にさらに伸びるかは追加検証が必要である。

## 4. 効果が薄かった、または悪化したもの

### Ridge Stack Only

`Ridge Stack Only` は 10.275 で悪化した。  
LightGBM / CatBoost の branch を Ridge で混ぜるだけでは、Public LB では改善しなかった。

この結果から、今回の改善余地はモデル本体の stack よりも、`last_known_tvt` anchor に対する後処理側にあると考えられる。

### v1 全部入り

`v1 fold-safe/adaptive 全部入り` は 10.212 で悪化した。  
fold-safe 特徴量、adaptive postprocess、clipping、adaptive smoothing などを一括で入れると、個別には効く要素があっても相互作用で悪化する可能性がある。

そのため、今後も一括投入ではなく ablation で一つずつ足す方針がよい。

## 5. 現時点の結論

今回の実験から言えることは次の通り。

1. スコア改善には、モデル本体より後処理が効いている。
2. 特に `confidence-based w_pf` が最も有望。
3. `adaptive postprocess` と `adaptive smoothing` も改善効果がある。
4. `Ridge stack only` と全部入り v1 は悪化した。
5. Optuna の探索結果は Public LB 上で大きくブレるため、良かった run の params を固定する必要がある。

## 6. 次にやるべきこと

### 優先 1: `Fixed Reproduce` を正本にする

`Confidence w_pf Fixed Reproduce` は 9.932 で、固定パラメータ版としては最も良い。  
今後はこれを次の比較基準にする。

### 優先 2: `Fixed Reproduce` と `Fixed Params` の Kaggle version 差を確認

ローカル上では実装差はほぼなく、差分は `VARIANT_SCORE_NAME` 程度だった。  
それにもかかわらず Kaggle では 9.932 と 10.112 の差が出たため、古い notebook version を提出した可能性が高い。

確認すること:

- Kaggle 上で `study.optimize` が走っていないか
- 固定 params がログに出ているか
- `submission.csv` がモデル予測で上書きされているか

### 優先 3: 固定 params で追加 ablation を試す

作成済みの固定提出候補:

- `Rogii-confidence-wpf-fixed-reproduce.ipynb`
- `Rogii-confidence-wpf-fixed-adaptive-smoothing.ipynb`
- `Rogii-confidence-wpf-fixed-no-clip.ipynb`
- `Rogii-adaptive-postprocess-fixed-no-clip.ipynb`

まずは `fixed-reproduce` を正本として再提出し、その後 `fixed-adaptive-smoothing` を比較する。

## 7. 報告会での要点

報告では次の順で説明すると分かりやすい。

1. baseline は 10.114。
2. stack 単体や全部入り v1 は悪化した。
3. 後処理系は明確に改善した。
4. 最も良かったのは `confidence-based w_pf` の 9.839。
5. ただし Optuna のブレがあり、固定 params 版では 9.932 まで再現。
6. 次は `Fixed Reproduce` を正本にして、smoothing や no-clip 版を比較する。

## 8. まとめ

今回の改善で最も重要だった発見は、`TVT` 予測ではモデル出力をそのまま使うより、井戸・行ごとの信頼度に応じて `pf_ancc` や model delta を調整する後処理が効くという点である。

今後は新しいモデルを増やすより、`confidence-based w_pf` を安定再現できる形に固定し、その上で smoothing や clipping の有無を小さく比較していく。
