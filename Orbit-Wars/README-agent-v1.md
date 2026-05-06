# Agent v1 README

このドキュメントは、`agent-v1.py` の現在の挙動・前提条件・最近の変更内容を、現行コードに合わせて記録するためのものです。

## 目的

- 現在の意思決定ロジックを一元管理する。
- 実装済み機能と実験用機能の入口を明確にする。
- README とコードの説明差分を減らし、検証時の混乱を避ける。

## 現在の戦略

`agent-v1.py` は、優先順位付きのスコア戦略を実装しています。大きな優先順位は `インターセプト > 通常攻撃 > 補給` です。

- `obs` から `player`、`planets`、`fleets`、`initial_planets`、`angular_velocity`、`time_offset` を取得する。
- 惑星を `自勢力` と `非保有` に分け、さらに自惑星を `前線 / 後方` に簡易分類する。
- 飛行中艦隊は `infer_fleet_target_and_eta_with_offset()` で到着先と ETA を近似推定し、`friendly_arrivals_by_target` / `enemy_arrivals_by_target` を組み立てる。
- 非保有惑星ごとに、敵同士の占領変化直後を狙う `インターセプト窓` を作り、最優先で候補化する。
- 通常攻撃は単純な最寄り優先ではなく、`mission_score()` による複合評価で選ぶ。スコアには生産量、ETA、tempo、gateway、payback、局所クラスタ、ソースの柔軟性コストなどが入る。
- 通常攻撃では `wait_turns` を使った待機後発射も評価する。待機候補が即時候補より有利でも、局面によっては `reserved_source_ids` に回してそのターンの出撃を見送る。
- 後方惑星から前線惑星への補給は、通常攻撃より優先度を下げて行う。序盤の拡張圧が強いときは補給を抑制する。
- `planned_arrivals_by_target` を使い、同一ターン内にすでに予定した味方着弾も考慮して、過剰投入を減らす。
- 出撃余力は `effective_defense_margin()` で決める。基本は `DEFENSE_MARGIN = 10` だが、`expansion_pressure` が高い序盤は守備マージンを緩める。

## 予測と命中モデル

- 惑星位置予測は `predict_planet_position()` で行う。公転対象は `initial_planets` 上で「太陽からの距離 + 半径 < 50」を満たすものだけで、それ以外は静止扱いに近い。
- 将来角度は `theta = theta0 + omega * (step + future_turns + time_offset)` で計算する。
- `time_offset` は観測 `obs["time_offset"]` から受け取り、`board_state["time_offset"]` に保持する。`board_state` が無いときの `board_time_offset()` は `-1` を返す。
- `solve_launch_solution()` は現在のコードでは惑星中心 `(source.x, source.y)` からの直線射撃を前提にし、角度と飛行時間を 1 回で求める。
- `candidate_intercept_turns()` は粗い ETA を種にして、整数ターンの候補集合を周辺探索で作る。
- `estimate_precise_intercept()` は候補ターンごとに `predict_planet_position()` と `validate_intercept_solution()` を組み合わせ、`(eta, time, |eta - turn_hint|)` の順で最良解を選ぶ。
- `validate_intercept_solution()` は離散ターン検証を行う。各ターンの艦位置を直進で更新し、同じターンの `predict(..., turn - 1, ...)` と目標半径の距離で命中判定する。
- 太陽回避は `segment_hits_sun()` による線分判定で行う。太陽へ入る射撃や盤外へ出る射撃は無効とする。
- 現行メイン経路では `launch_point()`、`point_to_segment_distance()`、`ray_circle_hit_distance()` を使った sweep / 実発射位置ベースの照準は採用していない。

## 意思決定フロー

1. `obs` からプレイヤー・惑星・艦隊・軌道情報を読み取る。
2. 惑星配列を `Planet`、艦隊配列を `Fleet` に変換する。
3. `build_arrivals_by_target()` で飛行中艦隊の到着先と ETA を推定する。
4. `compute_expansion_pressure()` で序盤拡張圧を計算し、`board_state` に保存する。
5. 非保有惑星ごとに `build_intercept_windows()` でインターセプト窓を作る。
6. 自惑星を前線 / 後方に分類する。
7. まず各自惑星についてインターセプト候補を評価し、最良の 1 手を採用する。
8. 次に通常攻撃候補を評価する。
   - ターゲット候補数は `MAX_REGULAR_TARGETS` を基準にしつつ、序盤は `EARLY_PEACE_TARGETS` まで広げる。
   - 各ターゲットで `select_preferred_attack_plan()` が即時発射と待機発射を比べる。
   - 候補が有効なら `planned_arrivals_by_target` に追加し、同ターン内の不足艦数計算へ反映する。
9. 最後に後方惑星から前線惑星への補給候補を評価する。
10. 採用した命令を `[from_planet_id, angle, num_ships]` の形式で返す。

## パラメータと前提

- **防衛マージン**: 基本値は `DEFENSE_MARGIN = 10`。ただし `effective_defense_margin()` が `expansion_pressure` に応じて緩和する。
- **序盤拡張圧**: `EARLY_GAME_TURNS = 40` を基準に、ターン数・中立惑星数・敵到着数から `compute_expansion_pressure()` を計算する。
- **待機攻撃**: `MAX_WAIT_TURNS = 4`。生産のある出撃元だけが待機候補を持つ。
- **補給抑制**: `EARLY_PEACE_SUPPLY_STEP = 20` より早い段階で拡張圧が高い場合、補給は原則抑制する。
- **ターゲット候補数**: 通常攻撃の探索数は `MAX_REGULAR_TARGETS = 6` を基準に、序盤は `EARLY_PEACE_TARGETS = 10` まで増える。
- **公転予測**: `initial_planets` と `angular_velocity` を使い、`time_offset` を含む位相で将来位置を出す。
- **時間軸オフセット**: `nearest_planet_sniper()` は `obs["time_offset"]` を読み取る。実験用には `TIME_OFFSET_EXPERIMENTS = (-1, 0, 1)` と `compare_single_target_time_offsets()` がある。
- **速度モデル**: `estimate_fleet_speed()` は艦数の対数スケーリングを使い、`DEFAULT_MAX_SPEED = 6.0` まで上がる。
- **必要艦数**: `compute_attack_need()` は `到着時守備 + 1 - 先着味方艦隊` を基準にする。敵所有惑星には `production * eta` を加え、中立は増産なしとみなす。
- **命中検証モード**: `choose_validation_mode()` は存在するが、現状は常に `"strict"` を返す。
- **未使用の古い定数 / 関数**: `AIM_ITERATIONS` や `LAUNCH_CLEARANCE` は残っているが、現在の主要照準経路では直接効いていない。

## 既知の制約

- `time_offset` の最適値はまだ確定しておらず、環境との位相ズレが残っている可能性がある。
- 命中モデルは惑星中心からの直線発射 + 離散ターン検証であり、実エンジンのスポーン位置や衝突順序と完全一致している保証はない。
- 飛行中艦隊の到着先推定は近似であり、完全な解析解ではない。
- 到着前の多者戦闘、敵増援、途中削りまで含めた完全な将来戦闘シミュレーションは行っていない。
- インターセプト窓は敵同士の占領変化を簡易に検出したもので、多者戦の最適解ではない。
- 補給ロジックの前線判定、必要補給量、待機攻撃の採否はヒューリスティックであり、全局面最適を保証しない。
- `strict / relaxed` の切り替えを前提にした旧設計の名残はあるが、現状の採用判定は `strict` 相当のみで動いている。
- `launch_point()` や sweep 関連ヘルパーはファイル内に残っているが、現行メイン経路では使っていない。
- 太陽回避は単純な交差判定のみで、迂回経路の探索は行っていない。

## 更新ログ

変更ごとに 1 エントリを追加し、新しいものを上に記載します。

### 2026-05-06 - README を現行実装へ同期

- 古い sweep 命中、実発射位置、連続時間反復エイム前提の説明を整理し、現行の中心発射 + 離散ターン検証モデルへ合わせた。
- `time_offset`、`board_time_offset()`、`compare_single_target_time_offsets()` など、最近追加した時間軸オフセット実験の入口を追記した。
- `strict / relaxed` 切り替えや未使用ヘルパーについて、現状のコードとの関係が分かるように説明を修正した。

### 2026-05-06 - 時間軸オフセット実験を追加

- `predict_planet_position()` に `time_offset` を流し、予測位相を `-1 / 0 / +1` で比較できるようにした。
- 飛行中艦隊の到着推定、通常攻撃、補給、距離評価にも同じオフセットが流れるように揃えた。
- 単一ターゲットの比較用に `compare_single_target_time_offsets()` を追加した。

### 2026-05-06 - 中心発射ベースの離散命中検証へ統一

- `solve_launch_solution()` を惑星中心からの直線射撃に簡略化した。
- `validate_intercept_solution()` と `infer_fleet_target_and_eta_with_offset()` を、`turn - 1` 基準の離散ターン判定へ揃えた。
- sweep 命中や実発射位置ベースの旧経路は、現行の主要判断から外した。

### 2026-05-06 - 序盤拡張圧と待機攻撃を追加

- `compute_expansion_pressure()` を導入し、序盤 40 ターンの拡張志向を連続値で扱うようにした。
- `effective_defense_margin()`、通常攻撃スコア、補給抑制、ターゲット数上限を拡張圧に応じて変えるようにした。
- `wait_turns` と `reserved_source_ids` により、即時発射と待機発射を同じ経路で比較できるようにした。

### 2026-04-22 - 初期の ROI / 迎撃 / 補給方針を導入

- 行動優先順位を `インターセプト > 通常攻撃 > 補給` とした。
- 到着時守備艦数、先着味方艦隊、太陽回避を考慮した候補評価を追加した。
- `initial_planets` と `angular_velocity` を使う公転予測の土台を入れた。

## 次の改善案

- `time_offset` の最適値を実験で確定し、通常運用の既定値を決める。
- 高価値惑星に対する複数出撃元の協調攻撃を実装する。
- 敵増援や途中戦闘を含めた将来戦力予測を強化する。
- 補給ネットワークを多段転送や需要予測まで拡張する。
- 太陽回避を二択のキャンセルだけでなく、代替目標や経路選択まで広げる。

## 更新チェックリスト

`agent-v1.py` を更新した際は、この README も必ず同期します。

1. 意思決定ロジックを変更した場合は `現在の戦略` と `意思決定フロー` を更新する。
2. 定数やしきい値を変更した場合は `パラメータと前提` を更新する。
3. 実装上の制約や未使用ヘルパーの扱いが変わった場合は `既知の制約` を見直す。
4. 大きな変更を加えた場合は `更新ログ` に要約を追加する。
