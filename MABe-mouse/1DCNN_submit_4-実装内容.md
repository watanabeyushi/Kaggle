# 1DCNN_submit_4.py 実装内容

## 概要

`1DCNN_submit_4.py`は、MABe Mouse Behavior Detectionコンペティション用の1D CNNモデルの**推論専用**スクリプトです。学習済みモデルを読み込んで予測を行い、提出ファイルを生成します。

学習は`1DCNN_model_4.py`で実行してください。このスクリプトは推論のみを行い、実行時間を短縮します。

## 主な特徴

1. **推論専用**: 学習機能は含まれていません
2. **モデル読み込み**: 学習済みモデルと前処理器を自動的に読み込みます
3. **Kaggle最適化**: Kaggle環境での実行を前提としています
4. **エラーハンドリング**: 詳細なエラーメッセージとデバッグ情報を提供します
5. **実行状況表示**: 処理の進行状況を詳細に表示します

## ファイル構成

### 主要関数

#### `prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH)`
- データフレームから1D CNN用のシーケンスデータを準備する関数
- 数値カラムのみを抽出し、シーケンスを作成
- データが少ない場合はパディングを実行

#### `find_tracking_file(data_dir, video_id, annotation_file_path=None, debug=False)`
- 追跡ファイルを検索する関数
- 再帰的検索と柔軟なマッチングに対応
- CSVとParquetファイルの両方をサポート

#### `load_model_and_preprocessors(model_dir=MODEL_DIR)`
- 学習済みモデルと前処理器を読み込む関数
- モデルファイル（`.h5`または`.keras`）を検索
- StandardScaler（`scaler.pkl`）を読み込み
- LabelEncoder（`label_encoder.pkl`）を読み込み
- `/kaggle/input`からも自動的に検索

#### `predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length=SEQUENCE_LENGTH)`
- 1D CNNモデルでテストデータを予測する関数
- 各テスト行に対して追跡ファイルを検索
- シーケンスを作成し、正規化して予測
- 複数シーケンスがある場合は平均を取る

#### `create_submission(predictions, test_metadata, label_encoder, output_path=None)`
- 提出ファイルを作成する関数
- 参考ノートブックの形式に合わせる
- `sample_submission.csv`の形式を確認して合わせる
- Kaggle環境では`/kaggle/working/submission.csv`に保存

#### `main()`
- メイン実行関数
- モデルと前処理器の読み込み、テストデータの読み込み、予測の実行、提出ファイルの生成を実行

## パス設定

参考ノートブック（https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost）の方式に合わせて実装されています。

### Kaggle環境
- データディレクトリ: `/kaggle/input/`配下を自動検出
- 出力ディレクトリ: `/kaggle/working`
- モデル読み込みディレクトリ: `/kaggle/working`（学習済みモデルが保存されている場所）

### ローカル環境
- データディレクトリ: `./data`
- 出力ディレクトリ: `./output`
- モデル読み込みディレクトリ: `./models`

## 必要なファイル

推論を実行するには、以下のファイルが必要です：

1. **モデルファイル**: `.h5`または`.keras`形式
   - 例: `best_1dcnn_model_4.h5`
   - `/kaggle/working`または`/kaggle/input`配下に配置

2. **StandardScaler**: `scaler.pkl`
   - 学習時に使用したStandardScalerを保存したファイル
   - モデルファイルと同じディレクトリに配置

3. **LabelEncoder**: `label_encoder.pkl`
   - 学習時に使用したLabelEncoderを保存したファイル
   - モデルファイルと同じディレクトリに配置

4. **テストデータ**: `test.csv`
   - データディレクトリに配置

## 実行フロー

1. **モデルと前処理器の読み込み**
   - モデルファイル（`.h5`または`.keras`）を検索して読み込み
   - StandardScaler（`scaler.pkl`）を読み込み
   - LabelEncoder（`label_encoder.pkl`）を読み込み
   - `/kaggle/input`からも自動的に検索

2. **テストデータの読み込み**
   - `test.csv`を読み込む
   - 代替パスも試行（`test/test.csv`、`data/test.csv`など）

3. **予測の実行**
   - 各テスト行に対して追跡ファイルを検索
   - シーケンスを作成し、正規化して予測
   - 複数シーケンスがある場合は平均を取る

4. **提出ファイルの生成**
   - 予測結果から提出ファイルを作成
   - `sample_submission.csv`の形式に合わせる
   - `/kaggle/working/submission.csv`に保存

## 出力ファイル

### `submission.csv`
予測結果が保存されます。Kaggle環境では`/kaggle/working/submission.csv`に保存されます。

形式:
```csv
row_id,action
0,rear
1,chase
2,attack
...
```

## エラーハンドリング

- モデルファイルが見つからない場合の詳細なエラーメッセージ
- 前処理器が見つからない場合の警告（デフォルト値を使用）
- 追跡ファイルが見つからない場合のデフォルト予測
- デバッグモードでの詳細なログ出力

## モデルファイルの検索順序

1. `MODEL_DIR`（通常は`/kaggle/working`）配下の`.h5`または`.keras`ファイル
2. `/kaggle/input`配下の各ディレクトリ内の`.h5`または`.keras`ファイル

前処理器（`scaler.pkl`、`label_encoder.pkl`）も同様の順序で検索されます。

## 注意事項

- このスクリプトは推論専用です。学習は`1DCNN_model_4.py`で実行してください
- 学習済みモデルと前処理器が必要です
- Kaggle環境での実行を前提としていますが、ローカル環境でも動作します
- モデルファイルと前処理器は同じディレクトリに配置することを推奨します

## 実行時間の短縮

推論専用スクリプトとして実装されているため、学習処理を含まないため実行時間が短縮されます。Kaggle環境では、学習済みモデルを読み込むだけで推論を実行できます。

## 参考

- Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
- 参考ノートブック: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
- 既存実装: `1DCNN_model_3.py`
- 学習用スクリプト: `1DCNN_model_4.py`

