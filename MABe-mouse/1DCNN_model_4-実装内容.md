# 1DCNN_model_4.py 実装内容

## 概要

`1DCNN_model_4.py`は、MABe Mouse Behavior Detectionコンペティション用の1D CNNモデルの**学習専用**スクリプトです。推論は`1DCNN_submit_4.py`で実行します。

このスクリプトは、データアクセスの確認を主目的としており、学習実装は含まれていません。Kaggle環境で実行可能な形で実装されています。

## 主な特徴

1. **学習専用**: 推論機能は含まれていません
2. **データアクセス確認**: 学習前にデータへのアクセスを検証します
3. **Kaggle最適化**: Kaggle環境での実行を前提としています
4. **エラーハンドリング**: 詳細なエラーメッセージとデバッグ情報を提供します
5. **実行状況表示**: 処理の進行状況を詳細に表示します

## ファイル構成

### 主要関数

#### `find_tracking_file(data_dir, video_id, annotation_file_path=None, debug=False)`
- 追跡ファイルを検索する関数
- 再帰的検索と柔軟なマッチングに対応
- CSVとParquetファイルの両方をサポート
- アノテーションファイルのパスから同じサブディレクトリを優先的に検索

#### `find_annotation_file(data_dir, video_id, debug=False)`
- アノテーションファイルを検索する関数
- 再帰的検索と柔軟なマッチングに対応
- CSVとParquetファイルの両方をサポート

#### `normalize_annotation_columns(df, lab_id=None, debug=False)`
- 研究所ごとに異なるアノテーションカラム名を統一する関数
- 様々なカラム名のバリエーションに対応（behavior, action, labelなど）

#### `load_annotation_data(data_dir, video_id, lab_id=None, debug=False)`
- アノテーションデータを読み込む関数
- カラム名の統一処理を含む
- CSVとParquetファイルの両方をサポート

#### `verify_data_access(train_df, data_dir)`
- データアクセスを検証する関数
- サンプル動画（最初の10件）を詳細に検証
- アノテーションファイルと追跡ファイルの両方が見つかるかを確認
- 検証結果をJSONファイルに保存

#### `main()`
- メイン実行関数
- データの読み込み、ディレクトリ構造の確認、データアクセスの検証を実行

## パス設定

参考ノートブック（https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost）の方式に合わせて実装されています。

### Kaggle環境
- データディレクトリ: `/kaggle/input/`配下を自動検出
- 出力ディレクトリ: `/kaggle/working`
- モデル保存ディレクトリ: `/kaggle/working`

### ローカル環境
- データディレクトリ: `./data`
- 出力ディレクトリ: `./output`
- モデル保存ディレクトリ: `./models`

## データ読み込み制限設定

テスト・デバッグ用にデータ読み込みを制限できます。

```python
LIMIT_TRAIN_ROWS = 0  # 0に設定すると制限なし
LIMIT_TRAIN_ROWS_COUNT = 10  # 制限する行数
```

- `LIMIT_TRAIN_ROWS = 1`: 最初のN行のみ読み込む
- `LIMIT_TRAIN_ROWS = 0`: すべての行を読み込む（通常モード）

## 実行フロー

1. **データの読み込み**
   - `train.csv`を読み込む
   - データ読み込み制限が設定されている場合は適用

2. **データディレクトリ構造の確認**
   - すべてのサブディレクトリを確認
   - CSVファイルとParquetファイルの数をカウント
   - 追跡データディレクトリとアノテーションディレクトリの詳細を表示

3. **データアクセスの検証**
   - サンプル動画（最初の10件）を詳細に検証
   - アノテーションファイルと追跡ファイルの両方が見つかるかを確認
   - 検証結果をJSONファイルに保存

## 出力ファイル

### `data_verification_results.json`
データアクセス検証の結果が保存されます。

```json
{
  "total_videos": 1000,
  "annotation_found": 10,
  "tracking_found": 10,
  "both_found": 10,
  "neither_found": 0,
  "sample_videos": [
    {
      "video_id": 12345,
      "lab_id": "CRIM13",
      "annotation_found": true,
      "tracking_found": true,
      "annotation_path": "train_annotation/CRIM13/12345.parquet",
      "tracking_path": "train_tracking/CRIM13/12345.parquet"
    }
  ]
}
```

## エラーハンドリング

- ファイルが見つからない場合の詳細なエラーメッセージ
- データディレクトリ構造の確認情報
- デバッグモードでの詳細なログ出力

## 次のステップ

1. 検証結果を確認してください
2. 学習実装を追加してください
3. 学習済みモデルを保存してください
4. `1DCNN_submit_4.py`で推論を実行してください

## 注意事項

- このスクリプトは学習実装を含んでいません
- 推論は`1DCNN_submit_4.py`で実行してください
- Kaggle環境での実行を前提としていますが、ローカル環境でも動作します

## 参考

- Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
- 参考ノートブック: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
- 既存実装: `1DCNN_model_3.py`

