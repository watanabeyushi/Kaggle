# MABe Mouse Behavior Detection - 1D CNN Model 2 実装履歴

このドキュメントは、Extra Trees GPUベースの1D CNNモデル（`1DCNN_model_2.py`）の実装履歴をまとめたものです。

**参考コード**: https://www.kaggle.com/code/mattiaangeli/mabe-extra-trees-gpu  
**既存実装参照**: `1DCNN_model.py`

---

## 概要

この実装は、Kaggleコンペティション「MABe-mouse-behavior-detection」において、Extra Trees GPUのコード構造に合わせて1D CNNモデルを統合するために作成されました。

## 実装の目的

1. Extra Trees GPUのコードベースに1D CNNモデルを追加
2. 既存の`1DCNN_model.py`のアーキテクチャを再利用
3. DataFrameベースのデータ処理に対応
4. Extra Trees GPUとの統合を容易にする設計

---

## 実装フェーズ

### フェーズ1: 設計とアーキテクチャの決定

#### 実装内容
- Extra Trees GPUのコード構造の分析
- 既存の`1DCNN_model.py`のアーキテクチャの確認
- 統合方法の設計

#### 設計決定事項
1. **関数ベースの設計**: クラスベースではなく関数ベースで実装し、Extra Trees GPUとの統合を容易に
2. **DataFrame対応**: Extra Trees GPUのDataFrame形式を直接処理
3. **モデルアーキテクチャの再利用**: `1DCNN_model.py`と同じアーキテクチャを使用

---

### フェーズ2: データ処理機能の実装

#### 実装内容

1. **シーケンス生成関数**
   ```python
   def prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH):
   ```
   - DataFrameから数値特徴量を抽出
   - 時系列シーケンスを生成
   - データ不足時のパディング処理

2. **追跡データ読み込み関数**
   ```python
   def load_tracking_data_for_1dcnn(data_dir, video_id, sequence_length=SEQUENCE_LENGTH):
   ```
   - 動画IDから追跡データを読み込み
   - 複数の可能なパスを自動検索
   - エラーハンドリング

3. **特徴量生成関数**
   ```python
   def create_1dcnn_features_from_dataframe(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
   ```
   - Extra Trees GPUのtrain_dfから1D CNN用特徴量を生成
   - ラベルエンコーディング
   - 特徴量の正規化

---

### フェーズ3: モデル構築と訓練機能の実装

#### 実装内容

1. **モデル構築関数**
   ```python
   def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES):
   ```
   - `1DCNN_model.py`と同じアーキテクチャを使用
   - 4層の畳み込みブロック
   - BatchNormalizationとDropoutによる正則化

2. **訓練関数**
   ```python
   def train_1dcnn_model(X_train, y_train, X_val, y_val, ...):
   ```
   - モデルのコンパイル
   - コールバックの設定
   - 訓練の実行

---

### フェーズ4: 予測と提出機能の実装

#### 実装内容

1. **予測関数**
   ```python
   def predict_with_1dcnn(model, test_df, data_dir, scaler, ...):
   ```
   - test_dfからテストデータを読み込み
   - フレーム範囲に対応
   - 予測の実行

2. **提出ファイル生成関数**
   ```python
   def create_submission_1dcnn(predictions, test_metadata, label_encoder, ...):
   ```
   - 予測結果をコンペ形式に変換
   - submission_1dcnn.csvとして保存

---

### フェーズ5: メイン実行フローの実装

#### 実装内容

```python
def main():
    """
    メイン実行関数
    Extra Trees GPUのコード構造に合わせて実装
    """
```

**処理フロー:**
1. train.csvとtest.csvの読み込み
2. 1D CNN用の特徴量生成
3. 訓練・検証セットへの分割
4. モデルの訓練
5. テストデータの予測
6. 提出ファイルの生成

---

## モデルアーキテクチャ

### 1D CNN構造

既存の`1DCNN_model.py`と同じアーキテクチャを使用:

```
入力: (batch_size, sequence_length=64, num_features)

↓ Permute (2, 1)
(batch_size, num_features, sequence_length)

↓ Conv1D (64 filters, kernel=3) + BatchNorm + MaxPool + Dropout
↓ Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool + Dropout
↓ Conv1D (256 filters, kernel=3) + BatchNorm + MaxPool + Dropout
↓ Conv1D (512 filters, kernel=3) + BatchNorm + GlobalAveragePooling + Dropout

↓ Dense (256) + BatchNorm + Dropout
↓ Dense (128) + Dropout

出力: (batch_size, num_classes=15)
```

### ハイパーパラメータ

- **SEQUENCE_LENGTH**: 64フレーム
- **NUM_CLASSES**: 15クラス
- **BATCH_SIZE**: 32
- **EPOCHS**: 100
- **LEARNING_RATE**: 0.001

---

## データ処理フロー

### 訓練データ処理

1. **データ読み込み**
   - train.csvからDataFrameを読み込み（Extra Trees GPUと同じ方式）
   - 動画IDを取得

2. **特徴量生成**
   - 各動画の追跡データを読み込み
   - シーケンスデータを生成
   - ラベルエンコーディング
   - 特徴量の正規化

3. **データ分割**
   - 訓練セット: 80%
   - 検証セット: 20%

### テストデータ処理

1. **データ読み込み**
   - test.csvからDataFrameを読み込み
   - 各テスト行のメタデータを取得

2. **予測**
   - フレーム範囲に対応する追跡データを抽出
   - シーケンスデータを生成
   - 正規化
   - 予測の実行

3. **提出ファイル生成**
   - 予測結果をコンペ形式に変換
   - submission_1dcnn.csvとして保存

---

## 主要な関数

### データ処理関数

- `prepare_sequences_for_1dcnn(df, sequence_length)`: DataFrameからシーケンスデータを生成
- `load_tracking_data_for_1dcnn(data_dir, video_id, sequence_length)`: 追跡データの読み込み
- `create_1dcnn_features_from_dataframe(train_df, data_dir, sequence_length)`: 特徴量生成

### モデル構築・訓練関数

- `build_1dcnn_model(input_shape, num_classes)`: モデルの構築
- `train_1dcnn_model(X_train, y_train, X_val, y_val, ...)`: モデルの訓練

### 予測・提出関数

- `predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length)`: テストデータの予測
- `create_submission_1dcnn(predictions, test_metadata, label_encoder, output_path)`: 提出ファイルの生成

---

## 使用方法

### スタンドアロン実行

```python
# Kaggleノートブックで実行
exec(open('/kaggle/working/1DCNN_model_2.py').read())
```

または:

```python
from 1DCNN_model_2 import main
main()
```

### Extra Trees GPUコードとの統合

```python
import 1DCNN_model_2 as cnn_model

# Extra Trees GPUのデータ読み込み後
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 1D CNN用の特徴量生成
X_sequences, y_labels, label_encoder, scaler = cnn_model.create_1dcnn_features_from_dataframe(
    train_df, DATA_DIR
)

# 訓練・検証セットに分割
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_labels, test_size=0.2, random_state=42
)

# モデルの訓練
input_shape = (64, X_train.shape[2])
model = cnn_model.build_1dcnn_model(input_shape)
history, trained_model = cnn_model.train_1dcnn_model(
    X_train, y_train, X_val, y_val, input_shape=input_shape
)

# 予測
predictions, test_metadata = cnn_model.predict_with_1dcnn(
    trained_model, test_df, DATA_DIR, scaler
)

# 提出ファイル生成
submission_df = cnn_model.create_submission_1dcnn(
    predictions, test_metadata, label_encoder
)
```

---

## 実装の特徴

### 1. Extra Trees GPUとの統合

- 同じデータ構造（DataFrame）を使用
- 同じデータ読み込み方式
- 関数ベースの設計により統合が容易

### 2. 既存実装の再利用

- `1DCNN_model.py`のモデルアーキテクチャを再利用
- 検証済みの構造を使用

### 3. 柔軟性

- 関数ベースの設計により、必要に応じて個別に使用可能
- カスタマイズが容易

### 4. エラーハンドリング

- 複数の可能なパスを自動検索
- データ不足時のパディング処理
- 予測失敗時のフォールバック

---

## ファイル構成

```
MABe-mouse/
├── 1DCNN_model.py          # 既存の包括的な実装
├── 1DCNN_model_2.py         # Extra Trees GPU統合用の実装
├── 1DCNN-2-実装履歴.md      # このドキュメント
└── 1DCNN-2-修正内容.md      # 修正内容の詳細
```

---

## 既存実装との比較

### 1DCNN_model.pyとの違い

| 項目 | 1DCNN_model.py | 1DCNN_model_2.py |
|------|---------------|------------------|
| 設計 | クラスベース | 関数ベース |
| データ処理 | MABeDataLoaderクラス | 専用関数 |
| データ形式 | ファイルパス直接 | DataFrame対応 |
| 用途 | スタンドアロン実行 | Extra Trees GPU統合 |
| コード行数 | 約1334行 | 約584行 |
| 機能 | 包括的 | シンプルで統合しやすい |

### 共通点

- 同じモデルアーキテクチャ
- 同じハイパーパラメータ
- 同じ訓練設定

---

## 今後の改善案

1. **アンサンブル機能**
   - Extra Trees GPUと1D CNNの予測を組み合わせる機能の追加

2. **特徴量エンジニアリング**
   - Extra Trees GPUの特徴量と1D CNNの特徴量を組み合わせる

3. **ハイパーパラメータ最適化**
   - Extra Trees GPUと1D CNNの両方のパラメータを最適化

4. **クロスバリデーション**
   - より堅牢な評価方法の実装

5. **モデル保存・読み込み**
   - 訓練済みモデルの保存と読み込み機能の追加

---

## 注意事項

1. **データ構造の確認**
   - Extra Trees GPUのデータ構造に合わせて実装されているため、データ形式が異なる場合は調整が必要

2. **メモリ使用量**
   - シーケンスデータの生成により、メモリ使用量が増加する可能性がある

3. **実行時間**
   - 各動画のデータを個別に処理するため、データ量が多い場合は実行時間が長くなる可能性がある

---

## フェーズ6: Extra Trees GPUノートブック形式への完全対応

### 実装内容

Extra Trees GPUノートブックと同じパス設定と提出方法に完全対応しました。

#### 主な変更点

1. **パス設定の統一**
   - データディレクトリの自動検出機能を追加
   - `/kaggle/input`内の最初のディレクトリを自動使用
   - モデル保存先を`/kaggle/working`に統一

2. **データ読み込みの改善**
   - 複数の可能なパスを自動検索
   - サブディレクトリ（train/, data/）も検索
   - デバッグ情報の充実

3. **提出ファイル名の統一**
   - `submission_1dcnn.csv`から`submission.csv`に変更
   - Extra Trees GPUノートブックと同じファイル名

4. **コード構造の調整**
   - Extra Trees GPUノートブックの構造に合わせて調整
   - より統合しやすい設計に

---

## 更新履歴

- **2024年**: 初期実装（Extra Trees GPU統合用）
- **2024年**: Extra Trees GPUノートブック形式への完全対応

---

## 参考資料

- [Kaggleコンペティションページ](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection)
- [参考コード: Extra Trees GPU](https://www.kaggle.com/code/mattiaangeli/mabe-extra-trees-gpu)
- [既存実装: 1DCNN_model.py](1DCNN_model.py)
- [修正内容の詳細: 1DCNN-2-修正内容.md](1DCNN-2-修正内容.md)

