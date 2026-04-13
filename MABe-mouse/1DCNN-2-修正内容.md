# MABe Mouse Behavior Detection - 1D CNN Model 2 修正内容一覧

このドキュメントは、Extra Trees GPUベースの1D CNNモデル（`1DCNN_model_2.py`）の修正内容をまとめたものです。

**参考コード**: https://www.kaggle.com/code/mattiaangeli/mabe-extra-trees-gpu  
**既存実装参照**: `1DCNN_model.py`

---

## 修正の概要

### 修正の目的
1. Extra Trees GPUのコード構造に合わせた1D CNN実装
2. 既存の`1DCNN_model.py`のアーキテクチャを活用
3. DataFrameベースのデータ処理への対応
4. Extra Trees GPUとの統合を容易にする設計

---

## 修正1: Extra Trees GPUデータ構造への対応

### 修正内容

#### 1.1 DataFrameベースのデータ処理

**実装した関数:**
```python
def prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH):
    """
    Extra Trees GPUのデータフレームから1D CNN用のシーケンスデータを準備
    """
```

**機能:**
- DataFrameから数値特徴量を自動抽出
- 時系列シーケンスを自動生成
- データ不足時のパディング処理

**従来の実装との違い:**
- 従来: ファイルパスから直接読み込み
- 新実装: DataFrameを直接処理（Extra Trees GPUのデータフローに対応）

#### 1.2 動画データの読み込み関数

**実装した関数:**
```python
def load_tracking_data_for_1dcnn(data_dir, video_id, sequence_length=SEQUENCE_LENGTH):
    """
    指定された動画IDの追跡データを読み込んで1D CNN用に変換
    """
```

**機能:**
- 複数の可能なパスを自動検索
- `train_tracking/`, `tracking/`, ルートディレクトリを順に試行
- エラーハンドリングの強化

**改善点:**
- データディレクトリ構造の違いに対応
- フォールバック機能の実装

---

## 修正2: 特徴量生成パイプラインの実装

### 修正内容

#### 2.1 DataFrameから特徴量生成

**実装した関数:**
```python
def create_1dcnn_features_from_dataframe(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
    """
    Extra Trees GPUのtrain_dfから1D CNN用の特徴量を生成
    """
```

**主な処理:**
1. train_dfから動画IDを取得
2. 各動画の追跡データを読み込み
3. シーケンスデータを生成
4. ラベルエンコーディング
5. 特徴量の正規化

**特徴:**
- Extra Trees GPUのデータフローに完全対応
- 自動的なラベルエンコーディング
- StandardScalerによる正規化

**従来実装との違い:**
- 従来: `MABeDataLoader`クラスで包括的に処理
- 新実装: 関数ベースでExtra Trees GPUの構造に合わせて実装

#### 2.2 シーケンス生成の最適化

**改善点:**
- データが少ない場合の自動パディング
- フレーム範囲の自動調整
- メモリ効率的な処理

---

## 修正3: 予測パイプラインの実装

### 修正内容

#### 3.1 テストデータ予測関数

**実装した関数:**
```python
def predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length=SEQUENCE_LENGTH):
    """
    1D CNNモデルでテストデータを予測
    """
```

**機能:**
- test_dfからテストデータを読み込み
- フレーム範囲（start_frame, stop_frame）に対応
- 複数シーケンスの平均を取る処理
- エラーハンドリングの強化

**特徴:**
- Extra Trees GPUのtest_df形式に対応
- メタデータの自動保持
- 予測失敗時のフォールバック

#### 3.2 提出ファイル生成

**実装した関数:**
```python
def create_submission_1dcnn(predictions, test_metadata, label_encoder, output_path=None):
    """
    1D CNNの予測結果から提出ファイルを作成
    """
```

**機能:**
- 予測結果をコンペ形式に変換
- メタデータと予測を結合
- submission_1dcnn.csvとして保存

**従来実装との違い:**
- 従来: より詳細な検証機能付き
- 新実装: Extra Trees GPUのシンプルな形式に対応

---

## 修正4: モデルアーキテクチャの継承

### 修正内容

#### 4.1 既存モデルの再利用

**実装:**
```python
def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES):
    """
    1D CNNモデルの構築
    1DCNN_model.pyの実装を参照
    """
```

**特徴:**
- `1DCNN_model.py`と同じアーキテクチャを使用
- 4層の畳み込みブロック（64→128→256→512フィルタ）
- BatchNormalizationとDropoutによる正則化
- GlobalAveragePooling1Dによる時系列集約

**利点:**
- 既存の実装を再利用
- 一貫性のあるモデル構造
- 検証済みのアーキテクチャ

---

## 修正5: 訓練パイプラインの実装

### 修正内容

#### 5.1 訓練関数の実装

**実装した関数:**
```python
def train_1dcnn_model(X_train, y_train, X_val, y_val, 
                     input_shape, num_classes=NUM_CLASSES,
                     batch_size=BATCH_SIZE, epochs=EPOCHS,
                     learning_rate=LEARNING_RATE,
                     model_save_path=MODEL_DIR):
    """
    1D CNNモデルの訓練
    """
```

**機能:**
- モデルの構築
- コールバックの設定（EarlyStopping, ReduceLROnPlateau, ModelCheckpoint）
- モデルのコンパイルと訓練
- 訓練履歴の返却

**特徴:**
- `1DCNN_model.py`と同じ訓練設定
- モデル保存機能
- 検証データによる早期停止

---

## 修正6: メイン実行フローの実装

### 修正内容

#### 6.1 Extra Trees GPU形式への対応

**実装:**
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

**特徴:**
- Extra Trees GPUと同じデータ読み込み方式
- シンプルで理解しやすいフロー
- エラーハンドリングの実装

---

## 修正の影響と効果

### パフォーマンスへの影響

1. **データ処理の効率化**
   - DataFrameベースの処理により、Extra Trees GPUとの統合が容易
   - メモリ効率的なシーケンス生成

2. **コードの再利用性**
   - 既存のモデルアーキテクチャを再利用
   - 関数ベースの設計により、他のコードとの統合が容易

### 機能面への影響

1. **Extra Trees GPUとの統合**
   - 同じデータ構造を使用
   - 同じフローで処理可能

2. **柔軟性の向上**
   - 関数ベースの設計により、必要に応じて個別に使用可能
   - カスタマイズが容易

---

## 修正のまとめ

### 主要な修正項目

| 修正項目 | 従来実装 | 新実装 | 効果 |
|---------|---------|--------|------|
| データ処理 | クラスベース | 関数ベース | Extra Trees GPUとの統合が容易 |
| データ読み込み | ファイルパス直接 | DataFrame対応 | 柔軟性向上 |
| 特徴量生成 | MABeDataLoader | 専用関数 | Extra Trees GPUフローに対応 |
| 予測処理 | 包括的 | シンプル | 統合しやすい |

### コード行数の比較

- **1DCNN_model.py**: 約1334行（包括的な実装）
- **1DCNN_model_2.py**: 約584行（Extra Trees GPU統合用）

### 主な追加機能

1. DataFrameベースのデータ処理（約100行）
2. Extra Trees GPU対応の特徴量生成（約150行）
3. シンプルな予測パイプライン（約100行）
4. 統合用のメイン関数（約100行）

---

## 使用例

### Extra Trees GPUコードとの統合

```python
# Extra Trees GPUのコード内で使用
import 1DCNN_model_2 as cnn_model

# データの読み込み（Extra Trees GPUと同じ）
train_df = pd.read_csv('/kaggle/input/.../train.csv')
test_df = pd.read_csv('/kaggle/input/.../test.csv')

# 1D CNN用の特徴量生成
X_sequences, y_labels, label_encoder, scaler = cnn_model.create_1dcnn_features_from_dataframe(
    train_df, DATA_DIR
)

# モデルの訓練
model = cnn_model.build_1dcnn_model(input_shape=(64, n_features))
history, trained_model = cnn_model.train_1dcnn_model(
    X_train, y_train, X_val, y_val, input_shape=(64, n_features)
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

## 今後の改善案

1. **アンサンブル機能**
   - Extra Trees GPUと1D CNNの予測を組み合わせる機能

2. **特徴量エンジニアリング**
   - Extra Trees GPUの特徴量と1D CNNの特徴量を組み合わせる

3. **ハイパーパラメータ最適化**
   - Extra Trees GPUと1D CNNの両方のパラメータを最適化

4. **クロスバリデーション**
   - より堅牢な評価方法の実装

---

---

## 修正6: Extra Trees GPUノートブック形式への完全対応

### 修正日
Extra Trees GPUノートブック統合後

### 修正内容

#### 6.1 パス設定の統一

**修正前:**
```python
if IS_KAGGLE:
    DATA_DIR = '/kaggle/input/mabe-mouse-behavior-detection'
    OUTPUT_DIR = '/kaggle/working'
    MODEL_DIR = '/kaggle/working/models'
```

**修正後:**
```python
# Extra Trees GPUノートブックと同じ方式
if IS_KAGGLE:
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        data_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if data_dirs:
            DATA_DIR = str(data_dirs[0])  # 最初に見つかったディレクトリを使用
        else:
            DATA_DIR = '/kaggle/input/mabe-mouse-behavior-detection'
    else:
        DATA_DIR = '/kaggle/input/mabe-mouse-behavior-detection'
    OUTPUT_DIR = '/kaggle/working'
    MODEL_DIR = '/kaggle/working'  # Extra Trees GPUと同じく/kaggle/workingに統一
```

**効果:**
- Extra Trees GPUノートブックと同じパス検出方式
- データセット名が異なっても自動検出
- モデル保存先を/kaggle/workingに統一

#### 6.2 データ読み込み方法の統一

**修正内容:**
- train.csvとtest.csvの検索方法を改善
- サブディレクトリ（train/, data/）も検索
- データディレクトリの内容を表示してデバッグを容易に

**修正前:**
```python
train_csv = Path(DATA_DIR) / 'train.csv'
if not train_csv.exists():
    print(f"警告: train.csvが見つかりません")
    return
```

**修正後:**
```python
train_csv = Path(DATA_DIR) / 'train.csv'
# 代替パスも試行
if not train_csv.exists():
    for subdir in ['train', 'data']:
        alt_path = Path(DATA_DIR) / subdir / 'train.csv'
        if alt_path.exists():
            train_csv = alt_path
            break

if not train_csv.exists():
    # デバッグ情報を表示
    print(f"データディレクトリの内容を確認中: {DATA_DIR}")
    data_path = Path(DATA_DIR)
    if data_path.exists():
        files = list(data_path.glob('*.csv'))
        dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
        print(f"  見つかったCSVファイル: {[f.name for f in files[:5]]}")
        print(f"  見つかったディレクトリ: {dirs[:5]}")
    return
```

**効果:**
- Extra Trees GPUノートブックと同じデータ検索方式
- エラー時のデバッグ情報を充実
- 複数のディレクトリ構造に対応

#### 6.3 提出ファイル名の統一

**修正前:**
```python
output_path = '/kaggle/working/submission_1dcnn.csv'
```

**修正後:**
```python
# Extra Trees GPUノートブックと同じく、submission.csvとして保存
output_path = '/kaggle/working/submission.csv'
```

**効果:**
- Extra Trees GPUノートブックと同じ提出ファイル名
- Kaggleの提出システムと完全に互換
- ファイル名の競合を回避

#### 6.4 提出ファイル生成の改善

**修正内容:**
- 提出ファイルパスを明示的に指定
- Extra Trees GPUノートブックと同じ保存方式

**修正後:**
```python
submission_df = create_submission_1dcnn(
    predictions, test_metadata, label_encoder,
    output_path='/kaggle/working/submission.csv' if IS_KAGGLE else os.path.join(OUTPUT_DIR, 'submission.csv')
)
print(f"提出ファイル: {'/kaggle/working/submission.csv' if IS_KAGGLE else os.path.join(OUTPUT_DIR, 'submission.csv')}")
```

**効果:**
- 提出ファイルの場所を明確に表示
- Extra Trees GPUノートブックと同じ動作

---

## 修正のまとめ（更新）

### 主要な修正項目（最新）

| 修正項目 | 修正前の状態 | 修正後の状態 | 効果 |
|---------|------------|------------|------|
| パス設定 | 固定パス | 自動検出 | Extra Trees GPUと同じ方式 |
| データ読み込み | 単一パス | 複数パス検索 | 柔軟性向上 |
| 提出ファイル名 | submission_1dcnn.csv | submission.csv | 統一性向上 |
| モデル保存先 | /kaggle/working/models | /kaggle/working | Extra Trees GPUと同じ |

---

## 修正7: アノテーションデータの読み込み処理の追加

### 修正日
エラー解消対応

### 修正内容

#### 7.1 アノテーションファイル読み込み関数の追加

**実装した関数:**
```python
def load_annotations_for_video(data_dir, video_id):
    """
    指定された動画IDのアノテーションデータを読み込む
    """
```

**機能:**
- 複数の可能なパスを自動検索（train_annotation/, train_annotations/, annotation/, annotations/）
- アノテーションファイルが見つからない場合はNoneを返す
- エラーハンドリングの実装

**修正前の問題点:**
```python
if 'action' in train_df.columns:
    y_encoded = label_encoder.fit_transform(train_df['action'].values)
else:
    raise ValueError("'action' column not found in train_df")
```

**修正後の改善点:**
```python
# train_dfにactionカラムがない場合、アノテーションファイルから読み込む
if 'action' in train_df.columns:
    # train_dfにactionカラムがある場合
    for _, row in train_df.iterrows():
        video_id = row.get('video_id', '')
        if video_id:
            video_actions[video_id] = row['action']
else:
    # train_dfにactionカラムがない場合、アノテーションファイルから読み込む
    print("train_dfにactionカラムが見つかりません。アノテーションファイルから読み込みます...")
    # アノテーションファイルから読み込む処理
```

#### 7.2 特徴量生成関数の改善

**改善点:**
- train_dfにactionカラムがない場合の処理を追加
- アノテーションファイルから行動データを読み込む
- デフォルト行動の設定
- 処理進捗の表示

**効果:**
- train.csvにactionカラムがなくても動作
- アノテーションファイルから自動的に行動データを取得
- エラーハンドリングの強化

---

## 修正8: アノテーションファイル検索の強化とデバッグ機能の追加

### 修正日
アノテーションファイル検索エラー解消対応

### 修正内容

#### 8.1 アノテーションファイル検索機能の強化

**修正前の問題点:**
- 固定パスのみを検索
- デバッグ情報が不足
- ファイル名のマッチングが不十分

**修正後の改善点:**

1. **より多くのパスパターンの検索**
   ```python
   possible_paths = [
       data_path / 'train_annotation' / f'{video_id}.csv',
       data_path / 'train_annotations' / f'{video_id}.csv',
       data_path / 'annotation' / f'{video_id}.csv',
       data_path / 'annotations' / f'{video_id}.csv',
       # 拡張子なしのパスも試行
       data_path / 'train_annotation' / f'{video_id}',
       data_path / 'train_annotations' / f'{video_id}',
   ]
   ```

2. **ディレクトリ内の全ファイル検索**
   - アノテーションディレクトリ内のすべてのCSVファイルを検索
   - ファイル名にvideo_idが含まれるファイルを自動検出

3. **デバッグモードの追加**
   - 最初の数件の動画で詳細なデバッグ情報を表示
   - データディレクトリの構造を確認
   - 見つかったファイルと見つからなかったファイルをカウント

#### 8.2 データディレクトリ構造の確認機能

**追加した機能:**
```python
# データディレクトリの構造を確認
data_path = Path(data_dir)
subdirs = [d.name for d in data_path.iterdir() if d.is_dir()]
annotation_dirs = [d for d in subdirs if 'annotation' in d.lower() or 'train' in d.lower()]

# アノテーションディレクトリ内のCSVファイルを確認
for ann_dir_name in annotation_dirs:
    ann_dir = data_path / ann_dir_name
    csv_files = list(ann_dir.glob('*.csv'))
    print(f"  {ann_dir_name}内のCSVファイル数: {len(csv_files)}")
```

**効果:**
- 実際のデータ構造を確認可能
- 問題の原因を特定しやすい
- デバッグが容易

#### 8.3 代替検索方法の実装

**追加した機能:**
1. **ファイル名からの動画IDマッチング**
   - アノテーションディレクトリ内のすべてのファイルを確認
   - ファイル名にvideo_idが含まれるファイルを検索
   - 部分一致でもマッチング

2. **頻度ベースの行動選択**
   - 複数の行動がある場合、最も頻度の高い行動を選択
   - `value_counts()`を使用して最頻値を取得

3. **フォールバック処理**
   - アノテーションファイルが見つからない場合の代替処理
   - デフォルト行動の設定

#### 8.4 統計情報の表示

**追加した情報:**
- 処理する動画ID数
- アノテーションファイルが見つかった動画数
- アノテーションファイルが見つからなかった動画数
- 動画IDの例
- アノテーションディレクトリ内のCSVファイル数

**効果:**
- 問題の特定が容易
- データの状態を把握可能
- デバッグが効率的

---

## 修正9: 再帰的ファイル検索の実装

### 修正日
サブディレクトリ内ファイル検索エラー解消対応

### 修正内容

#### 9.1 再帰的検索機能の追加

**問題:**
- `train_annotation`と`train_tracking`ディレクトリ内に直接CSVファイルがない
- ファイルがサブディレクトリ内にある可能性がある
- `glob('*.csv')`ではサブディレクトリ内のファイルを検索できない

**修正前:**
```python
csv_files = list(annotation_dir.glob('*.csv'))  # サブディレクトリを検索しない
```

**修正後:**
```python
csv_files = list(annotation_dir.rglob('*.csv'))  # 再帰的に検索
```

**効果:**
- サブディレクトリ内のファイルも検索可能
- より柔軟なファイル構造に対応

#### 9.2 ファイル名マッチングの改善

**改善点:**
- 数値の文字列変換に対応
- 部分一致の改善
- 型変換を考慮したマッチング

**修正後:**
```python
video_id_str = str(video_id)
for csv_file in csv_files:
    file_stem = csv_file.stem
    # 完全一致、部分一致、または数値として一致
    if (file_stem == video_id_str or 
        video_id_str in file_stem or 
        file_stem in video_id_str or
        file_stem == str(int(video_id)) if isinstance(video_id, (int, float)) else False):
        possible_paths.append(csv_file)
```

#### 9.3 デバッグ情報の強化

**追加した情報:**
- 再帰的検索で見つかったファイル数
- ファイルの相対パスを表示
- サブディレクトリの構造を表示
- サブディレクトリ内のファイル数も確認

**効果:**
- 実際のファイル構造を把握可能
- 問題の原因を特定しやすい

#### 9.4 追跡データ読み込み関数の改善

**修正内容:**
- `load_tracking_data_for_1dcnn`にも再帰的検索を追加
- デバッグモードの追加
- エラーメッセージの改善

**効果:**
- 追跡データもサブディレクトリから検索可能
- デバッグが容易

---

## 修正11: 実行状況表示機能の追加

### 修正日
実行状況表示機能追加

### 修正内容

#### 11.1 進捗バーライブラリ（tqdm）の追加

**追加した機能:**
- `tqdm`ライブラリを使用した進捗バー表示
- `tqdm`が利用できない場合の代替処理

**実装:**
```python
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # tqdmがない場合の代替関数
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(desc)
        return iterable
```

**効果:**
- データ処理の進捗を視覚的に確認可能
- 残り時間の推定表示

#### 11.2 処理時間計測機能の追加

**追加した機能:**
- 各処理ステップの実行時間を計測
- 経過時間を`HH:MM:SS`形式で表示

**実装:**
```python
import time
from datetime import datetime, timedelta

def print_progress(message, start_time=None):
    """進捗メッセージを表示（経過時間付き）"""
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f"[{elapsed_str}] {message}")
    else:
        print(message)
```

**計測している処理:**
- データ読み込み時間
- アノテーションファイル読み込み時間
- シーケンス生成時間
- 特徴量正規化時間
- モデル訓練時間
- 予測時間
- 総実行時間

#### 11.3 セクション表示機能の追加

**追加した機能:**
- 各処理ステップを明確に区別
- セクションレベルの表示

**実装:**
```python
def print_section(title, level=1):
    """セクションタイトルを表示"""
    if level == 1:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)
    elif level == 2:
        print("\n" + "-" * 60)
        print(f"  {title}")
        print("-" * 60)
```

#### 11.4 システム情報表示機能の追加

**追加した機能:**
- GPU/CPU環境の自動検出と表示
- TensorFlowバージョンの表示
- メモリ使用量の表示（可能な場合）

**実装:**
```python
def get_system_info():
    """システム情報を取得して表示"""
    # GPU情報
    gpus = tf.config.list_physical_devices('GPU')
    # メモリ情報（psutilが利用可能な場合）
    # TensorFlowバージョン
```

**表示内容:**
- Kaggle環境の有無
- GPU検出状況とデバイス名
- TensorFlowバージョン
- 利用可能メモリ（可能な場合）

#### 11.5 データ処理ループでの進捗バー表示

**追加した機能:**
- アノテーションファイル読み込み時の進捗バー
- 動画データ処理時の進捗バー

**実装:**
```python
for idx, video_id in enumerate(tqdm(video_ids, desc="アノテーション読み込み", disable=not TQDM_AVAILABLE)):
    # 処理
```

**効果:**
- 大量のデータ処理でも進捗を把握可能
- 残り時間の推定表示

#### 11.6 訓練結果の詳細表示

**追加した機能:**
- 訓練開始時のパラメータ表示
- 訓練完了時の結果表示
- 最終エポック数と検証結果の表示

**実装:**
```python
print(f"訓練開始: バッチサイズ={batch_size}, エポック数={epochs}")
print(f"訓練データサイズ: {X_train.shape[0]:,}サンプル")
print(f"検証データサイズ: {X_val.shape[0]:,}サンプル")
# 訓練後
print(f"訓練時間: {timedelta(seconds=int(train_elapsed))}")
print(f"最終エポック: {len(history.history['loss'])}")
print(f"最終検証損失: {history.history['val_loss'][-1]:.4f}")
```

#### 11.7 開始・終了時刻の表示

**追加した機能:**
- 実行開始時刻の表示
- 実行終了時刻の表示
- 総実行時間の表示

**実装:**
```python
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# 処理実行
print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"総実行時間: {timedelta(seconds=int(total_elapsed))}")
```

**効果:**
- 実行時間の把握が容易
- ログとして記録可能

---

## 修正12: 提出形式の修正とエラーハンドリングの強化

### 修正日
提出形式対応とエラーハンドリング強化

### 修正内容

#### 12.1 sample_submission.csv形式への対応

**問題:**
- 提出ファイルがsample_submission.csvの形式と一致していない可能性

**修正内容:**
- `sample_submission.csv`の形式を自動検出
- カラム順序をsample_submission.csvに合わせる
- データ型の確認と表示

**実装:**
```python
def create_submission_1dcnn(predictions, test_metadata, label_encoder, 
                            output_path=None, sample_submission_path=None):
    # sample_submission.csvの形式を確認
    if sample_submission_path and Path(sample_submission_path).exists():
        sample_df = pd.read_csv(sample_submission_path)
        sample_columns = list(sample_df.columns)
        # カラム順序をsample_submission.csvに合わせる
        if sample_columns:
            available_columns = [col for col in sample_columns if col in submission_df.columns]
            if available_columns:
                submission_df = submission_df[available_columns]
```

**効果:**
- sample_submission.csvと完全に同じ形式で出力
- カラム順序の一致
- データ型の確認

#### 12.2 サブディレクトリ内の追跡ファイル検索の改善

**問題:**
- train_trackingディレクトリ内のサブディレクトリ（CRIM13, DeliriousFlyなど）内のファイルを検索できていない

**修正内容:**
- サブディレクトリ内のファイルを直接検索
- より柔軟なファイル名マッチング

**実装:**
```python
# サブディレクトリ内も直接検索（例: CRIM13/video_id.csv）
if tracking_dir.exists():
    subdirs = [d for d in tracking_dir.iterdir() if d.is_dir()]
    for subdir in subdirs:
        subdir_file = subdir / f'{video_id}.csv'
        if subdir_file.exists():
            possible_paths.append(subdir_file)
        # サブディレクトリ内のファイル名にvideo_idが含まれる場合も検索
        subdir_csv_files = list(subdir.glob('*.csv'))
        for csv_file in subdir_csv_files:
            if video_id_str in csv_file.stem or csv_file.stem in video_id_str:
                possible_paths.append(csv_file)
```

**効果:**
- サブディレクトリ内のファイルも検索可能
- より多くのファイルを発見

#### 12.3 エラーハンドリングの強化

**問題:**
- ファイルが見つからない場合に処理が停止する
- エラーメッセージが大量に表示される

**修正内容:**
- ファイルが見つからない動画をスキップして続行
- エラー統計の表示
- デバッグモードでのみ詳細なエラーを表示

**実装:**
```python
try:
    sequences, feature_cols = load_tracking_data_for_1dcnn(...)
except FileNotFoundError as e:
    # ファイルが見つからない場合はスキップして続行
    if not debug_tracking:
        if processed_videos % 100 == 0:  # 100件ごとに統計を表示
            print(f"  処理済み動画数: {processed_videos}/{len(video_ids)}")
    continue
except Exception as e:
    # その他のエラーもスキップして続行
    if processed_videos < 10:  # 最初の10件のみ詳細なエラーを表示
        print(f"Warning: Error loading tracking data for video {video_id}: {e}")
    continue
```

**エラー統計の表示:**
```python
print(f"\n動画処理統計:")
print(f"  総動画数: {total_videos}")
print(f"  成功: {successful_videos}")
print(f"  失敗（ファイル未検出など）: {failed_videos}")

if failed_videos > 0:
    print(f"警告: {failed_videos}個の動画の処理に失敗しましたが、{successful_videos}個の動画のデータを使用して続行します。")
```

**効果:**
- 一部のファイルが見つからなくても処理を続行
- エラーメッセージの過剰な表示を防止
- 処理状況の把握が容易

#### 12.4 空ファイルのチェック

**追加した機能:**
- ファイルが存在しても空の場合のチェック
- 空ファイルのスキップ

**実装:**
```python
df = pd.read_csv(tracking_path)
if len(df) == 0:
    if debug:
        print(f"  Warning: ファイルが空です: {tracking_path}")
    continue
```

#### 12.5 デバッグ情報の改善

**追加した機能:**
- サブディレクトリ内のファイル例を表示
- より詳細なエラー情報

**実装:**
```python
if debug:
    # サブディレクトリ内のファイル例を表示
    for subdir_name in subdirs[:3]:
        subdir = tracking_dir / subdir_name
        sub_csv_files = list(subdir.glob('*.csv'))
        if len(sub_csv_files) > 0:
            print(f"      {subdir_name}内のCSVファイル例: {[f.name for f in sub_csv_files[:3]]}")
```

---

## 参考

- 既存実装: `1DCNN_model.py`
- 参考コード: https://www.kaggle.com/code/mattiaangeli/mabe-extra-trees-gpu
- 実装履歴の詳細は `1DCNN-2-実装履歴.md` を参照

