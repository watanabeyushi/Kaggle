# MABe Mouse Behavior Detection - 1D CNN Model 3 修正内容と実装履歴

このドキュメントは、Kaggle最適化版の1D CNNモデル（`1DCNN_model_3.py`）の修正内容と実装履歴をまとめたものです。

**参考コード**: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost  
**既存実装参照**: `1DCNN_model_2.py`

---

## 実装の概要

### 実装の目的
1. 参考ノートブック（harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost）のパス設定方式に合わせる
2. Kaggle環境での実行を最適化
3. 提出ファイル形式を参考ノートブックに合わせる
4. コードの簡素化と効率化

### 実装日
2024年（作成日時）

---

## 主な変更点（1DCNN_model_2.pyからの変更）

### 変更1: パス設定の最適化

#### 1.1 データディレクトリの自動検出

**変更前（1DCNN_model_2.py）:**
```python
if IS_KAGGLE:
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        data_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if data_dirs:
            DATA_DIR = str(data_dirs[0])  # 最初に見つかったディレクトリを使用
```

**変更後（1DCNN_model_3.py）:**
```python
if IS_KAGGLE:
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        # データセットディレクトリを検索（test.csvやsample_submission.csvを含むディレクトリ）
        data_dirs = []
        for item in input_dir.iterdir():
            if item.is_dir():
                # test.csvまたはsample_submission.csvが存在するか確認
                if (item / 'test.csv').exists() or (item / 'sample_submission.csv').exists():
                    data_dirs.append(str(item))
```

**改善点:**
- `test.csv`や`sample_submission.csv`の存在を確認してからデータディレクトリを決定
- より確実に正しいデータディレクトリを検出
- 参考ノートブックの方式に合わせた実装

---

### 変更2: ファイル検索関数の簡素化

#### 2.1 追跡ファイル検索関数

**変更前（1DCNN_model_2.py）:**
- `get_cached_csv_files()`: キャッシュ機能付きの複雑な検索
- `match_video_id_to_file()`: 柔軟なマッチング機能
- `load_tracking_data_for_1dcnn()`: 再帰的検索とデバッグ機能

**変更後（1DCNN_model_3.py）:**
```python
def find_tracking_file(data_dir, video_id):
    """
    追跡ファイルを検索（参考ノートブックの方式に合わせる）
    """
    data_path = Path(data_dir)
    video_id_str = str(video_id)
    
    # 可能なパスを試行（参考ノートブックの方式）
    possible_paths = [
        data_path / 'train_tracking' / f'{video_id_str}.csv',
        data_path / 'tracking' / f'{video_id_str}.csv',
        data_path / 'test_tracking' / f'{video_id_str}.csv',
        data_path / f'{video_id_str}.csv',
    ]
    
    # サブディレクトリも検索
    for subdir_name in ['train_tracking', 'tracking', 'test_tracking']:
        subdir = data_path / subdir_name
        if subdir.exists() and subdir.is_dir():
            for csv_file in subdir.glob('*.csv'):
                if csv_file.stem == video_id_str or video_id_str in csv_file.stem:
                    possible_paths.append(csv_file)
    
    # ファイルを検索
    for path in possible_paths:
        if path.exists() and path.suffix.lower() == '.csv':
            return path
    
    return None
```

**改善点:**
- キャッシュ機能を削除してコードを簡素化
- 直接的なパス検索で処理速度を向上
- 参考ノートブックの方式に合わせた実装

#### 2.2 アノテーションファイル検索関数

**変更前（1DCNN_model_2.py）:**
- `load_annotations_for_video()`: キャッシュ機能とデバッグ機能付き

**変更後（1DCNN_model_3.py）:**
```python
def find_annotation_file(data_dir, video_id):
    """
    アノテーションファイルを検索
    """
    # 同様に簡素化された実装
```

**改善点:**
- シンプルで理解しやすい実装
- 不要な機能を削除してコードを簡潔化

---

### 変更3: 特徴量生成関数の簡素化

#### 3.1 関数名と実装の変更

**変更前（1DCNN_model_2.py）:**
```python
def create_1dcnn_features_from_dataframe(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
    """
    Extra Trees GPUのtrain_dfから1D CNN用の特徴量を生成
    train_dfにactionカラムがない場合は、train_annotation/から読み込む
    """
    # 複雑なデバッグ情報とエラーハンドリング
```

**変更後（1DCNN_model_3.py）:**
```python
def create_1dcnn_features(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
    """
    train_dfから1D CNN用の特徴量を生成（参考ノートブックの方式に合わせる）
    """
    # シンプルで効率的な実装
```

**改善点:**
- 関数名を簡潔に変更
- 不要なデバッグ情報を削減
- エラーハンドリングを簡素化

---

### 変更4: 提出ファイル作成関数の改善

#### 4.1 提出ファイル形式の調整

**変更前（1DCNN_model_2.py）:**
```python
def create_submission_1dcnn(predictions, test_metadata, label_encoder, output_path=None, 
                           sample_submission_path=None, default_action=None, 
                           probability_threshold=0.1):
    """
    1D CNNの予測結果から提出ファイルを作成
    """
    # 複雑な検証と修正処理
    # validate_and_fix_submission()関数を呼び出し
```

**変更後（1DCNN_model_3.py）:**
```python
def create_submission(predictions, test_metadata, label_encoder, output_path=None):
    """
    提出ファイルを作成（参考ノートブックの方式に合わせる）
    """
    # sample_submission.csvの形式に合わせて自動調整
    # シンプルで効率的な実装
```

**改善点:**
- 関数名を簡潔に変更
- `validate_and_fix_submission()`関数を削除（必要最小限の処理に簡素化）
- 参考ノートブックの形式に合わせた実装
- `sample_submission.csv`の形式を自動検出して合わせる

#### 4.2 提出ファイルの列構成

**変更前（1DCNN_model_2.py）:**
- すべての列を含める（`row_id`, `video_id`, `agent_id`, `target_id`, `action`, `start_frame`, `stop_frame`）
- 複雑な検証処理

**変更後（1DCNN_model_3.py）:**
- `sample_submission.csv`の形式に合わせて自動調整
- 基本的には`row_id`と`action`を必須とし、他の列は`sample_submission.csv`に合わせる

---

### 変更5: 予測関数の改善

#### 5.1 予測処理の最適化

**変更前（1DCNN_model_2.py）:**
```python
def predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length=SEQUENCE_LENGTH):
    """
    1D CNNモデルでテストデータを予測
    """
    # 複雑なパス検索とエラーハンドリング
```

**変更後（1DCNN_model_3.py）:**
```python
def predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length=SEQUENCE_LENGTH):
    """
    1D CNNモデルでテストデータを予測（参考ノートブックの方式に合わせる）
    """
    # find_tracking_file()を使用してシンプルに実装
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="予測中", disable=not TQDM_AVAILABLE):
        # シンプルで効率的な処理
```

**改善点:**
- `find_tracking_file()`を使用してコードを統一
- 進捗バーの表示を改善
- エラーハンドリングを簡素化

---

### 変更6: メイン関数の簡素化

#### 6.1 実行フローの最適化

**変更前（1DCNN_model_2.py）:**
- 詳細なセクション表示（`print_section()`）
- 複雑なエラーハンドリング
- 詳細な統計情報の表示

**変更後（1DCNN_model_3.py）:**
- シンプルなセクション表示（`[1/4]`, `[2/4]`など）
- 必要最小限の情報表示
- 参考ノートブックの方式に合わせた実装

---

## 削除された機能

### 削除1: ファイル検索のキャッシュ機能

**削除理由:**
- コードの複雑さを増やすだけで、Kaggle環境では大きな効果がない
- 参考ノートブックでも使用されていない

**影響:**
- コードが簡潔になり、理解しやすくなった
- 実行速度への影響は最小限

### 削除2: 詳細なデバッグ情報

**削除理由:**
- Kaggle環境では不要な詳細情報
- 実行時間の短縮

**影響:**
- コードが簡潔になった
- 必要最小限の情報のみ表示

### 削除3: 提出ファイルの複雑な検証処理

**削除理由:**
- `sample_submission.csv`の形式に合わせることで十分
- 参考ノートブックでも同様の簡素化された処理

**影響:**
- コードが簡潔になった
- `sample_submission.csv`の形式に自動的に合わせるため、問題なし

---

## 追加された機能

### 追加1: データディレクトリの自動検出改善

**機能:**
- `test.csv`や`sample_submission.csv`の存在を確認してからデータディレクトリを決定
- より確実に正しいデータディレクトリを検出

**実装箇所:**
```python
# パス設定（参考ノートブックの方式に合わせる）
if IS_KAGGLE:
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        data_dirs = []
        for item in input_dir.iterdir():
            if item.is_dir():
                if (item / 'test.csv').exists() or (item / 'sample_submission.csv').exists():
                    data_dirs.append(str(item))
```

---

## 保持された機能

### 保持1: 1D CNNモデルアーキテクチャ

**理由:**
- `1DCNN_model_2.py`のアーキテクチャは効果的
- モデル構造の変更は不要

**実装:**
```python
def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES):
    """
    1D CNNモデルの構築
    1DCNN_model_2.pyの実装を参照
    """
    # 同じアーキテクチャを維持
```

### 保持2: シーケンス生成ロジック

**理由:**
- データ処理のロジックは効果的
- 変更の必要がない

**実装:**
```python
def prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH):
    """
    データフレームから1D CNN用のシーケンスデータを準備
    """
    # 同じロジックを維持
```

### 保持3: 訓練パラメータ

**理由:**
- Kaggle提出要件（9時間以内）を考慮した適切な設定
- 変更の必要がない

**実装:**
```python
SEQUENCE_LENGTH = 64
NUM_CLASSES = 15
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
```

---

## 実装履歴

### バージョン3.0（現在）

**実装日:** 2024年

**主な変更:**
1. 参考ノートブック（harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost）の方式に合わせた実装
2. パス設定の最適化
3. ファイル検索関数の簡素化
4. 提出ファイル作成関数の改善
5. コード全体の簡素化と効率化

**目的:**
- Kaggle環境での実行を最適化
- 参考ノートブックとの互換性を向上
- コードの可読性と保守性を向上

---

## 参考ノートブックとの対応

### パス設定

**参考ノートブック:**
- `/kaggle/input`内のデータセットディレクトリを自動検出
- `test.csv`や`sample_submission.csv`の存在を確認

**1DCNN_model_3.py:**
- 同様の方式を実装
- より確実なデータディレクトリ検出

### 提出ファイル形式

**参考ノートブック:**
- `sample_submission.csv`の形式に合わせる
- `row_id`と`action`を必須とする

**1DCNN_model_3.py:**
- 同様の方式を実装
- `sample_submission.csv`の形式を自動検出して合わせる

### データ読み込み

**参考ノートブック:**
- シンプルで直接的なファイル検索
- キャッシュ機能なし

**1DCNN_model_3.py:**
- 同様の方式を実装
- `find_tracking_file()`と`find_annotation_file()`で統一

---

## コードの比較

### コード行数の比較

| ファイル | 行数 | 特徴 |
|---------|------|------|
| `1DCNN_model_2.py` | 1,566行 | 詳細なデバッグ情報とキャッシュ機能 |
| `1DCNN_model_3.py` | 966行 | 簡素化された実装 |

**削減率:** 約38%のコード削減

### 関数数の比較

| ファイル | 関数数 | 特徴 |
|---------|--------|------|
| `1DCNN_model_2.py` | 約15関数 | 複雑な検証と修正処理 |
| `1DCNN_model_3.py` | 約12関数 | シンプルで効率的な実装 |

---

## 使用方法

### Kaggle環境での実行

1. Kaggle Notebookで`1DCNN_model_3.py`をアップロード
2. データセットを追加（MABe Mouse Behavior Detection）
3. ノートブックを実行
4. `/kaggle/working/submission.csv`が自動生成される

### ローカル環境での実行

1. データディレクトリを`./data`に配置
2. `python 1DCNN_model_3.py`を実行
3. `./output/submission.csv`が生成される

---

## 今後の改善案

### 改善案1: さらなる最適化

- データ読み込みの並列化
- メモリ使用量の最適化
- 実行時間の短縮

### 改善案2: 機能の追加

- モデルアンサンブル機能
- クロスバリデーション機能
- ハイパーパラメータの自動調整

### 改善案3: エラーハンドリングの強化

- より詳細なエラーメッセージ
- リトライ機能
- フォールバック処理の改善

---

## まとめ

`1DCNN_model_3.py`は、参考ノートブック（harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost）の方式に合わせて実装された、Kaggle最適化版の1D CNNモデルです。

**主な特徴:**
1. 参考ノートブックのパス設定方式に合わせた実装
2. コードの簡素化と効率化（約38%のコード削減）
3. 提出ファイル形式の自動調整
4. Kaggle環境での実行を最適化

**利点:**
- コードが簡潔で理解しやすい
- 参考ノートブックとの互換性が高い
- Kaggle環境での実行が容易
- 保守性が向上

**注意点:**
- キャッシュ機能がないため、大量のデータ処理では若干の速度低下の可能性
- デバッグ情報が少ないため、問題発生時の調査がやや困難

---

## 参考資料

- **参考ノートブック**: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
- **既存実装**: `1DCNN_model_2.py`
- **コンペティション**: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection

---

**最終更新日:** 2024年

