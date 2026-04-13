"""
MABe Mouse Behavior Detection - 1D CNN実装（Kaggle最適化版 - 学習用）
Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
参考: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
既存実装参照: 1DCNN_model_3.py

このスクリプトは学習専用です。推論は1DCNN_submit_4.pyで実行してください。
"""

import os
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

# TensorFlowとprotobufの互換性問題の警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERRORレベルのみ表示
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

# すべての警告を抑制
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# 特定の警告メッセージを抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*MessageFactory.*GetPrototype.*')
warnings.filterwarnings('ignore', message='.*Unable to register.*factory.*')
warnings.filterwarnings('ignore', message='.*cuFFT.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cuda.*')

# protobufエラーを完全に抑制
_stderr_buffer = StringIO()
with redirect_stderr(_stderr_buffer):
    try:
        import google.protobuf
        if hasattr(google.protobuf, 'message'):
            pass
    except (ImportError, AttributeError, TypeError):
        pass
    except Exception:
        pass

import numpy as np
import pandas as pd
import time
import re
import pickle
import json
import gc
import os
from datetime import datetime, timedelta

# psutilのインポート（オプション、メモリ監視用）
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# 進捗バーライブラリ（Kaggle環境では利用可能）
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(desc)
        return iterable

# TensorFlowのログレベルを設定（stderrをリダイレクト）
_stderr_buffer = StringIO()
with redirect_stderr(_stderr_buffer):
    try:
        import tensorflow as tf
        # TensorFlowのログレベルを最高レベルに設定
        tf.get_logger().setLevel('ERROR')
        # TensorFlowの警告を抑制
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('tensorflow').propagate = False
    except Exception:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            import logging
            logging.getLogger('tensorflow').setLevel(logging.ERROR)
            logging.getLogger('tensorflow').propagate = False
        except Exception:
            pass

from tensorflow import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import pickle

# Kaggle環境の検出
IS_KAGGLE = os.path.exists('/kaggle/input')

# ファイル検索のキャッシュ（効率化のため）
_file_cache = {
    'tracking': {},
    'annotation': {},
}

# モデルパラメータ（Kaggle提出要件: 9時間以内の実行時間を考慮）
SEQUENCE_LENGTH = 64
NUM_CLASSES = 15
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# メモリ効率化設定
USE_GENERATOR = True  # データジェネレータを使用するかどうか
MEMORY_EFFICIENT_MODE = True  # メモリ効率モード
REDUCED_BATCH_SIZE = 16  # メモリ不足時のバッチサイズ
REDUCED_SEQUENCE_LENGTH = 32  # メモリ不足時のシーケンス長
ENABLE_GC = True  # ガベージコレクションを有効にするかどうか
MEMORY_MONITORING = True  # メモリ使用量を監視するかどうか

# データ読み込み制限設定（テスト・デバッグ用）
# LIMIT_TRAIN_ROWS = 1: 最初のN行のみ読み込む
# LIMIT_TRAIN_ROWS = 0: すべての行を読み込む（通常モード）
LIMIT_TRAIN_ROWS = 0  # 0に設定すると制限なし
LIMIT_TRAIN_ROWS_COUNT = 10  # 制限する行数

# アノテーション読み込み制限設定（テスト・デバッグ用）
# LIMIT_ANNOTATION_LOAD = 0.5: 半分だけ読み込む
# LIMIT_ANNOTATION_LOAD = 1.0: 全件読み込む（通常モード）
# LIMIT_ANNOTATION_LOAD = 0.0: 制限なし（全件読み込む）
LIMIT_ANNOTATION_LOAD = 0.3  # 0.5に設定すると半分、1.0で全件、0.5未満でその割合だけ読み込む

# パス設定（参考ノートブックの方式に合わせる）
# https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
if IS_KAGGLE:
    # Kaggle環境では/kaggle/input/からデータセットを読み込む
    # 参考ノートブックの方式: 直接パス指定または自動検出
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        # データセットディレクトリを検索（test.csvやsample_submission.csvを含むディレクトリ）
        data_dirs = []
        for item in input_dir.iterdir():
            if item.is_dir():
                # test.csvまたはsample_submission.csvが存在するか確認
                if (item / 'test.csv').exists() or (item / 'sample_submission.csv').exists():
                    data_dirs.append(str(item))
        
        if data_dirs:
            DATA_DIR = data_dirs[0]  # 最初に見つかったディレクトリを使用
        else:
            # フォールバック: 最初のディレクトリを使用
            all_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            if all_dirs:
                DATA_DIR = str(all_dirs[0])
            else:
                DATA_DIR = '/kaggle/input/mabe-mouse-behavior-detection'
    else:
        DATA_DIR = '/kaggle/input/mabe-mouse-behavior-detection'
    OUTPUT_DIR = '/kaggle/working'
    MODEL_DIR = '/kaggle/working'
else:
    DATA_DIR = './data'
    OUTPUT_DIR = './output'
    MODEL_DIR = './models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"データディレクトリ: {DATA_DIR}")
print(f"出力ディレクトリ: {OUTPUT_DIR}")
print(f"モデル保存ディレクトリ: {MODEL_DIR}")


def get_memory_usage():
    """
    現在のメモリ使用量を取得（GB単位）
    
    Returns:
        dict: メモリ使用量の情報
    """
    if not PSUTIL_AVAILABLE:
        return {'rss': 0, 'vms': 0, 'percent': 0}
    
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss': mem_info.rss / (1024 ** 3),  # Resident Set Size (GB)
            'vms': mem_info.vms / (1024 ** 3),  # Virtual Memory Size (GB)
            'percent': process.memory_percent()
        }
    except Exception:
        return {'rss': 0, 'vms': 0, 'percent': 0}


def print_memory_usage(stage=""):
    """
    メモリ使用量を表示
    
    Args:
        stage: ステージ名（表示用）
    """
    if MEMORY_MONITORING and PSUTIL_AVAILABLE:
        mem = get_memory_usage()
        print(f"メモリ使用量 {stage}: RSS={mem['rss']:.2f}GB, VMS={mem['vms']:.2f}GB, Percent={mem['percent']:.1f}%")


def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES, lightweight=False):
    """
    1D CNNモデルの構築
    1DCNN_model_2.pyの実装を参照
    
    Args:
        input_shape: (sequence_length, num_features) のタプル
        num_classes: 分類クラス数
        lightweight: 軽量モデルを使用するかどうか（メモリ削減）
    
    Returns:
        Kerasモデル
    """
    inputs = layers.Input(shape=input_shape)
    
    # Conv1Dは (batch, sequence_length, features) の形式を期待
    # Permuteは使わず、そのままConv1Dを適用
    x = inputs
    
    # フィルター数を決定（軽量モードの場合は削減）
    if lightweight:
        filters1, filters2, filters3, filters4 = 32, 64, 128, 256
        dense1, dense2 = 128, 64
    else:
        filters1, filters2, filters3, filters4 = 64, 128, 256, 512
        dense1, dense2 = 256, 128
    
    sequence_length, num_features = input_shape
    
    # 第1畳み込みブロック
    x = layers.Conv1D(filters=filters1, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # sequence_lengthが2以上の場合のみpoolingを適用
    if sequence_length >= 2:
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第2畳み込みブロック
    x = layers.Conv1D(filters=filters2, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    # pooling後のsequence_lengthを計算（各poolingで半分になる）
    pooled_length = sequence_length // 2 if sequence_length >= 2 else sequence_length
    if pooled_length >= 2:
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第3畳み込みブロック
    x = layers.Conv1D(filters=filters3, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    pooled_length = pooled_length // 2 if pooled_length >= 2 else pooled_length
    if pooled_length >= 2:
        x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第4畳み込みブロック
    x = layers.Conv1D(filters=filters4, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    # 全結合層
    x = layers.Dense(dense1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(dense2, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # 出力層
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MABe_1DCNN')
    return model


class DataGenerator(Sequence):
    """
    メモリ効率的なデータジェネレータ（Keras Sequence継承）
    バッチごとにデータを読み込むことでメモリ使用量を削減
    """
    def __init__(self, video_ids, video_actions, data_dir, label_encoder, scaler, 
                 batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, 
                 annotation_file_paths=None, shuffle=True):
        """
        Args:
            video_ids: 動画IDのリスト
            video_actions: video_id -> action のマッピング
            data_dir: データディレクトリ
            label_encoder: LabelEncoderインスタンス
            scaler: StandardScalerインスタンス（fit済み）
            batch_size: バッチサイズ
            sequence_length: シーケンス長
            annotation_file_paths: video_id -> annotation_file_path のマッピング
            shuffle: エポックごとにシャッフルするかどうか
        """
        self.video_ids = video_ids
        self.video_actions = video_actions
        self.data_dir = data_dir
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.annotation_file_paths = annotation_file_paths or {}
        self.shuffle = shuffle
        self.indices = np.arange(len(video_ids))
        
        # 特徴量の次元数を事前に取得（最初の動画から）
        if len(video_ids) > 0:
            try:
                test_sequences, _ = load_tracking_data(
                    data_dir, video_ids[0], 
                    sequence_length=sequence_length,
                    annotation_file_path=self.annotation_file_paths.get(video_ids[0])
                )
                if len(test_sequences) > 0:
                    self.n_features = test_sequences.shape[1]
                else:
                    self.n_features = None
            except Exception:
                self.n_features = None
        else:
            self.n_features = None
    
    def __len__(self):
        """バッチ数を返す"""
        return int(np.ceil(len(self.video_ids) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        指定されたバッチのデータを読み込んで返す
        
        Args:
            idx: バッチインデックス
        
        Returns:
            tuple: (X_batch, y_batch)
        """
        # バッチのインデックスを取得
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_sequences = []
        batch_labels = []
        
        for video_idx in batch_indices:
            video_id = self.video_ids[video_idx]
            
            try:
                # 動画の行動を取得
                if video_id in self.video_actions:
                    action = self.video_actions[video_id]
                    label = self.label_encoder.transform([action])[0]
                else:
                    continue
                
                # アノテーションファイルのパスを取得
                annotation_file_path = self.annotation_file_paths.get(video_id, None)
                
                # 追跡データを読み込み
                sequences, _ = load_tracking_data(
                    self.data_dir, video_id, 
                    sequence_length=self.sequence_length,
                    annotation_file_path=annotation_file_path
                )
                
                # 各シーケンスに対応するラベルを設定
                for seq in sequences:
                    batch_sequences.append(seq)
                    batch_labels.append(label)
            
            except Exception as e:
                # エラーが発生した場合はスキップ
                continue
        
        if len(batch_sequences) == 0:
            # 空のバッチの場合はダミーデータを返す
            if self.n_features is not None:
                dummy_seq = np.zeros((self.sequence_length, self.n_features), dtype=np.float32)
                batch_sequences = [dummy_seq]
                batch_labels = [0]
            else:
                return None, None
        
        # 配列に変換（float32に変換してメモリ使用量を削減）
        X_batch = np.array(batch_sequences, dtype=np.float32)
        y_batch = np.array(batch_labels)
        
        # 正規化
        if self.scaler is not None:
            n_samples, seq_len, n_features = X_batch.shape
            X_reshaped = X_batch.reshape(-1, n_features)
            X_scaled = self.scaler.transform(X_reshaped).astype(np.float32)
            X_batch = X_scaled.reshape(n_samples, seq_len, n_features)
        
        # ガベージコレクション（メモリ解放）
        if ENABLE_GC:
            gc.collect()
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        """エポック終了時に呼ばれる（シャッフル）"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def extract_body_parts_from_columns(df, exclude_cols=None, max_body_parts=7):
    """
    カラム名からボディーパーツを抽出し、指定数のボディーパーツに関連するカラムを返す
    
    Args:
        df: 追跡データのDataFrame
        exclude_cols: 除外するカラム名のリスト
        max_body_parts: 選択するボディーパーツの最大数
    
    Returns:
        feature_cols: 選択されたボディーパーツに関連する特徴量カラム名のリスト
    """
    if exclude_cols is None:
        exclude_cols = ['frame', 'mouse_id', 'agent_id', 'target_id', 'video_id']
    
    # 数値型のカラムのみを選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # IDカラムを除外
    candidate_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(candidate_cols) == 0:
        return candidate_cols
    
    # ボディーパーツ名を抽出（カラム名のパターン: {bodypart}_{suffix}）
    body_parts = set()
    for col in candidate_cols:
        # アンダースコアで分割して最初の部分をボディーパーツ名として扱う
        parts = col.split('_')
        if len(parts) >= 2:
            # 最後の部分が suffix (x, y, likelihood など) であると仮定
            body_part = '_'.join(parts[:-1])
            body_parts.add(body_part)
    
    # ボディーパーツをソートして最大max_body_parts個を選択
    sorted_body_parts = sorted(body_parts)[:max_body_parts]
    
    # 選択されたボディーパーツに関連するカラムのみを抽出
    feature_cols = []
    for col in candidate_cols:
        for body_part in sorted_body_parts:
            if col.startswith(body_part + '_'):
                feature_cols.append(col)
                break
    
    # ボディーパーツが見つからない場合は、全てのカラムを使用
    if len(feature_cols) == 0:
        feature_cols = candidate_cols
    
    return feature_cols


def prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH):
    """
    データフレームから1D CNN用のシーケンスデータを準備
    
    Args:
        df: 追跡データのDataFrame
        sequence_length: シーケンス長
    
    Returns:
        sequences: シーケンスデータ (n_samples, sequence_length, n_features)
        feature_cols: 特徴量カラム名
    """
    # 数値カラムのみを抽出（IDカラムと文字列カラムを除外）
    exclude_cols = ['frame', 'mouse_id', 'agent_id', 'target_id', 'video_id']
    
    # ボディーパーツを7個に絞って特徴量カラムを選択
    feature_cols = extract_body_parts_from_columns(df, exclude_cols=exclude_cols, max_body_parts=7)
    
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found")
    
    features = df[feature_cols].values.astype(np.float32)
    
    # シーケンスを作成
    sequences = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
    
    if len(sequences) == 0:
        # データが少ない場合はパディング
        if len(features) < sequence_length:
            padding = np.zeros((sequence_length - len(features), len(feature_cols)), dtype=np.float32)
            features_padded = np.vstack([features.astype(np.float32), padding])
            sequences = [features_padded]
        else:
            sequences = [features[-sequence_length:].astype(np.float32)]
    
    return np.array(sequences, dtype=np.float32), feature_cols


def find_tracking_file(data_dir, video_id, annotation_file_path=None, debug=False):
    """
    追跡ファイルを検索（再帰的検索と柔軟なマッチングに対応）
    アノテーションファイルが見つかった場合、同じサブディレクトリ内で検索
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        annotation_file_path: アノテーションファイルのパス（オプション、同じサブディレクトリで検索するため）
        debug: デバッグ情報を表示するか
    
    Returns:
        tracking_file_path: 見つかったファイルパス、見つからない場合はNone
    """
    data_path = Path(data_dir)
    video_id_str = str(video_id)
    
    # アノテーションファイルのパスからサブディレクトリを取得
    annotation_subdir = None
    if annotation_file_path:
        annotation_path = Path(annotation_file_path)
        # train_annotation/CRIM13/xxx.parquet -> CRIM13
        if len(annotation_path.parts) >= 2:
            annotation_subdir = annotation_path.parts[-2]
            if debug:
                print(f"    アノテーションファイルのサブディレクトリ: {annotation_subdir}")
    
    # 可能なパスを試行（標準的なパス、CSVとParquetの両方）
    possible_paths = [
        data_path / 'train_tracking' / f'{video_id_str}.csv',
        data_path / 'train_tracking' / f'{video_id_str}.parquet',
        data_path / 'tracking' / f'{video_id_str}.csv',
        data_path / 'tracking' / f'{video_id_str}.parquet',
        data_path / 'test_tracking' / f'{video_id_str}.csv',
        data_path / 'test_tracking' / f'{video_id_str}.parquet',
        data_path / f'{video_id_str}.csv',
        data_path / f'{video_id_str}.parquet',
    ]
    
    # 再帰的にすべてのCSVファイルを検索
    tracking_dirs = ['train_tracking', 'tracking', 'test_tracking']
    for subdir_name in tracking_dirs:
        subdir = data_path / subdir_name
        if subdir.exists() and subdir.is_dir():
            # まず、サブディレクトリを確認
            subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            if debug and len(subdirs) > 0:
                print(f"    {subdir_name}のサブディレクトリ: {[d.name for d in subdirs[:5]]}")
            
            # サブディレクトリ内のファイルを直接検索
            for sub_subdir in subdirs:
                # アノテーションファイルのサブディレクトリと同じ場合、優先的に検索
                priority_search = (annotation_subdir and sub_subdir.name == annotation_subdir)
                
                # サブディレクトリ内のすべてのParquetファイルを検索
                for parquet_file in sub_subdir.glob('*.parquet'):
                    file_stem = parquet_file.stem
                    # マッチング条件
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if parquet_file not in possible_paths:
                            if priority_search:
                                # 優先サブディレクトリのファイルを先頭に追加
                                possible_paths.insert(0, parquet_file)
                            else:
                                possible_paths.append(parquet_file)
                
                # ファイル名が一致しない場合、ファイル内のvideo_idカラムを確認
                if priority_search and len([p for p in possible_paths if p.stem == video_id_str]) == 0:
                    for parquet_file in sub_subdir.glob('*.parquet'):
                        if parquet_file not in possible_paths:
                            try:
                                # ファイルを読み込んでvideo_idカラムを確認
                                df = pd.read_parquet(parquet_file)
                                if 'video_id' in df.columns:
                                    if video_id in df['video_id'].values:
                                        if debug:
                                            print(f"    ファイル内のvideo_idカラムで一致: {parquet_file.relative_to(data_path)}")
                                        possible_paths.insert(0, parquet_file)
                                        break
                            except Exception:
                                continue
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            if debug:
                print(f"  ✓ 追跡ファイルを発見: {path.relative_to(data_path)}")
            return path
    
    if debug:
        print(f"  ✗ 追跡ファイルが見つかりません: video_id={video_id}")
    
    return None


def find_annotation_file(data_dir, video_id, debug=False):
    """
    アノテーションファイルを検索（再帰的検索と柔軟なマッチングに対応）
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        debug: デバッグ情報を表示するか
    
    Returns:
        annotation_file_path: 見つかったファイルパス、見つからない場合はNone
    """
    data_path = Path(data_dir)
    video_id_str = str(video_id)
    
    # 可能なパスを試行（標準的なパス、CSVとParquetの両方）
    possible_paths = [
        data_path / 'train_annotation' / f'{video_id_str}.csv',
        data_path / 'train_annotation' / f'{video_id_str}.parquet',
        data_path / 'train_annotations' / f'{video_id_str}.csv',
        data_path / 'train_annotations' / f'{video_id_str}.parquet',
        data_path / 'annotation' / f'{video_id_str}.csv',
        data_path / 'annotation' / f'{video_id_str}.parquet',
        data_path / 'annotations' / f'{video_id_str}.csv',
        data_path / 'annotations' / f'{video_id_str}.parquet',
    ]
    
    # 再帰的にすべてのファイルを検索
    annotation_dirs = ['train_annotation', 'train_annotations', 'annotation', 'annotations']
    for subdir_name in annotation_dirs:
        subdir = data_path / subdir_name
        if subdir.exists() and subdir.is_dir():
            # まず、サブディレクトリを確認
            subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            if debug and len(subdirs) > 0:
                print(f"    {subdir_name}のサブディレクトリ: {[d.name for d in subdirs[:5]]}")
            
            # サブディレクトリ内のファイルを直接検索
            for sub_subdir in subdirs:
                # サブディレクトリ内のすべてのParquetファイルを検索
                for parquet_file in sub_subdir.glob('*.parquet'):
                    file_stem = parquet_file.stem
                    # マッチング条件（ファイル名が数値のみの場合も含む）
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if parquet_file not in possible_paths:
                            possible_paths.append(parquet_file)
            
            # Parquetファイルも再帰的に検索
            for parquet_file in subdir.rglob('*.parquet'):
                file_stem = parquet_file.stem
                # より柔軟なマッチング
                if file_stem == video_id_str:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                elif video_id_str in file_stem:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                elif any(video_id_str in part for part in file_stem.split('_')):
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                elif any(char.isdigit() for char in file_stem):
                    numbers_in_file = re.findall(r'\d+', file_stem)
                    if video_id_str in numbers_in_file:
                        if parquet_file not in possible_paths:
                            possible_paths.append(parquet_file)
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            if debug:
                print(f"  ✓ アノテーションファイルを発見: {path.relative_to(data_path)}")
            return path
    
    if debug:
        print(f"  ✗ アノテーションファイルが見つかりません: video_id={video_id}")
    
    return None


def normalize_annotation_columns(df, lab_id=None, debug=False):
    """
    研究所ごとに異なるアノテーションカラム名を統一する
    
    Args:
        df: アノテーションデータのDataFrame
        lab_id: 研究所ID（オプション、デバッグ用）
        debug: デバッグ情報を表示するか
    
    Returns:
        normalized_df: カラム名を統一したDataFrame
    """
    if df is None or len(df) == 0:
        return df
    
    df = df.copy()
    
    # カラム名のマッピング辞書（同様の意味を持つカラム名を統一）
    column_mapping = {
        # 行動/アクション関連
        'behavior': 'action',
        'behavior_type': 'action',
        'action_type': 'action',
        'behavior_label': 'action',
        'action_label': 'action',
        'label': 'action',
        
        # エージェントID関連
        'agent': 'agent_id',
        'mouse_id': 'agent_id',
        'subject_id': 'agent_id',
        
        # ターゲットID関連
        'target': 'target_id',
        'target_mouse_id': 'target_id',
        'object_id': 'target_id',
        
        # フレーム関連
        'start': 'start_frame',
        'start_time': 'start_frame',
        'begin_frame': 'start_frame',
        'stop': 'stop_frame',
        'end_frame': 'stop_frame',
        'end_time': 'stop_frame',
        
        # ビデオID関連
        'video': 'video_id',
        'video_name': 'video_id',
        'clip_id': 'video_id',
    }
    
    # カラム名を統一
    renamed_columns = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in column_mapping:
            new_name = column_mapping[col_lower]
            if col != new_name:
                renamed_columns[col] = new_name
        else:
            for key, value in column_mapping.items():
                if key in col_lower or col_lower in key:
                    new_name = value
                    if col != new_name:
                        renamed_columns[col] = new_name
                    break
    
    # カラム名を変更
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
        if debug:
            print(f"    カラム名を統一しました（lab_id: {lab_id}）:")
            for old_name, new_name in renamed_columns.items():
                print(f"      {old_name} -> {new_name}")
    
    # 重複カラムの処理（同じ意味のカラムが複数ある場合）
    if 'action' in df.columns and 'behavior' in df.columns:
        if df['behavior'].notna().sum() > df['action'].notna().sum():
            df['action'] = df['action'].fillna(df['behavior'])
        df = df.drop(columns=['behavior'])
        if debug:
            print(f"    'behavior'カラムを'action'に統合しました")
    
    return df


def load_annotation_data(data_dir, video_id, lab_id=None, debug=False):
    """
    指定された動画IDのアノテーションデータを読み込む（カラム名を統一）
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        lab_id: 研究所ID（オプション）
        debug: デバッグ情報を表示するか
    
    Returns:
        tuple: (annotation_df, annotation_file_path) または (None, None)
    """
    annotation_file = find_annotation_file(data_dir, video_id, debug=debug)
    
    if annotation_file is None:
        if debug:
            print(f"  ✗ アノテーションファイルが見つかりません: video_id={video_id}")
        return None, None
    
    try:
        # CSVまたはParquetファイルを読み込む
        if annotation_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(annotation_file)
        else:
            df = pd.read_csv(annotation_file)
        
        if debug:
            print(f"  ✓ アノテーションファイルを発見: {annotation_file}")
            print(f"    元のカラム: {list(df.columns)}")
            print(f"    行数: {len(df)}")
            print(f"    形式: {annotation_file.suffix}")
        
        # カラム名を統一
        df = normalize_annotation_columns(df, lab_id=lab_id, debug=debug)
        
        if debug:
            print(f"    統一後のカラム: {list(df.columns)}")
        
        return df, str(annotation_file)
    except Exception as e:
        if debug:
            print(f"  Warning: Error reading annotation file {annotation_file}: {e}")
        return None, None


def load_tracking_data(data_dir, video_id, sequence_length=SEQUENCE_LENGTH, annotation_file_path=None, debug=False):
    """
    指定された動画IDの追跡データを読み込んで1D CNN用に変換
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        sequence_length: シーケンス長
        annotation_file_path: アノテーションファイルのパス（オプション、同じサブディレクトリで検索するため）
        debug: デバッグ情報を表示するか
    
    Returns:
        sequences: シーケンスデータ
        feature_cols: 特徴量カラム名
    
    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    tracking_file = find_tracking_file(data_dir, video_id, annotation_file_path=annotation_file_path, debug=debug)
    
    if tracking_file is None:
        raise FileNotFoundError(f"Tracking file not found for video {video_id}")
    
    try:
        # CSVまたはParquetファイルを読み込む
        if tracking_file.suffix.lower() == '.parquet':
            df = pd.read_parquet(tracking_file)
        else:
            df = pd.read_csv(tracking_file)
        
        if len(df) == 0:
            raise ValueError(f"Empty tracking file: {tracking_file}")
        
        if debug:
            print(f"    読み込んだデータ: {len(df)}行, {len(df.columns)}カラム (形式: {tracking_file.suffix})")
        
        sequences, feature_cols = prepare_sequences_for_1dcnn(df, sequence_length)
        if debug:
            print(f"    生成されたシーケンス数: {len(sequences)}")
        return sequences, feature_cols
    except Exception as e:
        if debug:
            print(f"    エラー: {e}")
        raise Exception(f"Error reading tracking file {tracking_file}: {e}")


def create_1dcnn_features(train_df, data_dir, sequence_length=SEQUENCE_LENGTH, use_generator=None):
    """
    train_dfから1D CNN用の特徴量を生成（参考ノートブックの方式に合わせる）
    
    Args:
        train_df: 訓練データのDataFrame
        data_dir: データディレクトリ
        sequence_length: シーケンス長
        use_generator: ジェネレータを使用するかどうか（Noneの場合はUSE_GENERATORを使用）
    
    Returns:
        use_generator=Trueの場合: (train_generator, val_generator, label_encoder, scaler)
        use_generator=Falseの場合: (X_sequences, y_labels, label_encoder, scaler)
    """
    # use_generatorがNoneの場合はグローバル設定を使用
    if use_generator is None:
        use_generator = USE_GENERATOR
    
    print("1D CNN用の特徴量を生成中...")
    if use_generator:
        print("  メモリ効率モード: データジェネレータを使用します")
    
    print_memory_usage("特徴量生成開始前")
    
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    if not use_generator:
        all_sequences = []
        all_labels = []
    
    # ラベルの取得
    video_actions = {}
    
    if 'action' in train_df.columns:
        # train_dfにactionカラムがある場合
        for _, row in train_df.iterrows():
            video_id = row.get('video_id', '')
            if video_id:
                video_actions[video_id] = row['action']
    else:
        # アノテーションファイルから読み込む
        print("train_dfにactionカラムが見つかりません。アノテーションファイルから読み込みます...")
        
        video_ids = train_df['video_id'].unique() if 'video_id' in train_df.columns else []
        total_video_count = len(video_ids)
        print(f"処理する動画ID数（制限前）: {total_video_count}")
        
        # アノテーション読み込みを制限（テスト・デバッグ用）
        if LIMIT_ANNOTATION_LOAD > 0 and LIMIT_ANNOTATION_LOAD < 1.0:
            limit_count = int(total_video_count * LIMIT_ANNOTATION_LOAD)
            video_ids = video_ids[:limit_count]
            print(f"注意: アノテーション読み込みを{limit_count}件に制限しています（全{total_video_count}件の{LIMIT_ANNOTATION_LOAD*100:.0f}%）")
        
        print(f"処理する動画ID数: {len(video_ids)}")
        if len(video_ids) > 0:
            print(f"動画IDの例: {video_ids[:5]}")
        
        found_count = 0
        not_found_count = 0
        no_column_count = 0
        mab22_excluded_count = 0
        annotation_file_paths = {}  # video_id -> annotation_file_path のマッピング
        
        for idx, video_id in enumerate(tqdm(video_ids, desc="アノテーション読み込み", disable=not TQDM_AVAILABLE)):
            # lab_idを取得し、MAB22で始まる場合はスキップ
            lab_id = None
            if 'lab_id' in train_df.columns:
                video_row = train_df[train_df['video_id'] == video_id]
                if len(video_row) > 0:
                    lab_id = video_row.iloc[0]['lab_id']
                    # MAB22から始まる研究室のデータをスキップ
                    if pd.notna(lab_id) and str(lab_id).startswith('MAB22'):
                        mab22_excluded_count += 1
                        continue
            
            # 最初の3件はデバッグモードで表示
            debug_mode = (idx < 3)
            # アノテーションデータを読み込む（ファイルパスも取得）
            annotation_df, annotation_file_path = load_annotation_data(data_dir, video_id, lab_id=lab_id, debug=debug_mode)
            
            # アノテーションファイルのパスを保存（追跡ファイル検索で使用）
            if annotation_file_path:
                annotation_file_paths[video_id] = annotation_file_path
            
            if annotation_df is not None:
                # 統一された'action'カラムを使用
                if 'action' in annotation_df.columns:
                    action_counts = annotation_df['action'].value_counts()
                    most_common_action = action_counts.index[0]
                    video_actions[video_id] = most_common_action
                    found_count += 1
                else:
                    # actionカラムがない場合
                    no_column_count += 1
                    if debug_mode:
                        print(f"  Warning: video {video_id}のアノテーションファイルにactionカラムがありません")
                        print(f"    カラム: {list(annotation_df.columns)}")
            else:
                not_found_count += 1
        
        print(f"\nアノテーションファイル読み込み結果:")
        print(f"  見つかった（actionあり）: {found_count}/{len(video_ids)}")
        print(f"  ファイル未検出: {not_found_count}/{len(video_ids)}")
        print(f"  カラムなし（actionなし）: {no_column_count}/{len(video_ids)}")
        if mab22_excluded_count > 0:
            print(f"  MAB22研究室除外: {mab22_excluded_count}件")
    
    # train_dfにactionカラムがある場合、annotation_file_pathsは空のまま
    # ラベルをエンコード
    if len(video_actions) > 0:
        unique_actions = list(set(video_actions.values()))
        label_encoder.fit(unique_actions)
        print(f"見つかった行動クラス: {unique_actions}")
    else:
        # アノテーションファイルが見つからない場合のフォールバック処理
        print("\n警告: アノテーションデータが見つかりません。")
        print("代替方法を試行します...")
        
        # デフォルト行動を設定（フォールバック）
        print("\nデフォルトの行動を使用します。")
        default_action = 'unknown'
        video_ids = train_df['video_id'].unique() if 'video_id' in train_df.columns else []
        for video_id in video_ids:
            if video_id not in video_actions:
                video_actions[video_id] = default_action
        
        if len(video_actions) > 0:
            unique_actions = list(set(video_actions.values()))
            label_encoder.fit(unique_actions)
            print(f"デフォルト行動を使用: {unique_actions}")
        else:
            raise ValueError("No actions found in train_df or annotation files, and no video_ids available for default action")
    
    # 各動画のデータを処理
    video_ids = train_df['video_id'].unique() if 'video_id' in train_df.columns else []
    
    # MAB22から始まる研究室の動画を除外
    if 'lab_id' in train_df.columns:
        before_video_count = len(video_ids)
        video_ids_filtered = []
        for video_id in video_ids:
            video_row = train_df[train_df['video_id'] == video_id]
            if len(video_row) > 0:
                lab_id = video_row.iloc[0]['lab_id']
                if pd.notna(lab_id) and str(lab_id).startswith('MAB22'):
                    continue  # MAB22で始まる研究室のデータをスキップ
            video_ids_filtered.append(video_id)
        video_ids = video_ids_filtered
        after_video_count = len(video_ids)
        if before_video_count != after_video_count:
            print(f"MAB22研究室の動画を除外: {before_video_count - after_video_count}件（残り: {after_video_count}件）")
    
    print(f"処理する動画数: {len(video_ids)}")
    
    # アノテーションファイルのパスマッピングを取得（追跡ファイル検索で使用）
    # 既にアノテーション読み込み時に取得していない場合は、ここで取得
    if 'annotation_file_paths' not in locals() or len(annotation_file_paths) == 0:
        annotation_file_paths = {}
        print("アノテーションファイルのパスマッピングを取得中...")
        for video_id in tqdm(video_ids, desc="パスマッピング", disable=not TQDM_AVAILABLE):
            annotation_file = find_annotation_file(data_dir, video_id, debug=False)
            if annotation_file:
                annotation_file_paths[video_id] = str(annotation_file)
    
    # ジェネレータモードの場合
    if use_generator:
        print("\nデータジェネレータモード: 全データをメモリに読み込みません")
        
        # スカラーをフィットするためにサンプルデータを使用
        print("StandardScalerのフィット用にサンプルデータを読み込み中...")
        sample_size = min(100, len(video_ids))
        sample_video_ids = video_ids[:sample_size]
        sample_sequences = []
        
        for video_id in tqdm(sample_video_ids, desc="サンプルデータ読み込み", disable=not TQDM_AVAILABLE):
            try:
                annotation_file_path = annotation_file_paths.get(video_id, None)
                sequences, feature_cols = load_tracking_data(
                    data_dir, video_id, sequence_length, 
                    annotation_file_path=annotation_file_path, debug=False
                )
                sample_sequences.extend(sequences)
                if len(sample_sequences) >= 1000:  # 十分なサンプル数
                    break
            except Exception:
                continue
        
        if len(sample_sequences) > 0:
            # スカラーをフィット
            sample_array = np.array(sample_sequences)
            n_samples, seq_len, n_features = sample_array.shape
            X_reshaped = sample_array.reshape(-1, n_features)
            scaler.fit(X_reshaped)
            print(f"  StandardScalerをフィットしました（サンプル数: {len(sample_sequences)}）")
            del sample_array, sample_sequences
            if ENABLE_GC:
                gc.collect()
        
        print_memory_usage("スカラーフィット後")
        
        # 訓練・検証用のvideo_idsを分割
        train_video_ids, val_video_ids = train_test_split(
            video_ids, test_size=0.2, random_state=42
        )
        
        # ジェネレータを作成
        train_generator = DataGenerator(
            train_video_ids, video_actions, data_dir, label_encoder, scaler,
            batch_size=BATCH_SIZE, sequence_length=sequence_length,
            annotation_file_paths=annotation_file_paths, shuffle=True
        )
        
        val_generator = DataGenerator(
            val_video_ids, video_actions, data_dir, label_encoder, scaler,
            batch_size=BATCH_SIZE, sequence_length=sequence_length,
            annotation_file_paths=annotation_file_paths, shuffle=False
        )
        
        print(f"\nデータジェネレータを作成しました:")
        print(f"  訓練バッチ数: {len(train_generator)}")
        print(f"  検証バッチ数: {len(val_generator)}")
        print_memory_usage("ジェネレータ作成後")
        
        return train_generator, val_generator, label_encoder, scaler
    
    # 従来のモード（全データをメモリに読み込む）
    processed_videos = 0
    not_found_count = 0
    error_count = 0
    
    for idx, video_id in enumerate(tqdm(video_ids, desc="動画データ処理", disable=not TQDM_AVAILABLE)):
        try:
            # 該当する動画の行動を取得
            if video_id in video_actions:
                action = video_actions[video_id]
                label = label_encoder.transform([action])[0]
            else:
                continue
            
            # 最初の3件はデバッグモードで表示
            debug_mode = (idx < 3)
            if debug_mode:
                print(f"\n動画データ処理: video_id={video_id}")
            
            # アノテーションファイルのパスを取得（追跡ファイル検索で使用）
            # 既にアノテーションファイルを読み込んでいる場合は、そのパスを使用
            annotation_file_path = annotation_file_paths.get(video_id, None)
            if annotation_file_path is None:
                # キャッシュがない場合、アノテーションファイルのパスを取得
                annotation_file = find_annotation_file(data_dir, video_id, debug=False)
                annotation_file_path = str(annotation_file) if annotation_file else None
            
            # 追跡データを読み込み
            try:
                sequences, feature_cols = load_tracking_data(data_dir, video_id, sequence_length, annotation_file_path=annotation_file_path, debug=debug_mode)
            except FileNotFoundError as e:
                not_found_count += 1
                if debug_mode:
                    print(f"  ✗ 追跡ファイルが見つかりません: {e}")
                continue
            except Exception as e:
                error_count += 1
                if debug_mode:
                    print(f"  ✗ エラー: {e}")
                continue
            
            # 各シーケンスに対応するラベルを設定
            for seq_idx in range(len(sequences)):
                all_sequences.append(sequences[seq_idx])
                all_labels.append(label)
            
            processed_videos += 1
            if debug_mode:
                print(f"  ✓ 処理成功: {len(sequences)}シーケンス生成")
            
        except Exception as e:
            error_count += 1
            if idx < 3:
                print(f"Warning: Error processing video {video_id}: {e}")
            continue
    
    print(f"\n動画データ処理結果:")
    print(f"  処理成功: {processed_videos}/{len(video_ids)}")
    print(f"  ファイル未検出: {not_found_count}/{len(video_ids)}")
    print(f"  エラー: {error_count}/{len(video_ids)}")
    print(f"  生成されたシーケンス総数: {len(all_sequences)}")
    
    if len(all_sequences) == 0:
        # より詳細なエラーメッセージ
        error_msg = f"No sequences created. Failed to process all {len(video_ids)} videos.\n"
        error_msg += f"  処理成功: {processed_videos}, ファイル未検出: {not_found_count}, エラー: {error_count}\n"
        error_msg += f"  データディレクトリ: {data_dir}\n"
        error_msg += f"  確認したディレクトリ: train_tracking, tracking, test_tracking"
        raise ValueError(error_msg)
    
    # シーケンスデータを配列に変換（float32に変換してメモリ使用量を削減）
    print("シーケンスデータを配列に変換中...")
    X_sequences = np.array(all_sequences, dtype=np.float32)
    y_labels = np.array(all_labels)
    
    # 特徴量の正規化
    print("特徴量の正規化中...")
    n_samples, seq_len, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped).astype(np.float32)
    X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
    
    print(f"\n生成されたシーケンス数: {len(X_sequences)}")
    print(f"シーケンス形状: {X_sequences.shape}")
    print(f"ラベル数: {len(y_labels)}")
    
    return X_sequences, y_labels, label_encoder, scaler


def train_1dcnn_model(X_train=None, y_train=None, X_val=None, y_val=None, 
                     train_generator=None, val_generator=None,
                     input_shape=None, num_classes=NUM_CLASSES,
                     batch_size=BATCH_SIZE, epochs=EPOCHS,
                     learning_rate=LEARNING_RATE,
                     model_save_path=MODEL_DIR,
                     lightweight=False):
    """
    1D CNNモデルの訓練（ジェネレータ対応）
    
    Args:
        X_train, y_train: 訓練データ（ジェネレータ未使用時）
        X_val, y_val: 検証データ（ジェネレータ未使用時）
        train_generator: 訓練データジェネレータ（ジェネレータ使用時）
        val_generator: 検証データジェネレータ（ジェネレータ使用時）
        input_shape: 入力形状
        num_classes: クラス数
        batch_size: バッチサイズ
        epochs: エポック数
        learning_rate: 学習率
        model_save_path: モデル保存パス
        lightweight: 軽量モデルを使用するかどうか
    
    Returns:
        history: 訓練履歴
        model: 訓練済みモデル
    """
    use_generator = (train_generator is not None)
    
    if use_generator:
        # ジェネレータから入力形状を取得
        if input_shape is None:
            # 最初のバッチを取得して形状を確認
            X_sample, _ = train_generator[0]
            if X_sample is not None:
                input_shape = (X_sample.shape[1], X_sample.shape[2])
            else:
                raise ValueError("ジェネレータからサンプルを取得できませんでした")
    else:
        # 通常モードの場合、X_trainから形状を取得
        if input_shape is None and X_train is not None:
            input_shape = (X_train.shape[1], X_train.shape[2])
        elif input_shape is None:
            raise ValueError("input_shapeが指定されていません。ジェネレータを使用するか、input_shapeを指定してください")
    
    print_memory_usage("モデル構築前")
    
    # モデルの構築
    model = build_1dcnn_model(input_shape, num_classes, lightweight=lightweight)
    
    # コールバックの設定
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_1dcnn_model_4.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # モデルのコンパイル
    # Top-3精度メトリクスをカスタムで定義
    def top_3_accuracy(y_true, y_pred):
        import tensorflow as tf
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', top_3_accuracy]
    )
    
    # モデル構造の表示
    if IS_KAGGLE:
        print(f"モデル入力形状: {input_shape}")
        print(f"モデル出力クラス数: {num_classes}")
        print(f"総パラメータ数: {model.count_params():,}")
    else:
        model.summary()
    
    # 訓練の実行
    print(f"\n訓練開始: バッチサイズ={batch_size}, エポック数={epochs}")
    if use_generator:
        print(f"訓練データ: ジェネレータモード（{len(train_generator)}バッチ）")
        print(f"検証データ: ジェネレータモード（{len(val_generator)}バッチ）")
    else:
        print(f"訓練データサイズ: {X_train.shape[0]:,}サンプル")
        print(f"検証データサイズ: {X_val.shape[0]:,}サンプル")
    
    print_memory_usage("訓練開始前")
    
    train_start_time = time.time()
    
    if use_generator:
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    train_elapsed = time.time() - train_start_time
    
    # メモリ解放
    if ENABLE_GC:
        gc.collect()
    
    print_memory_usage("訓練完了後")
    
    print(f"\n訓練完了!")
    print(f"訓練時間: {timedelta(seconds=int(train_elapsed))}")
    print(f"最終エポック: {len(history.history['loss'])}")
    if 'val_loss' in history.history:
        print(f"最終検証損失: {history.history['val_loss'][-1]:.4f}")
        print(f"最終検証精度: {history.history['val_accuracy'][-1]:.4f}")
    
    return history, model


def save_model_and_preprocessors(model, scaler, label_encoder, model_dir=MODEL_DIR):
    """
    学習済みモデルと前処理器を保存する
    
    Args:
        model: 訓練済みモデル
        scaler: StandardScaler
        label_encoder: LabelEncoder
        model_dir: 保存先ディレクトリ
    """
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # モデルを保存
    model_file = model_path / 'best_1dcnn_model_4.h5'
    model.save(str(model_file))
    print(f"✓ モデルを保存しました: {model_file}")
    
    # StandardScalerを保存
    if scaler is not None:
        scaler_file = model_path / 'scaler.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ StandardScalerを保存しました: {scaler_file}")
    
    # LabelEncoderを保存
    if label_encoder is not None:
        label_encoder_file = model_path / 'label_encoder.pkl'
        with open(label_encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"✓ LabelEncoderを保存しました: {label_encoder_file}")


def verify_data_access(train_df, data_dir):
    """
    データアクセスを検証する（学習前の確認用）
    
    Args:
        train_df: 訓練データのDataFrame
        data_dir: データディレクトリ
    
    Returns:
        dict: 検証結果
    """
    print("\n" + "=" * 60)
    print("データアクセス検証")
    print("=" * 60)
    
    verification_results = {
        'total_videos': 0,
        'annotation_found': 0,
        'tracking_found': 0,
        'both_found': 0,
        'neither_found': 0,
        'sample_videos': []
    }
    
    if 'video_id' not in train_df.columns:
        print("エラー: train_dfに'video_id'カラムがありません")
        return verification_results
    
    video_ids = train_df['video_id'].unique()
    verification_results['total_videos'] = len(video_ids)
    
    print(f"検証する動画数: {len(video_ids)}")
    
    # サンプル動画（最初の10件）を詳細に検証
    sample_size = min(10, len(video_ids))
    sample_video_ids = video_ids[:sample_size]
    
    print(f"\nサンプル動画（最初の{sample_size}件）の詳細検証:")
    
    for idx, video_id in enumerate(tqdm(sample_video_ids, desc="データ検証", disable=not TQDM_AVAILABLE)):
        annotation_found = False
        tracking_found = False
        
        # アノテーションファイルの検索
        lab_id = None
        if 'lab_id' in train_df.columns:
            video_row = train_df[train_df['video_id'] == video_id]
            if len(video_row) > 0:
                lab_id = video_row.iloc[0]['lab_id']
        
        annotation_df, annotation_path = load_annotation_data(
            data_dir, video_id, lab_id=lab_id, debug=(idx < 3)
        )
        
        if annotation_df is not None and 'action' in annotation_df.columns:
            annotation_found = True
            verification_results['annotation_found'] += 1
        
        # 追跡ファイルの検索
        tracking_file = find_tracking_file(
            data_dir, video_id, annotation_file_path=annotation_path, debug=(idx < 3)
        )
        
        if tracking_file is not None:
            tracking_found = True
            verification_results['tracking_found'] += 1
        
        if annotation_found and tracking_found:
            verification_results['both_found'] += 1
        elif not annotation_found and not tracking_found:
            verification_results['neither_found'] += 1
        
        # サンプル情報を保存
        if idx < 5:
            verification_results['sample_videos'].append({
                'video_id': video_id,
                'lab_id': lab_id,
                'annotation_found': annotation_found,
                'tracking_found': tracking_found,
                'annotation_path': str(annotation_path) if annotation_path else None,
                'tracking_path': str(tracking_file) if tracking_file else None
            })
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("検証結果サマリー")
    print("=" * 60)
    print(f"総動画数: {verification_results['total_videos']}")
    print(f"アノテーションファイルが見つかった動画: {verification_results['annotation_found']}/{sample_size}")
    print(f"追跡ファイルが見つかった動画: {verification_results['tracking_found']}/{sample_size}")
    print(f"両方見つかった動画: {verification_results['both_found']}/{sample_size}")
    print(f"両方見つからなかった動画: {verification_results['neither_found']}/{sample_size}")
    
    if verification_results['sample_videos']:
        print("\nサンプル動画の詳細:")
        for sample in verification_results['sample_videos']:
            print(f"  video_id={sample['video_id']}, lab_id={sample['lab_id']}")
            print(f"    アノテーション: {'✓' if sample['annotation_found'] else '✗'}")
            print(f"    追跡データ: {'✓' if sample['tracking_found'] else '✗'}")
            if sample['annotation_path']:
                print(f"    アノテーションパス: {sample['annotation_path']}")
            if sample['tracking_path']:
                print(f"    追跡パス: {sample['tracking_path']}")
    
    return verification_results


def main():
    """
    メイン実行関数（データアクセス確認まで）
    """
    total_start_time = time.time()
    
    print("=" * 60)
    print("MABe Mouse Behavior Detection - 1D CNN (学習用)")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kaggle環境: {IS_KAGGLE}")
    print(f"データディレクトリ: {DATA_DIR}")
    
    # GPU情報を表示
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU検出: {len(gpus)}基")
        else:
            print("GPU: 検出されませんでした（CPUで実行）")
    except Exception as e:
        print(f"GPU情報取得エラー: {e}")
    
    # データの読み込み
    try:
        print("\n[1/3] データの読み込み")
        data_load_start = time.time()
        
        train_csv = Path(DATA_DIR) / 'train.csv'
        
        # 代替パスも試行
        if not train_csv.exists():
            for subdir in ['train', 'data']:
                alt_path = Path(DATA_DIR) / subdir / 'train.csv'
                if alt_path.exists():
                    train_csv = alt_path
                    break
        
        if not train_csv.exists():
            print(f"エラー: train.csvが見つかりません: {train_csv}")
            print(f"データディレクトリの内容を確認中: {DATA_DIR}")
            data_path = Path(DATA_DIR)
            if data_path.exists():
                files = list(data_path.glob('*.csv'))
                dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
                print(f"  見つかったCSVファイル: {[f.name for f in files[:5]]}")
                print(f"  見つかったディレクトリ: {dirs[:5]}")
            return
        
        # データ読み込み制限の適用
        if LIMIT_TRAIN_ROWS == 1:
            train_df = pd.read_csv(train_csv, nrows=LIMIT_TRAIN_ROWS_COUNT)
            print(f"注意: データ読み込みを最初の{LIMIT_TRAIN_ROWS_COUNT}行に制限しています（テスト・デバッグ用）")
        else:
            train_df = pd.read_csv(train_csv)
        
        print(f"訓練データ（読み込み後）: {len(train_df)}行")
        
        # MAB22から始まる研究室のデータを除外
        if 'lab_id' in train_df.columns:
            before_count = len(train_df)
            train_df = train_df[~train_df['lab_id'].astype(str).str.startswith('MAB22', na=False)]
            after_count = len(train_df)
            excluded_count = before_count - after_count
            if excluded_count > 0:
                print(f"MAB22から始まる研究室のデータを除外: {excluded_count}行（残り: {after_count}行）")
            else:
                print(f"MAB22から始まる研究室のデータは見つかりませんでした")
        else:
            print("lab_idカラムが見つかりません。MAB22のフィルタリングはスキップします")
        
        print(f"訓練データ: {len(train_df)}行")
        print(f"訓練データのカラム: {list(train_df.columns)[:10]}")
        print(f"データ読み込み時間: {timedelta(seconds=int(time.time() - data_load_start))}")
        
        # データディレクトリ構造の確認
        print("\n[2/3] データディレクトリ構造の確認")
        data_path = Path(DATA_DIR)
        if data_path.exists():
            # すべてのサブディレクトリを確認
            all_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            print(f"見つかったディレクトリ数: {len(all_dirs)}")
            print(f"ディレクトリ一覧: {[d.name for d in all_dirs[:20]]}")
            
            # すべてのCSVファイルとParquetファイルを再帰的に検索
            all_csv_files = list(data_path.rglob('*.csv'))
            all_parquet_files = list(data_path.rglob('*.parquet'))
            print(f"\n見つかったCSVファイル数（再帰的）: {len(all_csv_files)}")
            print(f"見つかったParquetファイル数（再帰的）: {len(all_parquet_files)}")
            
            # 特定のディレクトリの詳細確認
            tracking_dirs = ['train_tracking', 'tracking', 'test_tracking']
            annotation_dirs = ['train_annotation', 'train_annotations', 'annotation', 'annotations']
            
            print("\n追跡データディレクトリの詳細:")
            for dir_name in tracking_dirs:
                dir_path = data_path / dir_name
                if dir_path.exists():
                    csv_files = list(dir_path.rglob('*.csv'))
                    parquet_files = list(dir_path.rglob('*.parquet'))
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    print(f"  {dir_name}:")
                    print(f"    CSVファイル数（再帰的）: {len(csv_files)}")
                    print(f"    Parquetファイル数（再帰的）: {len(parquet_files)}")
                    print(f"    サブディレクトリ数: {len(subdirs)}")
                    if len(subdirs) > 0:
                        print(f"    サブディレクトリ例: {[d.name for d in subdirs[:5]]}")
            
            print("\nアノテーションディレクトリの詳細:")
            for dir_name in annotation_dirs:
                dir_path = data_path / dir_name
                if dir_path.exists():
                    csv_files = list(dir_path.rglob('*.csv'))
                    parquet_files = list(dir_path.rglob('*.parquet'))
                    subdirs = [d for d in dir_path.iterdir() if d.is_dir()]
                    print(f"  {dir_name}:")
                    print(f"    CSVファイル数（再帰的）: {len(csv_files)}")
                    print(f"    Parquetファイル数（再帰的）: {len(parquet_files)}")
                    print(f"    サブディレクトリ数: {len(subdirs)}")
                    if len(subdirs) > 0:
                        print(f"    サブディレクトリ例: {[d.name for d in subdirs[:5]]}")
        else:
            print(f"警告: データディレクトリが存在しません: {data_path}")
        
        # データアクセスの検証（オプション）
        print("\n[3/5] データアクセスの検証（オプション）")
        verification_results = verify_data_access(train_df, DATA_DIR)
        
        # 検証結果を保存
        verification_file = Path(MODEL_DIR) / 'data_verification_results.json'
        with open(verification_file, 'w', encoding='utf-8') as f:
            json.dump(verification_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n検証結果を保存しました: {verification_file}")
        
        # メモリ使用量に基づいてバッチサイズとシーケンス長を調整
        effective_batch_size = BATCH_SIZE
        effective_sequence_length = SEQUENCE_LENGTH
        
        if MEMORY_EFFICIENT_MODE:
            mem = get_memory_usage()
            if mem['rss'] > 8.0:  # 8GB以上使用している場合
                print(f"\nメモリ使用量が高いため、バッチサイズとシーケンス長を削減します")
                effective_batch_size = REDUCED_BATCH_SIZE
                effective_sequence_length = REDUCED_SEQUENCE_LENGTH
                print(f"  バッチサイズ: {BATCH_SIZE} → {effective_batch_size}")
                print(f"  シーケンス長: {SEQUENCE_LENGTH} → {effective_sequence_length}")
        
        # 1D CNN用の特徴量を生成
        print("\n[4/5] 1D CNN用の特徴量生成")
        feature_start = time.time()
        
        features_result = create_1dcnn_features(
            train_df, DATA_DIR, sequence_length=effective_sequence_length,
            use_generator=USE_GENERATOR
        )
        
        print(f"特徴量生成時間: {timedelta(seconds=int(time.time() - feature_start))}")
        
        # ジェネレータモードかどうかで処理を分岐
        if USE_GENERATOR and isinstance(features_result[0], DataGenerator):
            train_generator, val_generator, label_encoder, scaler = features_result
            
            # モデルの訓練（ジェネレータモード）
            print("\n[5/5] 1D CNNモデルの訓練（ジェネレータモード）")
            # 入力形状を取得（ジェネレータから）
            X_sample, _ = train_generator[0]
            input_shape = (X_sample.shape[1], X_sample.shape[2])
            
            # 軽量モデルを使用するかどうか（メモリ不足時）
            use_lightweight = MEMORY_EFFICIENT_MODE and get_memory_usage()['rss'] > 6.0
            
            history, model = train_1dcnn_model(
                train_generator=train_generator,
                val_generator=val_generator,
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                batch_size=effective_batch_size,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                lightweight=use_lightweight
            )
        else:
            # 従来のモード
            X_sequences, y_labels, label_encoder, scaler = features_result
            
            # 訓練・検証セットに分割
            print("\n訓練・検証セットに分割中...")
            split_start = time.time()
            X_train, X_val, y_train, y_val = train_test_split(
                X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
            )
            print(f"分割時間: {timedelta(seconds=int(time.time() - split_start))}")
            
            print(f"訓練セット: {X_train.shape}, {y_train.shape}")
            print(f"検証セット: {X_val.shape}, {y_val.shape}")
            
            # メモリ解放
            del X_sequences, y_labels
            if ENABLE_GC:
                gc.collect()
            
            # モデルの訓練
            print("\n[5/5] 1D CNNモデルの訓練")
            input_shape = (effective_sequence_length, X_train.shape[2])
            
            # 軽量モデルを使用するかどうか（メモリ不足時）
            use_lightweight = MEMORY_EFFICIENT_MODE and get_memory_usage()['rss'] > 6.0
            
            history, model = train_1dcnn_model(
                X_train, y_train, X_val, y_val,
                input_shape=input_shape,
                num_classes=NUM_CLASSES,
                batch_size=effective_batch_size,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                lightweight=use_lightweight
            )
            
            # メモリ解放
            del X_train, X_val, y_train, y_val
            if ENABLE_GC:
                gc.collect()
        
        print("✓ モデルの訓練が完了しました")
        
        # モデルと前処理器の保存
        print("\nモデルと前処理器の保存")
        save_model_and_preprocessors(model, scaler, label_encoder, MODEL_DIR)
        
        # 総実行時間を表示
        total_elapsed = time.time() - total_start_time
        print("\n" + "=" * 60)
        print("学習完了")
        print("=" * 60)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {timedelta(seconds=int(total_elapsed))}")
        print("=" * 60)
        print("\n次のステップ:")
        print("  1. 学習済みモデルが保存されました")
        print("  2. 1DCNN_submit_4.pyで推論を実行してください")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

