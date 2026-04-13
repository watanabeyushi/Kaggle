"""
MABe Mouse Behavior Detection - 1D CNN実装（Extra Trees GPUベース）
Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
参考: https://www.kaggle.com/code/mattiaangeli/mabe-extra-trees-gpu
既存実装参照: 1DCNN_model.py
"""

import os
import warnings
import sys
from contextlib import redirect_stderr
from io import StringIO

# TensorFlowとprotobufの互換性問題の警告を抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*MessageFactory.*GetPrototype.*')
warnings.filterwarnings('ignore', message='.*Unable to register.*factory.*')

# protobufエラーを完全に抑制
_stderr_buffer = StringIO()
with redirect_stderr(_stderr_buffer):
    try:
        import google.protobuf
        # protobufの互換性問題を回避
        if hasattr(google.protobuf, 'message'):
            pass
    except (ImportError, AttributeError, TypeError):
        pass
    except Exception:
        # その他のprotobuf関連エラーも無視
        pass

import numpy as np
import pandas as pd
import time
import re
from datetime import datetime, timedelta

# 進捗バーライブラリ（Kaggle環境では利用可能）
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

_stderr_buffer = StringIO()
with redirect_stderr(_stderr_buffer):
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except Exception:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Kaggle環境の検出
IS_KAGGLE = os.path.exists('/kaggle/input')

# ファイル検索のキャッシュ（効率化のため）
_file_cache = {
    'tracking': {},  # {subdir: [list of csv files]}
    'annotation': {},  # {subdir: [list of csv files]}
    'file_patterns': {}  # {subdir: pattern_info}
}

# 実行状況表示用のユーティリティ関数
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
    else:
        print(f"\n{title}")

def print_progress(message, start_time=None):
    """進捗メッセージを表示（経過時間付き）"""
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f"[{elapsed_str}] {message}")
    else:
        print(message)

def get_system_info():
    """システム情報を取得して表示"""
    info = []
    info.append("=" * 60)
    info.append("システム情報")
    info.append("=" * 60)
    
    # Kaggle環境
    info.append(f"Kaggle環境: {IS_KAGGLE}")
    
    # GPU情報
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info.append(f"GPU検出: {len(gpus)}基")
            for i, gpu in enumerate(gpus):
                info.append(f"  GPU {i}: {gpu.name}")
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if 'device_name' in gpu_details:
                        info.append(f"    デバイス名: {gpu_details['device_name']}")
                except:
                    pass
        else:
            info.append("GPU: 検出されませんでした（CPUで実行）")
    except Exception as e:
        info.append(f"GPU情報取得エラー: {e}")
    
    # TensorFlowバージョン
    try:
        info.append(f"TensorFlowバージョン: {tf.__version__}")
    except:
        pass
    
    # メモリ情報（可能な場合）
    try:
        import psutil
        memory = psutil.virtual_memory()
        info.append(f"利用可能メモリ: {memory.available / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB")
    except ImportError:
        pass
    
    info.append("=" * 60)
    return "\n".join(info)

# モデルパラメータ（Kaggle提出要件: 9時間以内の実行時間を考慮）
SEQUENCE_LENGTH = 64
NUM_CLASSES = 15
BATCH_SIZE = 64  # 実行時間短縮のためバッチサイズを増加
EPOCHS = 50  # 実行時間短縮のためエポック数を削減（EarlyStoppingで調整）
LEARNING_RATE = 0.001
# 早期停止のパラメータ（実行時間を考慮して調整）
EARLY_STOPPING_PATIENCE = 10  # 15から10に削減
REDUCE_LR_PATIENCE = 5

# パス設定（Extra Trees GPUノートブックと同じ方式）
# Extra Trees GPUノートブックでは通常、データセット名を直接指定
if IS_KAGGLE:
    # Kaggle環境では/kaggle/input/からデータセットを読み込む
    # データセット名は環境に応じて自動検出
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        # データセットディレクトリを検索
        data_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if data_dirs:
            DATA_DIR = str(data_dirs[0])  # 最初に見つかったディレクトリを使用
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


def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES):
    """
    1D CNNモデルの構築
    1DCNN_model.pyの実装を参照
    
    Args:
        input_shape: (sequence_length, num_features) のタプル
        num_classes: 分類クラス数
    
    Returns:
        Kerasモデル
    """
    inputs = layers.Input(shape=input_shape)
    
    # 入力の形状を調整: (batch, sequence_length, features) -> (batch, features, sequence_length)
    x = layers.Permute((2, 1))(inputs)
    
    # 第1畳み込みブロック
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第2畳み込みブロック
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第3畳み込みブロック
    x = layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)
    
    # 第4畳み込みブロック
    x = layers.Conv1D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    # 全結合層
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # 出力層
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='MABe_1DCNN')
    return model


def prepare_sequences_for_1dcnn(df, sequence_length=SEQUENCE_LENGTH):
    """
    Extra Trees GPUのデータフレームから1D CNN用のシーケンスデータを準備
    
    Args:
        df: 追跡データのDataFrame
        sequence_length: シーケンス長
    
    Returns:
        sequences: シーケンスデータ (n_samples, sequence_length, n_features)
    """
    # 数値カラムのみを抽出（frame, mouse_idなどのIDカラムを除外）
    feature_cols = [col for col in df.columns 
                    if col not in ['frame', 'mouse_id', 'agent_id', 'target_id', 'video_id']]
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")
    
    features = df[feature_cols].values
    
    # シーケンスを作成
    sequences = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
    
    if len(sequences) == 0:
        # データが少ない場合はパディング
        if len(features) < sequence_length:
            padding = np.zeros((sequence_length - len(features), len(feature_cols)))
            features_padded = np.vstack([features, padding])
            sequences = [features_padded]
        else:
            sequences = [features[-sequence_length:]]
    
    return np.array(sequences), feature_cols


def extract_numbers_from_filename(filename):
    """
    ファイル名から数値を抽出する（柔軟なマッチングのため）
    
    Args:
        filename: ファイル名（拡張子なし）
    
    Returns:
        numbers: 抽出された数値のリスト
    """
    # ファイル名からすべての数値を抽出
    numbers = re.findall(r'\d+', filename)
    return [int(n) for n in numbers] if numbers else []


def get_cached_csv_files(tracking_dir, cache_key='tracking'):
    """
    サブディレクトリごとのCSVファイルリストをキャッシュから取得または生成
    
    Args:
        tracking_dir: 検索対象ディレクトリ
        cache_key: キャッシュキー（'tracking' または 'annotation'）
    
    Returns:
        csv_files_by_subdir: {subdir_path: [csv_files]} の辞書
    """
    global _file_cache
    
    cache_dir_str = str(tracking_dir)
    if cache_dir_str not in _file_cache[cache_key]:
        csv_files_by_subdir = {}
        
        if tracking_dir.exists() and tracking_dir.is_dir():
            # サブディレクトリを取得
            subdirs = [d for d in tracking_dir.iterdir() if d.is_dir()]
            
            for subdir in subdirs:
                # サブディレクトリ内のCSVファイルを再帰的に検索（CSVファイルのみ）
                csv_files = [f for f in subdir.rglob('*.csv') if f.suffix.lower() == '.csv']
                csv_files_by_subdir[str(subdir)] = csv_files
            
            # ルートディレクトリ直下のCSVファイルも検索（CSVファイルのみ）
            root_csv_files = [f for f in tracking_dir.glob('*.csv') if f.suffix.lower() == '.csv']
            if root_csv_files:
                csv_files_by_subdir[str(tracking_dir)] = root_csv_files
        
        _file_cache[cache_key][cache_dir_str] = csv_files_by_subdir
    
    return _file_cache[cache_key][cache_dir_str]


def match_video_id_to_file(video_id, file_path):
    """
    video_idとファイル名を柔軟にマッチング
    
    Args:
        video_id: 動画ID（数値または文字列）
        file_path: ファイルパス
    
    Returns:
        bool: マッチする場合True
    """
    video_id_str = str(video_id)
    file_stem = file_path.stem
    
    # 1. 完全一致
    if file_stem == video_id_str:
        return True
    
    # 2. 部分一致（video_idがファイル名に含まれる、またはその逆）
    if video_id_str in file_stem or file_stem in video_id_str:
        return True
    
    # 3. 数値抽出によるマッチング
    file_numbers = extract_numbers_from_filename(file_stem)
    try:
        video_id_int = int(video_id)
        if video_id_int in file_numbers:
            return True
        # ファイル名の数値がvideo_idと一致する場合
        if len(file_numbers) == 1 and file_numbers[0] == video_id_int:
            return True
    except (ValueError, TypeError):
        pass
    
    # 4. ファイル名パターンマッチング（例: tracking_44566106.csv, 44566106_tracking.csv）
    patterns = [
        f'{video_id_str}',
        f'tracking_{video_id_str}',
        f'{video_id_str}_tracking',
        f'video_{video_id_str}',
        f'{video_id_str}_video',
    ]
    for pattern in patterns:
        if pattern in file_stem:
            return True
    
    return False


def load_tracking_data_for_1dcnn(data_dir, video_id, sequence_length=SEQUENCE_LENGTH, debug=False):
    """
    指定された動画IDの追跡データを読み込んで1D CNN用に変換（再帰的検索対応・エラーハンドリング強化・キャッシュ対応）
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        sequence_length: シーケンス長
        debug: デバッグ情報を表示するか
    
    Returns:
        sequences: シーケンスデータ
        feature_cols: 特徴量カラム名
    
    Raises:
        FileNotFoundError: ファイルが見つからない場合
    """
    data_path = Path(data_dir)
    video_id_str = str(video_id)
    
    # 複数の可能なパスを試行（直接パス）
    possible_paths = [
        data_path / 'train_tracking' / f'{video_id}.csv',
        data_path / 'tracking' / f'{video_id}.csv',
        data_path / f'{video_id}.csv',
    ]
    
    # 再帰的にファイルを検索
    tracking_dirs = [
        data_path / 'train_tracking',
        data_path / 'tracking',
    ]
    
    for tracking_dir in tracking_dirs:
        if not tracking_dir.exists() or not tracking_dir.is_dir():
            continue
        
        # キャッシュからCSVファイルリストを取得
        csv_files_by_subdir = get_cached_csv_files(tracking_dir, cache_key='tracking')
        
        # デバッグ情報を表示
        total_csv_files = sum(len(files) for files in csv_files_by_subdir.values())
        if debug:
            print(f"  {tracking_dir}内のCSVファイル数（キャッシュ）: {total_csv_files}")
            if total_csv_files > 0:
                # 最初のサブディレクトリのファイル例を表示
                for subdir_str, csv_files in list(csv_files_by_subdir.items())[:1]:
                    if len(csv_files) > 0:
                        print(f"    例（{Path(subdir_str).name}）: {[f.name for f in csv_files[:3]]}")
        
        # 各サブディレクトリのCSVファイルをチェック
        for subdir_str, csv_files in csv_files_by_subdir.items():
            for csv_file in csv_files:
                if match_video_id_to_file(video_id, csv_file):
                    possible_paths.append(csv_file)
        
        # サブディレクトリ内も直接検索（例: CRIM13/video_id.csv）
        subdirs = [d for d in tracking_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            subdir_file = subdir / f'{video_id}.csv'
            if subdir_file.exists():
                possible_paths.append(subdir_file)
    
    if debug:
        print(f"  動画ID {video_id}の追跡ファイルを検索中...")
        print(f"  試行するパス数: {len(possible_paths)}")
        if len(possible_paths) > 0 and len(possible_paths) <= 10:
            print(f"    試行パス例: {[str(p.relative_to(data_path)) for p in possible_paths[:5]]}")
    
    for tracking_path in possible_paths:
        # CSVファイルのみを対象とする
        if tracking_path.exists() and tracking_path.suffix.lower() == '.csv':
            try:
                if debug:
                    print(f"  ✓ 追跡ファイルを発見: {tracking_path}")
                df = pd.read_csv(tracking_path)
                if len(df) == 0:
                    if debug:
                        print(f"  Warning: ファイルが空です: {tracking_path}")
                    continue
                sequences, feature_cols = prepare_sequences_for_1dcnn(df, sequence_length)
                return sequences, feature_cols
            except Exception as e:
                if debug:
                    print(f"  Warning: Error reading tracking file {tracking_path}: {e}")
                continue
    
    # ファイルが見つからない場合の詳細情報
    error_msg = f"Tracking file not found for video {video_id}"
    if debug:
        print(f"  ✗ 追跡ファイルが見つかりませんでした: {video_id}")
        # ディレクトリ構造を確認
        for tracking_dir in tracking_dirs:
            if tracking_dir.exists():
                csv_files_by_subdir = get_cached_csv_files(tracking_dir, cache_key='tracking')
                subdirs = list(csv_files_by_subdir.keys())
                if len(subdirs) > 0:
                    print(f"    {tracking_dir}のサブディレクトリ数: {len(subdirs)}")
                    # 最初のサブディレクトリのファイル例を表示
                    for subdir_str in list(subdirs)[:3]:
                        subdir = Path(subdir_str)
                        csv_files = csv_files_by_subdir[subdir_str]
                        if len(csv_files) > 0:
                            print(f"      {subdir.name}内のCSVファイル例: {[f.name for f in csv_files[:3]]}")
    
    raise FileNotFoundError(error_msg)


def load_annotations_for_video(data_dir, video_id, debug=False):
    """
    指定された動画IDのアノテーションデータを読み込む（キャッシュ対応・柔軟なマッチング）
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        debug: デバッグ情報を表示するか
    
    Returns:
        annotations: アノテーションデータのDataFrame、見つからない場合はNone
    """
    data_path = Path(data_dir)
    video_id_str = str(video_id)
    
    # 複数の可能なパスを試行（CSVファイルのみ）
    possible_paths = [
        data_path / 'train_annotation' / f'{video_id}.csv',
        data_path / 'train_annotations' / f'{video_id}.csv',
        data_path / 'annotation' / f'{video_id}.csv',
        data_path / 'annotations' / f'{video_id}.csv',
    ]
    
    # ディレクトリ内のすべてのCSVファイルも検索
    annotation_dirs = [
        data_path / 'train_annotation',
        data_path / 'train_annotations',
        data_path / 'annotation',
        data_path / 'annotations',
    ]
    
    for annotation_dir in annotation_dirs:
        if not annotation_dir.exists() or not annotation_dir.is_dir():
            continue
        
        # キャッシュからCSVファイルリストを取得
        csv_files_by_subdir = get_cached_csv_files(annotation_dir, cache_key='annotation')
        
        # デバッグ情報を表示
        total_csv_files = sum(len(files) for files in csv_files_by_subdir.values())
        if debug:
            print(f"  {annotation_dir}内のCSVファイル数（キャッシュ）: {total_csv_files}")
            if total_csv_files > 0:
                # 最初のサブディレクトリのファイル例を表示
                for subdir_str, csv_files in list(csv_files_by_subdir.items())[:1]:
                    if len(csv_files) > 0:
                        print(f"    例（{Path(subdir_str).name}）: {[f.name for f in csv_files[:3]]}")
        
        # 各サブディレクトリのCSVファイルをチェック
        for subdir_str, csv_files in csv_files_by_subdir.items():
            for csv_file in csv_files:
                if match_video_id_to_file(video_id, csv_file):
                    possible_paths.append(csv_file)
    
    if debug:
        print(f"  動画ID {video_id}のアノテーションファイルを検索中...")
        print(f"  試行するパス数: {len(possible_paths)}")
        if len(possible_paths) > 0 and len(possible_paths) <= 10:
            print(f"    試行パス例: {[str(p.relative_to(data_path)) for p in possible_paths[:5]]}")
    
    for annotation_path in possible_paths:
        # CSVファイルのみを対象とする
        if annotation_path.exists() and annotation_path.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(annotation_path)
                if debug:
                    print(f"  ✓ アノテーションファイルを発見: {annotation_path}")
                    print(f"    カラム: {list(df.columns)}")
                    print(f"    行数: {len(df)}")
                return df
            except Exception as e:
                if debug:
                    print(f"  Warning: Error reading annotation file {annotation_path}: {e}")
                continue
    
    if debug:
        print(f"  ✗ アノテーションファイルが見つかりませんでした: {video_id}")
        # ディレクトリ構造を確認
        for annotation_dir in annotation_dirs:
            if annotation_dir.exists():
                csv_files_by_subdir = get_cached_csv_files(annotation_dir, cache_key='annotation')
                subdirs = list(csv_files_by_subdir.keys())
                if len(subdirs) > 0:
                    print(f"    {annotation_dir}のサブディレクトリ数: {len(subdirs)}")
                    # 最初のサブディレクトリのファイル例を表示
                    for subdir_str in list(subdirs)[:3]:
                        subdir = Path(subdir_str)
                        csv_files = csv_files_by_subdir[subdir_str]
                        if len(csv_files) > 0:
                            print(f"      {subdir.name}内のCSVファイル例: {[f.name for f in csv_files[:3]]}")
    
    return None


def create_1dcnn_features_from_dataframe(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
    """
    Extra Trees GPUのtrain_dfから1D CNN用の特徴量を生成
    train_dfにactionカラムがない場合は、train_annotation/から読み込む
    
    Args:
        train_df: 訓練データのDataFrame（Extra Trees GPUの形式）
        data_dir: データディレクトリ
        sequence_length: シーケンス長
    
    Returns:
        X_sequences: シーケンスデータ
        y_labels: ラベル
        label_encoder: LabelEncoder
        scaler: StandardScaler
    """
    print("1D CNN用の特徴量を生成中...")
    
    all_sequences = []
    all_labels = []
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    # ラベルの取得（train_dfにactionカラムがある場合とない場合を処理）
    video_actions = {}  # video_id -> action のマッピング
    
    if 'action' in train_df.columns:
        # train_dfにactionカラムがある場合
        for _, row in train_df.iterrows():
            video_id = row.get('video_id', '')
            if video_id:
                video_actions[video_id] = row['action']
    else:
        # train_dfにactionカラムがない場合、アノテーションファイルから読み込む
        print("train_dfにactionカラムが見つかりません。アノテーションファイルから読み込みます...")
        
        # データディレクトリの構造を確認
        data_path = Path(data_dir)
        print(f"データディレクトリ: {data_dir}")
        print(f"データディレクトリの存在: {data_path.exists()}")
        
        if data_path.exists():
            # サブディレクトリを確認
            subdirs = [d.name for d in data_path.iterdir() if d.is_dir()]
            print(f"サブディレクトリ: {subdirs}")
            
            # アノテーションディレクトリを確認
            annotation_dirs = [d for d in subdirs if 'annotation' in d.lower() or 'train' in d.lower()]
            print(f"アノテーション関連ディレクトリ: {annotation_dirs}")
            
            for ann_dir_name in annotation_dirs:
                ann_dir = data_path / ann_dir_name
                if ann_dir.exists():
                    # 再帰的にCSVファイルを検索
                    csv_files = list(ann_dir.rglob('*.csv'))
                    print(f"  {ann_dir_name}内のCSVファイル数（再帰的）: {len(csv_files)}")
                    if len(csv_files) > 0:
                        if len(csv_files) <= 10:
                            print(f"    すべてのファイル: {[str(f.relative_to(data_path)) for f in csv_files]}")
                        else:
                            print(f"    例: {[str(f.relative_to(data_path)) for f in csv_files[:5]]}")
                    else:
                        # サブディレクトリを確認
                        subdirs_in_ann = [d.name for d in ann_dir.iterdir() if d.is_dir()]
                        if len(subdirs_in_ann) > 0:
                            print(f"    サブディレクトリ: {subdirs_in_ann[:10]}")
                            # サブディレクトリ内のファイルも確認
                            for subdir_name in subdirs_in_ann[:3]:
                                subdir = ann_dir / subdir_name
                                sub_csv_files = list(subdir.glob('*.csv'))
                                if len(sub_csv_files) > 0:
                                    print(f"      {subdir_name}内のCSVファイル数: {len(sub_csv_files)}")
                                    print(f"        例: {[f.name for f in sub_csv_files[:3]]}")
        
        video_ids = train_df['video_id'].unique() if 'video_id' in train_df.columns else []
        print(f"処理する動画ID数: {len(video_ids)}")
        if len(video_ids) > 0:
            print(f"動画IDの例: {video_ids[:5]}")
        
        all_actions = []
        found_count = 0
        not_found_count = 0
        
        # 進捗バーを使用してアノテーションファイルを読み込み
        print("\nアノテーションファイルの読み込み中...")
        annotation_start_time = time.time()
        
        for idx, video_id in enumerate(tqdm(video_ids, desc="アノテーション読み込み", disable=not TQDM_AVAILABLE)):
            if idx < 3:  # 最初の3件はデバッグ情報を表示
                annotation_df = load_annotations_for_video(data_dir, video_id, debug=True)
            else:
                annotation_df = load_annotations_for_video(data_dir, video_id, debug=False)
            
            if annotation_df is not None:
                found_count += 1
                # behaviorカラムを優先、なければactionカラムを使用
                if 'behavior' in annotation_df.columns:
                    # 最も頻度の高い行動を取得
                    behavior_counts = annotation_df['behavior'].value_counts()
                    most_common_behavior = behavior_counts.index[0]
                    video_actions[video_id] = most_common_behavior
                    all_actions.append(most_common_behavior)
                elif 'action' in annotation_df.columns:
                    action_counts = annotation_df['action'].value_counts()
                    most_common_action = action_counts.index[0]
                    video_actions[video_id] = most_common_action
                    all_actions.append(most_common_action)
                else:
                    # その他のカラムから行動を推測
                    print(f"  Warning: video {video_id}のアノテーションファイルにbehavior/actionカラムがありません")
                    print(f"    カラム: {list(annotation_df.columns)}")
                    not_found_count += 1
            else:
                not_found_count += 1
        
        annotation_elapsed = time.time() - annotation_start_time
        print(f"\nアノテーションファイルが見つかった動画: {found_count}/{len(video_ids)}")
        print(f"アノテーションファイルが見つからなかった動画: {not_found_count}/{len(video_ids)}")
        print(f"アノテーション読み込み時間: {timedelta(seconds=int(annotation_elapsed))}")
        
        if len(all_actions) == 0:
            print("警告: アノテーションデータが見つかりません。")
            print("代替方法を試行します...")
            
            # 代替方法1: train.csvの他のカラムから行動を推測
            # 代替方法2: すべての動画にデフォルト行動を設定
            # 代替方法3: アノテーションディレクトリ内のすべてのファイルを確認
            
            # アノテーションディレクトリ内のすべてのファイルを確認
            annotation_dirs = [
                data_path / 'train_annotation',
                data_path / 'train_annotations',
                data_path / 'annotation',
                data_path / 'annotations',
            ]
            
            all_annotation_files = []
            for ann_dir in annotation_dirs:
                if ann_dir.exists():
                    csv_files = list(ann_dir.glob('*.csv'))
                    all_annotation_files.extend(csv_files)
            
            if len(all_annotation_files) > 0:
                print(f"アノテーションディレクトリ内に{len(all_annotation_files)}個のCSVファイルが見つかりました")
                print(f"ファイル名の例: {[f.name for f in all_annotation_files[:5]]}")
                
                # ファイル名から動画IDを抽出してマッチングを試行
                for ann_file in all_annotation_files[:10]:  # 最初の10件を確認
                    try:
                        ann_df = pd.read_csv(ann_file)
                        if 'behavior' in ann_df.columns:
                            behavior = ann_df['behavior'].iloc[0] if len(ann_df) > 0 else None
                            if behavior:
                                # ファイル名から動画IDを推測
                                file_stem = ann_file.stem
                                # video_idと一致するか確認
                                for vid in video_ids:
                                    if vid in file_stem or file_stem == vid:
                                        if vid not in video_actions:
                                            video_actions[vid] = behavior
                                            all_actions.append(behavior)
                    except Exception as e:
                        continue
            
            # それでも見つからない場合はデフォルト行動を使用
            if len(all_actions) == 0:
                print("デフォルトの行動を使用します。")
                default_action = 'unknown'
                for video_id in video_ids:
                    if video_id not in video_actions:
                        video_actions[video_id] = default_action
                all_actions = [default_action]
    
    # ラベルをエンコード
    if len(video_actions) > 0:
        unique_actions = list(set(video_actions.values()))
        y_encoded_dict = {action: idx for idx, action in enumerate(unique_actions)}
        label_encoder.fit(unique_actions)
    else:
        raise ValueError("No actions found in train_df or annotation files")
    
    # 各動画のデータを処理
    video_ids = train_df['video_id'].unique() if 'video_id' in train_df.columns else []
    
    print(f"処理する動画数: {len(video_ids)}")
    processed_videos = 0
    
    # 進捗バーを使用して動画データを処理
    sequence_start_time = time.time()
    print("\nシーケンスデータの生成中...")
    
    for video_id in tqdm(video_ids, desc="動画データ処理", disable=not TQDM_AVAILABLE):
        try:
            # 該当する動画の行動を取得
            if video_id in video_actions:
                action = video_actions[video_id]
                label = label_encoder.transform([action])[0]
            else:
                # 動画IDが見つからない場合はスキップ
                print(f"Warning: Action not found for video {video_id}, skipping...")
                continue
            
            # 追跡データを読み込み（最初の数件はデバッグモード）
            debug_tracking = (processed_videos < 3)
            try:
                sequences, feature_cols = load_tracking_data_for_1dcnn(
                    data_dir, video_id, sequence_length, debug=debug_tracking
                )
            except FileNotFoundError as e:
                # ファイルが見つからない場合はスキップして続行
                if not debug_tracking:  # デバッグモードでない場合のみ警告を表示（大量の警告を避ける）
                    if processed_videos % 100 == 0:  # 100件ごとに統計を表示
                        print(f"  処理済み動画数: {processed_videos}/{len(video_ids)} (ファイル未検出: {processed_videos - len(all_sequences)})")
                continue
            except Exception as e:
                # その他のエラーもスキップして続行
                print(f"Warning: Error loading tracking data for video {video_id}: {e}")
                continue
            
            # 各シーケンスに対応するラベルを設定
            for seq_idx in range(len(sequences)):
                all_sequences.append(sequences[seq_idx])
                all_labels.append(label)
            
            processed_videos += 1
            # 進捗バーを使用している場合は、100件ごとの表示は不要
            if not TQDM_AVAILABLE and processed_videos % 100 == 0:
                print(f"  処理済み動画数: {processed_videos}/{len(video_ids)}")
                
        except FileNotFoundError as e:
            # ファイルが見つからない場合はスキップして続行（エラー統計は後で表示）
            continue
        except Exception as e:
            # その他のエラーもスキップして続行
            if processed_videos < 10:  # 最初の10件のみ詳細なエラーを表示
                print(f"Warning: Error processing video {video_id}: {e}")
            continue
    
    sequence_elapsed = time.time() - sequence_start_time
    print(f"\nシーケンス生成時間: {timedelta(seconds=int(sequence_elapsed))}")
    
    # エラー統計を表示
    total_videos = len(video_ids)
    successful_videos = processed_videos
    failed_videos = total_videos - successful_videos
    
    print(f"\n動画処理統計:")
    print(f"  総動画数: {total_videos}")
    print(f"  成功: {successful_videos}")
    print(f"  失敗（ファイル未検出など）: {failed_videos}")
    
    if len(all_sequences) == 0:
        raise ValueError(f"No sequences created. Failed to process all {total_videos} videos.")
    
    if failed_videos > 0:
        print(f"警告: {failed_videos}個の動画の処理に失敗しましたが、{successful_videos}個の動画のデータを使用して続行します。")
    
    print("シーケンスデータを配列に変換中...")
    array_start_time = time.time()
    X_sequences = np.array(all_sequences)
    y_labels = np.array(all_labels)
    print(f"配列変換時間: {timedelta(seconds=int(time.time() - array_start_time))}")
    
    # 特徴量の正規化
    print("特徴量の正規化中...")
    norm_start_time = time.time()
    # シーケンスデータを2Dに変換して正規化
    n_samples, seq_len, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
    print(f"正規化時間: {timedelta(seconds=int(time.time() - norm_start_time))}")
    
    print(f"\n生成されたシーケンス数: {len(X_sequences)}")
    print(f"シーケンス形状: {X_sequences.shape}")
    print(f"ラベル数: {len(y_labels)}")
    
    return X_sequences, y_labels, label_encoder, scaler


def train_1dcnn_model(X_train, y_train, X_val, y_val, 
                     input_shape, num_classes=NUM_CLASSES,
                     batch_size=BATCH_SIZE, epochs=EPOCHS,
                     learning_rate=LEARNING_RATE,
                     model_save_path=MODEL_DIR):
    """
    1D CNNモデルの訓練
    
    Args:
        X_train, y_train: 訓練データ
        X_val, y_val: 検証データ
        input_shape: 入力形状
        num_classes: クラス数
        batch_size: バッチサイズ
        epochs: エポック数
        learning_rate: 学習率
        model_save_path: モデル保存パス
    
    Returns:
        history: 訓練履歴
        model: 訓練済みモデル
    """
    # モデルの構築
    model = build_1dcnn_model(input_shape, num_classes)
    
    # コールバックの設定（Kaggle提出要件: 9時間以内の実行時間を考慮）
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,  # 10に設定（実行時間短縮）
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,  # 5に設定
            min_lr=1e-7,
            verbose=1
        ),
        # モデル保存は実行時間を考慮してオプション化（必要に応じてコメントアウト可能）
        ModelCheckpoint(
            filepath=os.path.join(model_save_path, 'best_1dcnn_model_2.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0  # 実行時間短縮のためverbose=0に変更
        )
    ]
    
    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    # モデル構造の表示（実行時間短縮のため、Kaggle環境では簡略化）
    if IS_KAGGLE:
        # Kaggle環境では簡潔な情報のみ表示
        print(f"モデル入力形状: {input_shape}")
        print(f"モデル出力クラス数: {num_classes}")
        print(f"総パラメータ数: {model.count_params():,}")
    else:
        model.summary()
    
    # 訓練の実行
    print(f"\n訓練開始: バッチサイズ={batch_size}, エポック数={epochs}, 早期停止パティエンス={EARLY_STOPPING_PATIENCE}")
    print(f"訓練データサイズ: {X_train.shape[0]:,}サンプル")
    print(f"検証データサイズ: {X_val.shape[0]:,}サンプル")
    
    train_start_time = time.time()
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    train_elapsed = time.time() - train_start_time
    
    print(f"\n訓練完了!")
    print(f"訓練時間: {timedelta(seconds=int(train_elapsed))}")
    print(f"最終エポック: {len(history.history['loss'])}")
    if 'val_loss' in history.history:
        print(f"最終検証損失: {history.history['val_loss'][-1]:.4f}")
        print(f"最終検証精度: {history.history['val_accuracy'][-1]:.4f}")
    
    return history, model


def predict_with_1dcnn(model, test_df, data_dir, scaler, sequence_length=SEQUENCE_LENGTH):
    """
    1D CNNモデルでテストデータを予測
    
    Args:
        model: 訓練済み1D CNNモデル
        test_df: テストデータのDataFrame
        data_dir: データディレクトリ
        scaler: 訓練時に使用したStandardScaler
        sequence_length: シーケンス長
    
    Returns:
        predictions: 予測結果（確率分布）
        test_metadata: テストデータのメタデータ
    """
    print("1D CNNでテストデータを予測中...")
    
    all_predictions = []
    test_metadata = []
    
    video_ids = test_df['video_id'].unique() if 'video_id' in test_df.columns else []
    
    for idx, row in test_df.iterrows():
        video_id = row.get('video_id', '')
        start_frame = row.get('start_frame', 0)
        stop_frame = row.get('stop_frame', 0)
        
        try:
            # 追跡データを読み込み
            tracking_file = Path(data_dir) / 'test_tracking' / f'{video_id}.csv'
            if not tracking_file.exists():
                # 代替パスを試行
                alt_paths = [
                    Path(data_dir) / 'tracking' / f'{video_id}.csv',
                    Path(data_dir) / 'train_tracking' / f'{video_id}.csv',
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        tracking_file = alt_path
                        break
                else:
                    print(f"Warning: Tracking file not found for video {video_id}")
                    test_metadata.append(row.to_dict())
                    continue
            
            df = pd.read_csv(tracking_file)
            
            # フレーム範囲でフィルタリング
            if 'frame' in df.columns:
                df_filtered = df[(df['frame'] >= start_frame) & (df['frame'] <= stop_frame)]
            else:
                df_filtered = df
            
            # シーケンスを作成
            sequences, _ = prepare_sequences_for_1dcnn(df_filtered, sequence_length)
            
            if len(sequences) == 0:
                # データが不足する場合はパディング
                feature_cols = [col for col in df.columns 
                              if col not in ['frame', 'mouse_id', 'agent_id', 'target_id', 'video_id']]
                if len(feature_cols) > 0:
                    padding = np.zeros((sequence_length, len(feature_cols)))
                    sequences = np.array([padding])
                else:
                    test_metadata.append(row.to_dict())
                    continue
            
            # 正規化
            n_samples, seq_len, n_features = sequences.shape
            sequences_reshaped = sequences.reshape(-1, n_features)
            sequences_scaled = scaler.transform(sequences_reshaped)
            sequences = sequences_scaled.reshape(n_samples, seq_len, n_features)
            
            # 予測
            pred = model.predict(sequences, batch_size=BATCH_SIZE, verbose=0)
            # シーケンスの平均を取る（複数シーケンスがある場合）
            pred_mean = np.mean(pred, axis=0)
            all_predictions.append(pred_mean)
            
            # メタデータを保存
            row_dict = row.to_dict()
            if 'row_id' not in row_dict:
                row_dict['row_id'] = idx
            test_metadata.append(row_dict)
            
        except Exception as e:
            print(f"Error predicting for row {idx}: {e}")
            test_metadata.append(row.to_dict())
            continue
    
    if len(all_predictions) == 0:
        return None, test_metadata
    
    predictions = np.array(all_predictions)
    return predictions, test_metadata


def validate_and_fix_submission(submission_df, sample_submission_path=None, probability_threshold=0.1):
    """
    提出ファイルを検証し、修正する
    
    Args:
        submission_df: 提出用DataFrame
        sample_submission_path: sample_submission.csvのパス（Noneの場合は自動検索）
        probability_threshold: 確率の閾値（この値未満の予測は除外）
    
    Returns:
        validated_df: 検証・修正後のDataFrame
    """
    print("\n提出ファイルの検証と修正を開始...")
    validated_df = submission_df.copy()
    
    # 1. sample_submission.csvと同じ順番・型で提出
    if sample_submission_path is None:
        # sample_submission.csvを自動検索
        possible_paths = [
            Path(DATA_DIR) / 'sample_submission.csv',
            Path('/kaggle/input') / 'sample_submission.csv',
        ]
        if IS_KAGGLE:
            input_dir = Path('/kaggle/input')
            if input_dir.exists():
                for item in input_dir.iterdir():
                    if item.is_dir():
                        possible_paths.append(item / 'sample_submission.csv')
        
        for path in possible_paths:
            if path.exists():
                sample_submission_path = path
                break
    
    if sample_submission_path and Path(sample_submission_path).exists():
        print(f"sample_submission.csvを読み込み: {sample_submission_path}")
        sample_df = pd.read_csv(sample_submission_path)
        
        # 順番をsample_submission.csvに合わせる
        if 'row_id' in sample_df.columns and 'row_id' in validated_df.columns:
            # row_idでマージして順番を合わせる
            validated_df = validated_df.merge(
                sample_df[['row_id']], 
                on='row_id', 
                how='right', 
                suffixes=('', '_sample')
            )
            # マージできなかった行は元のデータから取得
            validated_df = validated_df.fillna(submission_df.set_index('row_id').reindex(sample_df['row_id']).reset_index())
            validated_df = validated_df[sample_df.columns.tolist()]  # 列の順番を合わせる
        
        # 型をsample_submission.csvに合わせる
        for col in validated_df.columns:
            if col in sample_df.columns:
                validated_df[col] = validated_df[col].astype(sample_df[col].dtype)
    
    # 2. タイポ修正（sniff__をsniffにするなど）
    if 'action' in validated_df.columns:
        # 一般的なタイポパターンを修正
        typo_fixes = {
            'sniff__': 'sniff',
            'sniff_': 'sniff',
            'groom_': 'groom',
            'eat_': 'eat',
            'drink_': 'drink',
        }
        for typo, correct in typo_fixes.items():
            mask = validated_df['action'].astype(str).str.contains(typo, na=False)
            if mask.any():
                print(f"タイポ修正: {typo} -> {correct} ({mask.sum()}件)")
                validated_df.loc[mask, 'action'] = validated_df.loc[mask, 'action'].str.replace(typo, correct, regex=False)
    
    # 3. 重複したフレームで別の行動を予測していないか確認
    if all(col in validated_df.columns for col in ['video_id', 'start_frame', 'stop_frame', 'action']):
        print("重複フレームのチェック中...")
        rows_to_drop = set()
        
        # 各video_idごとに処理
        for video_id in validated_df['video_id'].unique():
            video_rows = validated_df[validated_df['video_id'] == video_id].copy()
            video_rows = video_rows.sort_values(['start_frame', 'stop_frame']).reset_index()
            
            for i in range(len(video_rows)):
                if video_rows.iloc[i].name in rows_to_drop:
                    continue
                
                row_i = video_rows.iloc[i]
                start_i, stop_i = row_i['start_frame'], row_i['stop_frame']
                
                # 重複している行を探す
                for j in range(i + 1, len(video_rows)):
                    if video_rows.iloc[j].name in rows_to_drop:
                        continue
                    
                    row_j = video_rows.iloc[j]
                    start_j, stop_j = row_j['start_frame'], row_j['stop_frame']
                    
                    # フレーム範囲が重複しているか確認
                    if not (stop_i < start_j or stop_j < start_i):
                        # 重複している
                        # 確率情報がある場合は、確率が高い方を残す
                        # 確率情報がない場合は、より長いフレーム範囲を持つものを優先
                        length_i = stop_i - start_i
                        length_j = stop_j - start_j
                        
                        if 'probability' in validated_df.columns:
                            prob_i = validated_df.loc[row_i.name, 'probability'] if row_i.name in validated_df.index else 0
                            prob_j = validated_df.loc[row_j.name, 'probability'] if row_j.name in validated_df.index else 0
                            if prob_i >= probability_threshold and prob_j >= probability_threshold:
                                # 両方とも閾値以上の場合、確率が高い方を残す
                                if prob_i > prob_j:
                                    rows_to_drop.add(row_j.name)
                                else:
                                    rows_to_drop.add(row_i.name)
                            elif prob_i >= probability_threshold:
                                rows_to_drop.add(row_j.name)
                            elif prob_j >= probability_threshold:
                                rows_to_drop.add(row_i.name)
                            else:
                                # 両方とも閾値未満の場合、より長い方を残す
                                if length_i >= length_j:
                                    rows_to_drop.add(row_j.name)
                                else:
                                    rows_to_drop.add(row_i.name)
                        else:
                            # 確率情報がない場合、より長いフレーム範囲を持つものを優先
                            if length_i >= length_j:
                                rows_to_drop.add(row_j.name)
                            else:
                                rows_to_drop.add(row_i.name)
        
        if rows_to_drop:
            print(f"警告: {len(rows_to_drop)}件の重複フレームを検出し、削除します")
            validated_df = validated_df.drop(index=list(rows_to_drop))
    
    # 4. ターゲットIDがセルフかどうかが一致しているか確認
    if 'target_id' in validated_df.columns and 'agent_id' in validated_df.columns:
        print("ターゲットIDの検証中...")
        # セルフ（自分自身）の場合は、agent_id == target_id
        self_mask = validated_df['agent_id'] == validated_df['target_id']
        non_self_mask = validated_df['agent_id'] != validated_df['target_id']
        
        # セルフの行と非セルフの行が混在していないか確認
        if self_mask.any() and non_self_mask.any():
            print(f"  セルフ: {self_mask.sum()}件, 非セルフ: {non_self_mask.sum()}件")
    
    # 5. スタートストップフレームが重複してないか確認（より詳細）
    if all(col in validated_df.columns for col in ['video_id', 'start_frame', 'stop_frame']):
        print("フレーム範囲の重複チェック中...")
        validated_df = validated_df.sort_values(['video_id', 'start_frame', 'stop_frame']).reset_index(drop=True)
        
        # 同じvideo_id内で、連続する行のフレーム範囲を確認
        adjusted_count = 0
        for i in range(len(validated_df) - 1):
            if validated_df.iloc[i]['video_id'] == validated_df.iloc[i+1]['video_id']:
                if validated_df.iloc[i]['stop_frame'] >= validated_df.iloc[i+1]['start_frame']:
                    # 重複している場合は、前の行のstop_frameを調整
                    new_stop = validated_df.iloc[i+1]['start_frame'] - 1
                    if new_stop >= validated_df.iloc[i]['start_frame']:
                        validated_df.iloc[i, validated_df.columns.get_loc('stop_frame')] = new_stop
                        adjusted_count += 1
        
        if adjusted_count > 0:
            print(f"  {adjusted_count}件のフレーム範囲を調整しました")
    
    print("✓ 提出ファイルの検証と修正が完了しました")
    return validated_df


def create_submission_1dcnn(predictions, test_metadata, label_encoder, output_path=None, sample_submission_path=None, default_action=None, probability_threshold=0.1):
    """
    1D CNNの予測結果から提出ファイルを作成
    Kaggle提出要件: sample_submission.csvと同様の形式で出力
    
    Args:
        predictions: 予測結果（確率分布）、Noneの場合はデフォルト予測を使用
        test_metadata: テストデータのメタデータ
        label_encoder: LabelEncoderインスタンス（ラベルをデコード）
        output_path: 出力ファイルパス（Noneの場合はKaggle環境に応じて自動設定）
        sample_submission_path: sample_submission.csvのパス（形式確認用）
        default_action: 予測がない場合のデフォルト行動（Noneの場合は最初のクラス）
        probability_threshold: 確率の閾値（この値未満の予測は除外）
    
    Returns:
        submission_df: 提出用DataFrame
    """
    # Kaggle環境の場合は/kaggle/working/submission.csvを優先
    if output_path is None:
        if IS_KAGGLE:
            output_path = '/kaggle/working/submission.csv'
        else:
            output_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    
    # 出力パスがsubmission.csvでない場合は警告
    if 'submission.csv' not in str(output_path):
        print(f"警告: 提出ファイル名がsubmission.csvではありません: {output_path}")
        print("Kaggle提出要件: 提出ファイル名はsubmission.csvである必要があります。")
    
    # sample_submission.csvの形式を確認（可能な場合）
    if sample_submission_path is None:
        # 自動検索
        possible_sample_paths = [
            Path(DATA_DIR) / 'sample_submission.csv',
            Path(DATA_DIR) / 'sample_submission' / 'sample_submission.csv',
            Path('/kaggle/input') / 'sample_submission.csv',
        ]
        if IS_KAGGLE:
            input_dir = Path('/kaggle/input')
            if input_dir.exists():
                for item in input_dir.iterdir():
                    if item.is_dir():
                        possible_sample_paths.append(item / 'sample_submission.csv')
        
        for sample_path in possible_sample_paths:
            if sample_path.exists():
                sample_submission_path = sample_path
                break
    
    # 予測がない場合の処理
    predicted_probabilities = None
    if predictions is None:
        print("Warning: No predictions provided, using default action")
        if default_action is None:
            # ラベルエンコーダーから最初のクラスを取得
            if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
                default_action = label_encoder.classes_[0]
            else:
                default_action = 'unknown'
        predicted_labels = [default_action] * len(test_metadata)
    else:
        # 予測クラスを取得（確率も保持）
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_probabilities = np.max(predictions, axis=1)
        
        # ラベルをデコード（文字列に変換）
        if hasattr(label_encoder, 'inverse_transform'):
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
        else:
            # ラベルエンコーダーが使えない場合はクラスインデックスをそのまま使用
            predicted_labels = [f'class_{cls}' for cls in predicted_classes]
    
    # 提出用DataFrameを作成（MABeコンペティションの要件に合わせてすべての列を含める）
    submission_data = []
    for i, metadata in enumerate(test_metadata):
        # 予測ラベルのインデックスを調整
        pred_idx = min(i, len(predicted_labels) - 1) if len(predicted_labels) > 0 else 0
        action = predicted_labels[pred_idx] if len(predicted_labels) > 0 else default_action
        
        # 確率情報も保持（検証時に使用）
        if predicted_probabilities is not None and len(predicted_probabilities) > pred_idx:
            probability = predicted_probabilities[pred_idx]
        else:
            probability = 1.0
        
        # MABeコンペティションの提出ファイル形式に合わせてすべての列を含める
        row = {
            'row_id': metadata.get('row_id', i),
            'video_id': metadata.get('video_id', ''),
            'agent_id': metadata.get('agent_id', ''),
            'target_id': metadata.get('target_id', ''),
            'action': action,
            'start_frame': metadata.get('start_frame', 0),
            'stop_frame': metadata.get('stop_frame', 0),
            'probability': probability  # 検証用（最終的には削除）
        }
        submission_data.append(row)
    
    submission_df = pd.DataFrame(submission_data)
    
    # 提出ファイルの検証と修正
    submission_df = validate_and_fix_submission(
        submission_df, 
        sample_submission_path=sample_submission_path,
        probability_threshold=probability_threshold
    )
    
    # 検証用のprobabilityカラムを削除（提出ファイルには不要）
    if 'probability' in submission_df.columns:
        submission_df = submission_df.drop(columns=['probability'])
    
    # CSVファイルに保存
    submission_df.to_csv(output_path, index=False)
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"提出データ形状: {submission_df.shape}")
    print(f"提出ファイルのカラム: {list(submission_df.columns)}")
    print("\n提出ファイルの最初の5行:")
    print(submission_df.head())
    
    return submission_df


def main():
    """
    メイン実行関数
    Extra Trees GPUノートブックのコード構造に合わせて実装
    """
    total_start_time = time.time()
    
    print("=" * 60)
    print("MABe Mouse Behavior Detection - 1D CNN (Extra Trees GPUベース)")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kaggle環境: {IS_KAGGLE}")
    print(f"データディレクトリ: {DATA_DIR}")
    
    # システム情報を表示
    print(get_system_info())
    
    # データの読み込み（Extra Trees GPUノートブックと同じ方式）
    try:
        # Extra Trees GPUノートブックでは通常、train.csvとtest.csvを直接読み込む
        train_csv = Path(DATA_DIR) / 'train.csv'
        test_csv = Path(DATA_DIR) / 'test.csv'
        
        # 代替パスも試行（Extra Trees GPUノートブックの構造に合わせる）
        if not train_csv.exists():
            # サブディレクトリを検索
            for subdir in ['train', 'data']:
                alt_path = Path(DATA_DIR) / subdir / 'train.csv'
                if alt_path.exists():
                    train_csv = alt_path
                    break
        
        if not train_csv.exists():
            print(f"警告: train.csvが見つかりません: {train_csv}")
            print(f"データディレクトリの内容を確認中: {DATA_DIR}")
            data_path = Path(DATA_DIR)
            if data_path.exists():
                files = list(data_path.glob('*.csv'))
                dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
                print(f"  見つかったCSVファイル: {[f.name for f in files[:5]]}")
                print(f"  見つかったディレクトリ: {dirs[:5]}")
            return
        
        print_section("[1/4] データの読み込み", level=1)
        data_load_start = time.time()
        
        train_df = pd.read_csv(train_csv)
        print(f"訓練データ: {len(train_df)}行")
        print(f"訓練データのカラム: {list(train_df.columns)[:10]}")
        print_progress(f"データ読み込み完了", data_load_start)
        
        # test.csvの検索（Extra Trees GPUノートブックと同じ方式）
        if not test_csv.exists():
            for subdir in ['test', 'data']:
                alt_path = Path(DATA_DIR) / subdir / 'test.csv'
                if alt_path.exists():
                    test_csv = alt_path
                    break
        
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            print(f"テストデータ: {len(test_df)}行")
            print(f"テストデータのカラム: {list(test_df.columns)[:10]}")
        else:
            print("警告: test.csvが見つかりません")
            test_df = None
        
        # 1D CNN用の特徴量を生成
        print_section("[2/4] 1D CNN用の特徴量生成", level=1)
        feature_start = time.time()
        
        X_sequences, y_labels, label_encoder, scaler = create_1dcnn_features_from_dataframe(
            train_df, DATA_DIR, sequence_length=SEQUENCE_LENGTH
        )
        
        print_progress(f"特徴量生成完了", feature_start)
        
        # 訓練・検証セットに分割
        print("訓練・検証セットに分割中...")
        split_start = time.time()
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        print(f"分割時間: {timedelta(seconds=int(time.time() - split_start))}")
        
        print(f"訓練セット: {X_train.shape}, {y_train.shape}")
        print(f"検証セット: {X_val.shape}, {y_val.shape}")
        
        # モデルの訓練
        print_section("[3/4] 1D CNNモデルの訓練", level=1)
        input_shape = (SEQUENCE_LENGTH, X_train.shape[2])
        history, model = train_1dcnn_model(
            X_train, y_train, X_val, y_val,
            input_shape=input_shape,
            num_classes=NUM_CLASSES,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE
        )
        
        print("✓ モデルの訓練が完了しました")
        
        # テストデータの予測と提出ファイル生成（Kaggle提出要件: submission.csv必須）
        print_section("[4/4] テストデータの予測と提出ファイル生成", level=1)
        submission_created = False
        submission_path = '/kaggle/working/submission.csv' if IS_KAGGLE else os.path.join(OUTPUT_DIR, 'submission.csv')
        
        if test_df is not None:
            predict_start = time.time()
            try:
                predictions, test_metadata = predict_with_1dcnn(
                    model, test_df, DATA_DIR, scaler, sequence_length=SEQUENCE_LENGTH
                )
                
                if predictions is not None:
                    # Kaggle提出要件: 提出ファイル名はsubmission.csvである必要がある
                    # sample_submission.csvのパスを検索
                    sample_submission_path = None
                    possible_sample_paths = [
                        Path(DATA_DIR) / 'sample_submission.csv',
                        Path(DATA_DIR) / 'sample_submission' / 'sample_submission.csv',
                    ]
                    if IS_KAGGLE:
                        input_dir = Path('/kaggle/input')
                        if input_dir.exists():
                            for item in input_dir.iterdir():
                                if item.is_dir():
                                    possible_sample_paths.append(item / 'sample_submission.csv')
                    
                    for sample_path in possible_sample_paths:
                        if sample_path.exists():
                            sample_submission_path = str(sample_path)
                            break
                    
                    # デフォルトの行動を設定（ラベルエンコーダーから取得）
                    default_action = None
                    if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
                        default_action = label_encoder.classes_[0]
                    
                    submission_df = create_submission_1dcnn(
                        predictions, test_metadata, label_encoder,
                        output_path=submission_path,
                        sample_submission_path=sample_submission_path,
                        default_action=default_action,
                        probability_threshold=0.1
                    )
                    submission_created = True
                    predict_elapsed = time.time() - predict_start
                    print("\n✓ 提出ファイルの生成が完了しました!")
                    print(f"予測時間: {timedelta(seconds=int(predict_elapsed))}")
                    print(f"提出ファイル: {submission_path}")
                else:
                    print("警告: 予測が生成されませんでした。デフォルト提出ファイルを生成します。")
            except Exception as pred_error:
                print(f"警告: 予測処理中にエラーが発生しました: {pred_error}")
                print("デフォルト提出ファイルを生成します。")
        
        # 提出ファイルが生成されていない場合、デフォルト提出ファイルを作成（Kaggle提出要件）
        if not submission_created:
            print("\nデフォルト提出ファイルを生成中...")
            try:
                # test.csvから最低限の提出ファイルを作成
                if test_df is not None:
                    default_submission = test_df.copy()
                    # actionカラムがない場合はデフォルト値を設定
                    if 'action' not in default_submission.columns:
                        # ラベルエンコーダーから最初のクラスを取得
                        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
                            default_action = label_encoder.classes_[0]
                        else:
                            default_action = 'unknown'
                        default_submission['action'] = default_action
                    
                    # 必須カラムを確認
                    required_cols = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
                    for col in required_cols:
                        if col not in default_submission.columns:
                            if col == 'row_id':
                                default_submission['row_id'] = range(len(default_submission))
                            else:
                                default_submission[col] = '' if col in ['video_id', 'agent_id', 'target_id', 'action'] else 0
                    
                    # 必須カラムのみを保持
                    submission_df = default_submission[required_cols]
                    submission_df.to_csv(submission_path, index=False)
                    submission_created = True
                    print(f"✓ デフォルト提出ファイルを生成しました: {submission_path}")
                else:
                    # test.csvもない場合は最小限の提出ファイルを作成
                    min_submission = pd.DataFrame({
                        'row_id': [0],
                        'video_id': [''],
                        'agent_id': [''],
                        'target_id': [''],
                        'action': ['unknown'],
                        'start_frame': [0],
                        'stop_frame': [0]
                    })
                    min_submission.to_csv(submission_path, index=False)
                    submission_created = True
                    print(f"✓ 最小限の提出ファイルを生成しました: {submission_path}")
            except Exception as sub_error:
                print(f"エラー: 提出ファイル生成中にエラーが発生しました: {sub_error}")
                import traceback
                traceback.print_exc()
        
        if not submission_created:
            print("\n警告: 提出ファイルが生成されませんでした。")
            print("Kaggle提出要件: submission.csvファイルが必要です。")
        
        # 総実行時間を表示
        total_elapsed = time.time() - total_start_time
        print_section("実行完了", level=1)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {timedelta(seconds=int(total_elapsed))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

