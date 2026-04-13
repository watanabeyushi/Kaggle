"""
MABe Mouse Behavior Detection - 1D CNN実装（Kaggle最適化版）
Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
参考: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
既存実装参照: 1DCNN_model_2.py
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
from datetime import datetime, timedelta

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

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

# データ読み込み制限設定（テスト・デバッグ用）
# LIMIT_TRAIN_ROWS = 1: 最初の10行のみ読み込む
# LIMIT_TRAIN_ROWS = 0: すべての行を読み込む（通常モード）
LIMIT_TRAIN_ROWS = 1  # 0に設定すると制限なし
LIMIT_TRAIN_ROWS_COUNT = 10  # 制限する行数

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


def build_1dcnn_model(input_shape, num_classes=NUM_CLASSES):
    """
    1D CNNモデルの構築
    1DCNN_model_2.pyの実装を参照
    
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
    データフレームから1D CNN用のシーケンスデータを準備
    
    Args:
        df: 追跡データのDataFrame
        sequence_length: シーケンス長
    
    Returns:
        sequences: シーケンスデータ (n_samples, sequence_length, n_features)
        feature_cols: 特徴量カラム名
    """
    # 数値カラムのみを抽出（IDカラムを除外）
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
                
                # サブディレクトリ内のすべてのCSVファイルを検索
                for csv_file in sub_subdir.glob('*.csv'):
                    file_stem = csv_file.stem
                    # マッチング条件
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if csv_file not in possible_paths:
                            if priority_search:
                                # 優先サブディレクトリのファイルを先頭に追加
                                possible_paths.insert(0, csv_file)
                            else:
                                possible_paths.append(csv_file)
                
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
            
            # 再帰的検索も実行（念のため、CSVとParquetの両方）
            for csv_file in subdir.rglob('*.csv'):
                file_stem = csv_file.stem
                # より柔軟なマッチング
                # 1. 完全一致
                if file_stem == video_id_str:
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 2. video_idがファイル名に含まれる
                elif video_id_str in file_stem:
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 3. アンダースコア区切りで確認
                elif any(video_id_str in part for part in file_stem.split('_')):
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 4. ファイル名から数値を抽出して比較
                elif any(char.isdigit() for char in file_stem):
                    numbers_in_file = re.findall(r'\d+', file_stem)
                    if video_id_str in numbers_in_file:
                        if csv_file not in possible_paths:
                            possible_paths.append(csv_file)
    
    if debug:
        print(f"  追跡ファイル検索: video_id={video_id}")
        print(f"    試行するパス数: {len(possible_paths)}")
        if len(possible_paths) <= 10:
            print(f"    試行パス例: {[str(p.relative_to(data_path)) for p in possible_paths[:5]]}")
        elif len(possible_paths) > 4:
            print(f"    標準パス: {[str(p.relative_to(data_path)) for p in possible_paths[:4]]}")
            print(f"    再帰的検索で見つかったファイル例: {[str(p.relative_to(data_path)) for p in possible_paths[4:9]]}")
        # サブディレクトリ内のファイル名例を表示
        for subdir_name in tracking_dirs:
            subdir = data_path / subdir_name
            if subdir.exists():
                subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    # 最初のサブディレクトリ内のファイルを確認
                    sample_subdir = subdirs[0]
                    sample_files = list(sample_subdir.glob('*.csv'))
                    sample_parquet = list(sample_subdir.glob('*.parquet'))
                    if len(sample_files) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のCSVファイル例: {[f.name for f in sample_files[:5]]}")
                    if len(sample_parquet) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のParquetファイル例: {[f.name for f in sample_parquet[:5]]}")
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            if debug:
                print(f"  ✓ 追跡ファイルを発見: {path.relative_to(data_path)}")
            return path
    
    if debug:
        print(f"  ✗ 追跡ファイルが見つかりません: video_id={video_id}")
        # デバッグ: サブディレクトリ内のファイル名を表示
        for subdir_name in tracking_dirs:
            subdir = data_path / subdir_name
            if subdir.exists():
                subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    # 最初のサブディレクトリ内のファイルを確認
                    sample_subdir = subdirs[0]
                    sample_files = list(sample_subdir.glob('*.csv'))
                    if len(sample_files) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のファイル例: {[f.name for f in sample_files[:5]]}")
                    else:
                        # CSVファイルがない場合、他のファイル形式を確認
                        all_files = list(sample_subdir.glob('*'))
                        if len(all_files) > 0:
                            print(f"    {subdir_name}/{sample_subdir.name}内のファイル（CSV以外）: {[f.name for f in all_files[:5]]}")
    
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
    
    # 再帰的にすべてのCSVファイルを検索
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
                # サブディレクトリ内のすべてのCSVファイルを検索
                for csv_file in sub_subdir.glob('*.csv'):
                    file_stem = csv_file.stem
                    # マッチング条件
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if csv_file not in possible_paths:
                            possible_paths.append(csv_file)
                
                # サブディレクトリ内のすべてのParquetファイルを検索
                for parquet_file in sub_subdir.glob('*.parquet'):
                    file_stem = parquet_file.stem
                    # マッチング条件（ファイル名が数値のみの場合も含む）
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if parquet_file not in possible_paths:
                            possible_paths.append(parquet_file)
            
            # 再帰的検索も実行（念のため、CSVとParquetの両方）
            for csv_file in subdir.rglob('*.csv'):
                file_stem = csv_file.stem
                # より柔軟なマッチング
                # 1. 完全一致
                if file_stem == video_id_str:
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 2. video_idがファイル名に含まれる
                elif video_id_str in file_stem:
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 3. アンダースコア区切りで確認
                elif any(video_id_str in part for part in file_stem.split('_')):
                    if csv_file not in possible_paths:
                        possible_paths.append(csv_file)
                # 4. ファイル名から数値を抽出して比較
                elif any(char.isdigit() for char in file_stem):
                    numbers_in_file = re.findall(r'\d+', file_stem)
                    if video_id_str in numbers_in_file:
                        if csv_file not in possible_paths:
                            possible_paths.append(csv_file)
            
            # Parquetファイルも再帰的に検索
            for parquet_file in subdir.rglob('*.parquet'):
                file_stem = parquet_file.stem
                # より柔軟なマッチング
                # 1. 完全一致
                if file_stem == video_id_str:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                # 2. video_idがファイル名に含まれる
                elif video_id_str in file_stem:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                # 3. アンダースコア区切りで確認
                elif any(video_id_str in part for part in file_stem.split('_')):
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                # 4. ファイル名から数値を抽出して比較
                elif any(char.isdigit() for char in file_stem):
                    numbers_in_file = re.findall(r'\d+', file_stem)
                    if video_id_str in numbers_in_file:
                        if parquet_file not in possible_paths:
                            possible_paths.append(parquet_file)
    
    if debug:
        print(f"  アノテーションファイル検索: video_id={video_id}")
        print(f"    試行するパス数: {len(possible_paths)}")
        if len(possible_paths) > 4:
            print(f"    再帰的検索で見つかったファイル例: {[str(p.relative_to(data_path)) for p in possible_paths[4:9]]}")
        # サブディレクトリ内のファイル名例を表示
        for subdir_name in annotation_dirs:
            subdir = data_path / subdir_name
            if subdir.exists():
                subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    # 最初のサブディレクトリ内のファイルを確認
                    sample_subdir = subdirs[0]
                    sample_files = list(sample_subdir.glob('*.csv'))
                    sample_parquet = list(sample_subdir.glob('*.parquet'))
                    if len(sample_files) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のCSVファイル例: {[f.name for f in sample_files[:5]]}")
                    if len(sample_parquet) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のParquetファイル例: {[f.name for f in sample_parquet[:5]]}")
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            if debug:
                print(f"  ✓ アノテーションファイルを発見: {path.relative_to(data_path)}")
            return path
    
    if debug:
        print(f"  ✗ アノテーションファイルが見つかりません: video_id={video_id}")
        # デバッグ: サブディレクトリ内のファイル名を表示
        for subdir_name in annotation_dirs:
            subdir = data_path / subdir_name
            if subdir.exists():
                subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    # 最初のサブディレクトリ内のファイルを確認
                    sample_subdir = subdirs[0]
                    sample_files = list(sample_subdir.glob('*.csv'))
                    if len(sample_files) > 0:
                        print(f"    {subdir_name}/{sample_subdir.name}内のファイル例: {[f.name for f in sample_files[:5]]}")
                    else:
                        # CSVファイルがない場合、他のファイル形式を確認
                        all_files = list(sample_subdir.glob('*'))
                        if len(all_files) > 0:
                            print(f"    {subdir_name}/{sample_subdir.name}内のファイル（CSV以外）: {[f.name for f in all_files[:5]]}")
    
    return None


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
    original_columns = list(df.columns)
    
    # カラム名のマッピング辞書（同様の意味を持つカラム名を統一）
    column_mapping = {
        # 行動/アクション関連
        'behavior': 'action',
        'behavior_type': 'action',
        'action_type': 'action',
        'behavior_label': 'action',
        'action_label': 'action',
        'behavior_name': 'action',
        'action_name': 'action',
        'behavior_class': 'action',
        'action_class': 'action',
        'behavior_category': 'action',
        'action_category': 'action',
        'label': 'action',
        'class': 'action',
        'category': 'action',
        
        # エージェントID関連
        'agent_id': 'agent_id',
        'agent': 'agent_id',
        'mouse_id': 'agent_id',
        'mouse': 'agent_id',
        'subject_id': 'agent_id',
        'subject': 'agent_id',
        'actor_id': 'agent_id',
        'actor': 'agent_id',
        
        # ターゲットID関連
        'target_id': 'target_id',
        'target': 'target_id',
        'target_mouse_id': 'target_id',
        'target_mouse': 'target_id',
        'object_id': 'target_id',
        'object': 'target_id',
        'recipient_id': 'target_id',
        'recipient': 'target_id',
        
        # フレーム関連
        'start_frame': 'start_frame',
        'start': 'start_frame',
        'start_time': 'start_frame',
        'frame_start': 'start_frame',
        'begin_frame': 'start_frame',
        'begin': 'start_frame',
        
        'stop_frame': 'stop_frame',
        'end_frame': 'stop_frame',
        'stop': 'stop_frame',
        'end': 'stop_frame',
        'end_time': 'stop_frame',
        'frame_end': 'stop_frame',
        'finish_frame': 'stop_frame',
        'finish': 'stop_frame',
        
        # ビデオID関連
        'video_id': 'video_id',
        'video': 'video_id',
        'video_name': 'video_id',
        'clip_id': 'video_id',
        'clip': 'video_id',
    }
    
    # カラム名を統一
    renamed_columns = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        # 完全一致を確認
        if col_lower in column_mapping:
            new_name = column_mapping[col_lower]
            if col != new_name:
                renamed_columns[col] = new_name
        else:
            # 部分一致を確認（例: 'behavior_type' -> 'action'）
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
    # 例: 'action'と'behavior'の両方がある場合、'action'を優先
    if 'action' in df.columns and 'behavior' in df.columns:
        # 'behavior'カラムの値が'action'カラムの値より多い場合、'behavior'を'action'に統合
        if df['behavior'].notna().sum() > df['action'].notna().sum():
            df['action'] = df['action'].fillna(df['behavior'])
        df = df.drop(columns=['behavior'])
        if debug:
            print(f"    'behavior'カラムを'action'に統合しました")
    
    # 必須カラムの確認
    required_columns = ['action']  # 最低限'action'カラムが必要
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns and debug:
        print(f"    警告: 必須カラムが見つかりません: {missing_columns}")
        print(f"    利用可能なカラム: {list(df.columns)}")
    
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
            annotation_df: アノテーションデータのDataFrame（カラム名統一済み）、見つからない場合はNone
            annotation_file_path: アノテーションファイルのパス、見つからない場合はNone
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


def create_1dcnn_features(train_df, data_dir, sequence_length=SEQUENCE_LENGTH):
    """
    train_dfから1D CNN用の特徴量を生成（参考ノートブックの方式に合わせる）
    
    Args:
        train_df: 訓練データのDataFrame
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
        print(f"処理する動画ID数: {len(video_ids)}")
        if len(video_ids) > 0:
            print(f"動画IDの例: {video_ids[:5]}")
        
        found_count = 0
        not_found_count = 0
        no_column_count = 0
        annotation_file_paths = {}  # video_id -> annotation_file_path のマッピング
        
        for idx, video_id in enumerate(tqdm(video_ids, desc="アノテーション読み込み", disable=not TQDM_AVAILABLE)):
            # lab_idを取得
            lab_id = None
            if 'lab_id' in train_df.columns:
                video_row = train_df[train_df['video_id'] == video_id]
                if len(video_row) > 0:
                    lab_id = video_row.iloc[0]['lab_id']
            
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
        
        # データディレクトリの構造を確認
        data_path = Path(data_dir)
        annotation_dirs = [
            data_path / 'train_annotation',
            data_path / 'train_annotations',
            data_path / 'annotation',
            data_path / 'annotations',
        ]
        
        print("アノテーションディレクトリの確認中...")
        for ann_dir in annotation_dirs:
            if ann_dir.exists():
                csv_files = list(ann_dir.glob('*.csv'))
                print(f"  {ann_dir}: {len(csv_files)}個のCSVファイル")
                if len(csv_files) > 0:
                    # 最初のファイルを確認
                    sample_file = csv_files[0]
                    try:
                        sample_df = pd.read_csv(sample_file)
                        print(f"    サンプルファイル: {sample_file.name}")
                        print(f"    カラム: {list(sample_df.columns)}")
                        if len(sample_df) > 0:
                            print(f"    最初の行: {sample_df.iloc[0].to_dict()}")
                    except Exception as e:
                        print(f"    エラー: {e}")
        
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
    print(f"処理する動画数: {len(video_ids)}")
    
    # アノテーションファイルのパスマッピングを取得（追跡ファイル検索で使用）
    # 既にアノテーション読み込み時に取得していない場合は、ここで取得
    if 'annotation_file_paths' not in locals() or len(annotation_file_paths) == 0:
        annotation_file_paths = {}
        for video_id in video_ids:
            annotation_file = find_annotation_file(data_dir, video_id, debug=False)
            if annotation_file:
                annotation_file_paths[video_id] = str(annotation_file)
    
    # データディレクトリの構造を確認
    data_path = Path(data_dir)
    print("\n追跡データディレクトリの確認中...")
    tracking_dirs = [
        data_path / 'train_tracking',
        data_path / 'tracking',
        data_path / 'test_tracking',
    ]
    for tracking_dir in tracking_dirs:
        if tracking_dir.exists():
            csv_files = list(tracking_dir.glob('*.csv'))
            print(f"  {tracking_dir}: {len(csv_files)}個のCSVファイル")
            if len(csv_files) > 0:
                print(f"    ファイル例: {[f.name for f in csv_files[:5]]}")
        else:
            print(f"  {tracking_dir}: 存在しません")
    
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
    
    # シーケンスデータを配列に変換
    print("シーケンスデータを配列に変換中...")
    X_sequences = np.array(all_sequences)
    y_labels = np.array(all_labels)
    
    # 特徴量の正規化
    print("特徴量の正規化中...")
    n_samples, seq_len, n_features = X_sequences.shape
    X_reshaped = X_sequences.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_sequences = X_scaled.reshape(n_samples, seq_len, n_features)
    
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
            filepath=os.path.join(model_save_path, 'best_1dcnn_model_3.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]
    
    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
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
    1D CNNモデルでテストデータを予測（参考ノートブックの方式に合わせる）
    
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
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="予測中", disable=not TQDM_AVAILABLE):
        video_id = row.get('video_id', '')
        start_frame = row.get('start_frame', 0)
        stop_frame = row.get('stop_frame', 0)
        
        try:
            # 追跡ファイルを検索
            tracking_file = find_tracking_file(data_dir, video_id)
            
            if tracking_file is None:
                # ファイルが見つからない場合はデフォルト予測
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
            if idx < 10:
                print(f"Error predicting for row {idx}: {e}")
            test_metadata.append(row.to_dict())
            continue
    
    if len(all_predictions) == 0:
        return None, test_metadata
    
    predictions = np.array(all_predictions)
    return predictions, test_metadata


def create_submission(predictions, test_metadata, label_encoder, output_path=None):
    """
    提出ファイルを作成（参考ノートブックの方式に合わせる）
    https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
    
    Args:
        predictions: 予測結果（確率分布）、Noneの場合はデフォルト予測を使用
        test_metadata: テストデータのメタデータ
        label_encoder: LabelEncoderインスタンス
        output_path: 出力ファイルパス（Noneの場合は自動設定）
    
    Returns:
        submission_df: 提出用DataFrame
    """
    # Kaggle環境の場合は/kaggle/working/submission.csvを優先
    if output_path is None:
        if IS_KAGGLE:
            output_path = '/kaggle/working/submission.csv'
        else:
            output_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    
    # sample_submission.csvの形式を確認（可能な場合）
    sample_submission_path = None
    possible_sample_paths = [
        Path(DATA_DIR) / 'sample_submission.csv',
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
    if predictions is None:
        print("Warning: No predictions provided, using default action")
        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
            default_action = label_encoder.classes_[0]
        else:
            default_action = 'unknown'
        predicted_labels = [default_action] * len(test_metadata)
    else:
        # 予測クラスを取得
        predicted_classes = np.argmax(predictions, axis=1)
        
        # ラベルをデコード（文字列に変換）
        if hasattr(label_encoder, 'inverse_transform'):
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
        else:
            predicted_labels = [f'class_{cls}' for cls in predicted_classes]
    
    # 提出用DataFrameを作成（参考ノートブックの形式に合わせる）
    submission_data = []
    for i, metadata in enumerate(test_metadata):
        # 予測ラベルのインデックスを調整
        pred_idx = min(i, len(predicted_labels) - 1) if len(predicted_labels) > 0 else 0
        action = predicted_labels[pred_idx] if len(predicted_labels) > 0 else 'unknown'
        
        # 参考ノートブックの形式に合わせる（row_id, action）
        row = {
            'row_id': metadata.get('row_id', i),
            'action': action,
        }
        
        # 他のカラムも含める（sample_submission.csvに合わせる）
        if sample_submission_path and sample_submission_path.exists():
            sample_df = pd.read_csv(sample_submission_path)
            for col in sample_df.columns:
                if col not in row:
                    row[col] = metadata.get(col, '')
        else:
            # sample_submission.csvがない場合は、test_metadataのすべてのカラムを含める
            for key, value in metadata.items():
                if key not in row:
                    row[key] = value
        
        submission_data.append(row)
    
    submission_df = pd.DataFrame(submission_data)
    
    # sample_submission.csvがある場合は、その形式に合わせる
    if sample_submission_path and sample_submission_path.exists():
        sample_df = pd.read_csv(sample_submission_path)
        
        # 列の順番をsample_submission.csvに合わせる
        if 'row_id' in sample_df.columns:
            # row_idでマージして順番を合わせる
            submission_df = submission_df.merge(
                sample_df[['row_id']], 
                on='row_id', 
                how='right', 
                suffixes=('', '_sample')
            )
            # マージできなかった行は元のデータから取得
            for col in submission_df.columns:
                if col.endswith('_sample'):
                    continue
                if col not in sample_df.columns:
                    submission_df = submission_df.drop(columns=[col])
            
            # 列の順番をsample_submission.csvに合わせる
            submission_df = submission_df[sample_df.columns.tolist()]
        
        # 型をsample_submission.csvに合わせる
        for col in submission_df.columns:
            if col in sample_df.columns:
                submission_df[col] = submission_df[col].astype(sample_df[col].dtype)
    
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
    メイン実行関数（参考ノートブックのコード構造に合わせて実装）
    """
    total_start_time = time.time()
    
    print("=" * 60)
    print("MABe Mouse Behavior Detection - 1D CNN (Kaggle最適化版)")
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
    
    # データの読み込み（参考ノートブックの方式）
    try:
        # train.csvとtest.csvを読み込む
        train_csv = Path(DATA_DIR) / 'train.csv'
        test_csv = Path(DATA_DIR) / 'test.csv'
        
        # 代替パスも試行
        if not train_csv.exists():
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
        
        print("\n[1/4] データの読み込み")
        data_load_start = time.time()
        
        # データ読み込み制限の適用
        if LIMIT_TRAIN_ROWS == 1:
            train_df = pd.read_csv(train_csv, nrows=LIMIT_TRAIN_ROWS_COUNT)
            print(f"注意: データ読み込みを最初の{LIMIT_TRAIN_ROWS_COUNT}行に制限しています（テスト・デバッグ用）")
        else:
            train_df = pd.read_csv(train_csv)
        print(f"訓練データ: {len(train_df)}行")
        print(f"訓練データのカラム: {list(train_df.columns)[:10]}")
        
        # test.csvの検索
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
        
        print(f"データ読み込み時間: {timedelta(seconds=int(time.time() - data_load_start))}")
        
        # データディレクトリ構造の詳細確認
        print("\nデータディレクトリ構造の確認中...")
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
            if len(all_csv_files) > 0:
                print("CSVファイルの例（最初の10件）:")
                for csv_file in all_csv_files[:10]:
                    rel_path = csv_file.relative_to(data_path)
                    print(f"  {rel_path}")
            if len(all_parquet_files) > 0:
                print("Parquetファイルの例（最初の10件）:")
                for parquet_file in all_parquet_files[:10]:
                    rel_path = parquet_file.relative_to(data_path)
                    print(f"  {rel_path}")
                
                # ファイル名のパターンを分析
                print("\nファイル名のパターン分析:")
                file_stems = [f.stem for f in all_parquet_files[:20]]
                print(f"  Parquetファイル名の例: {file_stems[:10]}")
                
                # 数値が含まれるファイル名を確認
                numeric_files = [f for f in all_parquet_files[:20] if any(c.isdigit() for c in f.stem)]
                if numeric_files:
                    print(f"  数値を含むParquetファイル名の例: {[f.stem for f in numeric_files[:5]]}")
            
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
                        # サブディレクトリ内のファイルを直接確認
                        for subdir in subdirs[:3]:
                            subdir_path = dir_path / subdir.name
                            subdir_csv = list(subdir_path.glob('*.csv'))
                            subdir_parquet = list(subdir_path.glob('*.parquet'))
                            if len(subdir_csv) > 0:
                                print(f"      {subdir.name}内のCSVファイル数: {len(subdir_csv)}")
                                print(f"      CSVファイル例: {[f.name for f in subdir_csv[:5]]}")
                                print(f"      CSVファイルパス例: {[str(f.relative_to(data_path)) for f in subdir_csv[:3]]}")
                            if len(subdir_parquet) > 0:
                                print(f"      {subdir.name}内のParquetファイル数: {len(subdir_parquet)}")
                                print(f"      Parquetファイル例: {[f.name for f in subdir_parquet[:5]]}")
                                print(f"      Parquetファイルパス例: {[str(f.relative_to(data_path)) for f in subdir_parquet[:3]]}")
                            if len(subdir_csv) == 0 and len(subdir_parquet) == 0:
                                # CSV/Parquetファイルがない場合、他のファイル形式を確認
                                all_files = list(subdir_path.glob('*'))
                                if len(all_files) > 0:
                                    print(f"      {subdir.name}内のファイル（CSV/Parquet以外）: {[f.name for f in all_files[:5]]}")
                    if len(csv_files) > 0:
                        print(f"    CSVファイル例: {[f.name for f in csv_files[:5]]}")
                    if len(parquet_files) > 0:
                        print(f"    Parquetファイル例: {[f.name for f in parquet_files[:5]]}")
            
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
                        # サブディレクトリ内のファイルを直接確認
                        for subdir in subdirs[:3]:
                            subdir_path = dir_path / subdir.name
                            subdir_csv = list(subdir_path.glob('*.csv'))
                            subdir_parquet = list(subdir_path.glob('*.parquet'))
                            if len(subdir_csv) > 0:
                                print(f"      {subdir.name}内のCSVファイル数: {len(subdir_csv)}")
                                print(f"      CSVファイル例: {[f.name for f in subdir_csv[:5]]}")
                                print(f"      CSVファイルパス例: {[str(f.relative_to(data_path)) for f in subdir_csv[:3]]}")
                            if len(subdir_parquet) > 0:
                                print(f"      {subdir.name}内のParquetファイル数: {len(subdir_parquet)}")
                                print(f"      Parquetファイル例: {[f.name for f in subdir_parquet[:5]]}")
                                print(f"      Parquetファイルパス例: {[str(f.relative_to(data_path)) for f in subdir_parquet[:3]]}")
                            if len(subdir_csv) == 0 and len(subdir_parquet) == 0:
                                # CSV/Parquetファイルがない場合、他のファイル形式を確認
                                all_files = list(subdir_path.glob('*'))
                                if len(all_files) > 0:
                                    print(f"      {subdir.name}内のファイル（CSV/Parquet以外）: {[f.name for f in all_files[:5]]}")
                    if len(csv_files) > 0:
                        print(f"    CSVファイル例: {[f.name for f in csv_files[:5]]}")
                    if len(parquet_files) > 0:
                        print(f"    Parquetファイル例: {[f.name for f in parquet_files[:5]]}")
        else:
            print(f"警告: データディレクトリが存在しません: {data_path}")
        
        # 1D CNN用の特徴量を生成
        print("\n[2/4] 1D CNN用の特徴量生成")
        feature_start = time.time()
        
        X_sequences, y_labels, label_encoder, scaler = create_1dcnn_features(
            train_df, DATA_DIR, sequence_length=SEQUENCE_LENGTH
        )
        
        print(f"特徴量生成時間: {timedelta(seconds=int(time.time() - feature_start))}")
        
        # 訓練・検証セットに分割
        print("\n訓練・検証セットに分割中...")
        split_start = time.time()
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequences, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )
        print(f"分割時間: {timedelta(seconds=int(time.time() - split_start))}")
        
        print(f"訓練セット: {X_train.shape}, {y_train.shape}")
        print(f"検証セット: {X_val.shape}, {y_val.shape}")
        
        # モデルの訓練
        print("\n[3/4] 1D CNNモデルの訓練")
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
        
        # テストデータの予測と提出ファイル生成
        print("\n[4/4] テストデータの予測と提出ファイル生成")
        submission_path = '/kaggle/working/submission.csv' if IS_KAGGLE else os.path.join(OUTPUT_DIR, 'submission.csv')
        submission_created = False
        
        if test_df is not None:
            predict_start = time.time()
            try:
                predictions, test_metadata = predict_with_1dcnn(
                    model, test_df, DATA_DIR, scaler, sequence_length=SEQUENCE_LENGTH
                )
                
                if predictions is not None:
                    submission_df = create_submission(
                        predictions, test_metadata, label_encoder,
                        output_path=submission_path
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
        
        # 提出ファイルが生成されていない場合、デフォルト提出ファイルを作成
        if not submission_created:
            print("\nデフォルト提出ファイルを生成中...")
            try:
                if test_df is not None:
                    default_submission = test_df.copy()
                    # actionカラムがない場合はデフォルト値を設定
                    if 'action' not in default_submission.columns:
                        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
                            default_action = label_encoder.classes_[0]
                        else:
                            default_action = 'unknown'
                        default_submission['action'] = default_action
                    
                    # row_idカラムを確認
                    if 'row_id' not in default_submission.columns:
                        default_submission['row_id'] = range(len(default_submission))
                    
                    # 必須カラムのみを保持（参考ノートブックの形式に合わせる）
                    if 'row_id' in default_submission.columns and 'action' in default_submission.columns:
                        submission_df = default_submission[['row_id', 'action']]
                    else:
                        submission_df = default_submission
                    
                    submission_df.to_csv(submission_path, index=False)
                    submission_created = True
                    print(f"✓ デフォルト提出ファイルを生成しました: {submission_path}")
            except Exception as sub_error:
                print(f"エラー: 提出ファイル生成中にエラーが発生しました: {sub_error}")
                import traceback
                traceback.print_exc()
        
        if not submission_created:
            print("\n警告: 提出ファイルが生成されませんでした。")
            print("Kaggle提出要件: submission.csvファイルが必要です。")
        
        # 総実行時間を表示
        total_elapsed = time.time() - total_start_time
        print("\n" + "=" * 60)
        print("実行完了")
        print("=" * 60)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {timedelta(seconds=int(total_elapsed))}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

