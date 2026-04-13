"""
MABe Mouse Behavior Detection - 1D CNN実装（Kaggle最適化版 - 推論用）
Kaggleコンペティション: https://www.kaggle.com/competitions/MABe-mouse-behavior-detection
参考: https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
既存実装参照: 1DCNN_model_3.py, 1DCNN_model_4.py

このスクリプトは推論専用です。学習済みモデルを読み込んで予測を行います。
学習は1DCNN_model_4.pyで実行してください。
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from pathlib import Path

# Kaggle環境の検出
IS_KAGGLE = os.path.exists('/kaggle/input')

# モデルパラメータ（学習時と同じ値を使用）
SEQUENCE_LENGTH = 64
NUM_CLASSES = 15
BATCH_SIZE = 64

# パス設定（参考ノートブックの方式に合わせる）
# https://www.kaggle.com/code/harshaorwhat/fast-mabe-social-behavior-detection-with-xgboost
if IS_KAGGLE:
    # Kaggle環境では/kaggle/input/からデータセットを読み込む
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

print(f"データディレクトリ: {DATA_DIR}")
print(f"出力ディレクトリ: {OUTPUT_DIR}")
print(f"モデル読み込みディレクトリ: {MODEL_DIR}")


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
    # 数値カラムのみを抽出（IDカラムを除外）
    exclude_cols = ['frame', 'mouse_id', 'agent_id', 'target_id', 'video_id']
    
    # ボディーパーツを7個に絞って特徴量カラムを選択
    feature_cols = extract_body_parts_from_columns(df, exclude_cols=exclude_cols, max_body_parts=7)
    
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found")
    
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
    
    Args:
        data_dir: データディレクトリ
        video_id: 動画ID
        annotation_file_path: アノテーションファイルのパス（オプション）
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
        if len(annotation_path.parts) >= 2:
            annotation_subdir = annotation_path.parts[-2]
    
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
    
    # 再帰的にすべてのファイルを検索
    tracking_dirs = ['train_tracking', 'tracking', 'test_tracking']
    for subdir_name in tracking_dirs:
        subdir = data_path / subdir_name
        if subdir.exists() and subdir.is_dir():
            subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            
            # サブディレクトリ内のファイルを直接検索
            for sub_subdir in subdirs:
                priority_search = (annotation_subdir and sub_subdir.name == annotation_subdir)
                
                # サブディレクトリ内のすべてのParquetファイルを検索
                for parquet_file in sub_subdir.glob('*.parquet'):
                    file_stem = parquet_file.stem
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if parquet_file not in possible_paths:
                            if priority_search:
                                possible_paths.insert(0, parquet_file)
                            else:
                                possible_paths.append(parquet_file)
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            return path
    
    return None


def load_model_and_preprocessors(model_dir=MODEL_DIR):
    """
    学習済みモデルと前処理器を読み込む
    
    Args:
        model_dir: モデルが保存されているディレクトリ
    
    Returns:
        tuple: (model, scaler, label_encoder) または (None, None, None)
    """
    model_path = Path(model_dir)
    
    # モデルファイルを検索
    model_files = list(model_path.glob('*.h5')) + list(model_path.glob('*.keras'))
    if not model_files:
        # /kaggle/inputからも検索
        if IS_KAGGLE:
            input_dir = Path('/kaggle/input')
            if input_dir.exists():
                for item in input_dir.iterdir():
                    if item.is_dir():
                        model_files.extend(list((item / '*.h5').parent.glob('*.h5')))
                        model_files.extend(list((item / '*.keras').parent.glob('*.keras')))
    
    if not model_files:
        print(f"エラー: モデルファイルが見つかりません: {model_dir}")
        return None, None, None
    
    # 最初に見つかったモデルファイルを使用
    model_file = model_files[0]
    print(f"モデルファイルを読み込みます: {model_file}")
    
    try:
        model = keras.models.load_model(model_file)
        print(f"✓ モデルを読み込みました: {model_file.name}")
    except Exception as e:
        print(f"エラー: モデルの読み込みに失敗しました: {e}")
        return None, None, None
    
    # 前処理器を読み込む
    scaler = None
    label_encoder = None
    
    scaler_file = model_path / 'scaler.pkl'
    if not scaler_file.exists() and IS_KAGGLE:
        # /kaggle/inputからも検索
        input_dir = Path('/kaggle/input')
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_dir():
                    alt_scaler = item / 'scaler.pkl'
                    if alt_scaler.exists():
                        scaler_file = alt_scaler
                        break
    
    if scaler_file.exists():
        try:
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            print(f"✓ StandardScalerを読み込みました: {scaler_file.name}")
        except Exception as e:
            print(f"警告: StandardScalerの読み込みに失敗しました: {e}")
    
    label_encoder_file = model_path / 'label_encoder.pkl'
    if not label_encoder_file.exists() and IS_KAGGLE:
        # /kaggle/inputからも検索
        input_dir = Path('/kaggle/input')
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_dir():
                    alt_le = item / 'label_encoder.pkl'
                    if alt_le.exists():
                        label_encoder_file = alt_le
                        break
    
    if label_encoder_file.exists():
        try:
            with open(label_encoder_file, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"✓ LabelEncoderを読み込みました: {label_encoder_file.name}")
        except Exception as e:
            print(f"警告: LabelEncoderの読み込みに失敗しました: {e}")
    
    return model, scaler, label_encoder


def normalize_action_label(action):
    """
    ラベル名を正規化する（sniff__ → sniffなど）
    
    Args:
        action: 行動ラベル（文字列）
    
    Returns:
        normalized_action: 正規化されたラベル
    """
    if not isinstance(action, str):
        return action
    
    # アンダースコアの重複を削除（sniff__ → sniff）
    normalized = action.replace('__', '_')
    # 先頭・末尾のアンダースコアを削除
    normalized = normalized.strip('_')
    # 連続するアンダースコアを1つに統一
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    
    return normalized


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
            subdirs = [d for d in subdir.iterdir() if d.is_dir()]
            
            # サブディレクトリ内のファイルを直接検索
            for sub_subdir in subdirs:
                # サブディレクトリ内のすべてのParquetファイルを検索
                for parquet_file in sub_subdir.glob('*.parquet'):
                    file_stem = parquet_file.stem
                    if (file_stem == video_id_str or 
                        video_id_str in file_stem or
                        any(video_id_str in part for part in file_stem.split('_'))):
                        if parquet_file not in possible_paths:
                            possible_paths.append(parquet_file)
            
            # Parquetファイルも再帰的に検索
            for parquet_file in subdir.rglob('*.parquet'):
                file_stem = parquet_file.stem
                if file_stem == video_id_str:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                elif video_id_str in file_stem:
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
                elif any(video_id_str in part for part in file_stem.split('_')):
                    if parquet_file not in possible_paths:
                        possible_paths.append(parquet_file)
    
    # ファイルを検索（CSVとParquetの両方）
    for path in possible_paths:
        if path.exists() and (path.suffix.lower() == '.csv' or path.suffix.lower() == '.parquet'):
            return path
    
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
    
    # カラム名のマッピング辞書
    column_mapping = {
        'behavior': 'action',
        'behavior_type': 'action',
        'action_type': 'action',
        'behavior_label': 'action',
        'action_label': 'action',
        'label': 'action',
        'agent': 'agent_id',
        'mouse_id': 'agent_id',
        'subject_id': 'agent_id',
        'target': 'target_id',
        'target_mouse_id': 'target_id',
        'object_id': 'target_id',
        'start': 'start_frame',
        'start_time': 'start_frame',
        'begin_frame': 'start_frame',
        'stop': 'stop_frame',
        'end_frame': 'stop_frame',
        'end_time': 'stop_frame',
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
    
    if renamed_columns:
        df = df.rename(columns=renamed_columns)
    
    # 重複カラムの処理
    if 'action' in df.columns and 'behavior' in df.columns:
        if df['behavior'].notna().sum() > df['action'].notna().sum():
            df['action'] = df['action'].fillna(df['behavior'])
        df = df.drop(columns=['behavior'])
    
    return df


def get_ground_truth_labels(test_df, data_dir):
    """
    テストデータの各行に対して、アノテーションファイルから正解ラベルを取得
    
    Args:
        test_df: テストデータのDataFrame
        data_dir: データディレクトリ
    
    Returns:
        ground_truth_labels: 正解ラベルのリスト（test_dfと同じ順序）
    """
    ground_truth_labels = []
    
    print("\n正解ラベルの取得中...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="正解ラベル取得", disable=not TQDM_AVAILABLE):
        video_id = row.get('video_id', '')
        start_frame = row.get('start_frame', 0)
        stop_frame = row.get('stop_frame', 0)
        target_id = row.get('target_id', None)
        agent_id = row.get('agent_id', None)
        
        # アノテーションファイルを検索
        annotation_file = find_annotation_file(data_dir, video_id, debug=False)
        
        if annotation_file is None:
            ground_truth_labels.append(None)
            continue
        
        try:
            # アノテーションファイルを読み込む
            if annotation_file.suffix.lower() == '.parquet':
                annotation_df = pd.read_parquet(annotation_file)
            else:
                annotation_df = pd.read_csv(annotation_file)
            
            # カラム名を統一
            annotation_df = normalize_annotation_columns(annotation_df)
            
            if 'action' not in annotation_df.columns:
                ground_truth_labels.append(None)
                continue
            
            # フレーム範囲でフィルタリング
            if 'start_frame' in annotation_df.columns and 'stop_frame' in annotation_df.columns:
                # start_frameとstop_frameが重複するアノテーションを検索
                matching_annotations = annotation_df[
                    (annotation_df['start_frame'] <= stop_frame) & 
                    (annotation_df['stop_frame'] >= start_frame)
                ]
            else:
                matching_annotations = annotation_df
            
            # target_idとagent_idでフィルタリング（指定されている場合）
            if target_id is not None and 'target_id' in matching_annotations.columns:
                matching_annotations = matching_annotations[matching_annotations['target_id'] == target_id]
            
            if agent_id is not None and 'agent_id' in matching_annotations.columns:
                matching_annotations = matching_annotations[matching_annotations['agent_id'] == agent_id]
            
            # 最も多く出現するactionを正解ラベルとする
            if len(matching_annotations) > 0:
                action_counts = matching_annotations['action'].value_counts()
                most_common_action = action_counts.index[0]
                ground_truth_labels.append(most_common_action)
            else:
                ground_truth_labels.append(None)
        
        except Exception as e:
            if idx < 5:
                print(f"  警告: row_id={idx}の正解ラベル取得に失敗: {e}")
            ground_truth_labels.append(None)
    
    found_count = sum(1 for label in ground_truth_labels if label is not None)
    print(f"  正解ラベル取得結果: {found_count}/{len(test_df)} ({found_count/len(test_df)*100:.1f}%)")
    
    return ground_truth_labels


def calculate_f1_score_metrics(y_true, y_pred, label_encoder=None):
    """
    F1スコアを計算（マクロ平均、マイクロ平均、重み付き平均）
    
    Args:
        y_true: 正解ラベルのリスト
        y_pred: 予測ラベルのリスト
        label_encoder: LabelEncoderインスタンス（オプション）
    
    Returns:
        dict: 評価結果の辞書
    """
    # Noneを除外
    valid_indices = [i for i in range(len(y_true)) if y_true[i] is not None and y_pred[i] is not None]
    
    if len(valid_indices) == 0:
        return {
            'f1_macro': 0.0,
            'f1_micro': 0.0,
            'f1_weighted': 0.0,
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'valid_samples': 0,
            'total_samples': len(y_true)
        }
    
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    # ラベルを正規化
    y_true_valid = [normalize_action_label(label) for label in y_true_valid]
    y_pred_valid = [normalize_action_label(label) for label in y_pred_valid]
    
    # すべてのユニークなラベルを取得
    all_labels = sorted(set(y_true_valid + y_pred_valid))
    
    # F1スコアを計算
    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', labels=all_labels, zero_division=0)
    f1_micro = f1_score(y_true_valid, y_pred_valid, average='micro', labels=all_labels, zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted', labels=all_labels, zero_division=0)
    
    # 精度を計算
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    
    # 適合率と再現率を計算
    precision_macro = precision_score(y_true_valid, y_pred_valid, average='macro', labels=all_labels, zero_division=0)
    recall_macro = recall_score(y_true_valid, y_pred_valid, average='macro', labels=all_labels, zero_division=0)
    
    # クラスごとのF1スコア
    f1_per_class = f1_score(y_true_valid, y_pred_valid, average=None, labels=all_labels, zero_division=0)
    f1_per_class_dict = {label: score for label, score in zip(all_labels, f1_per_class)}
    
    # 混同行列
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=all_labels)
    
    return {
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_per_class': f1_per_class_dict,
        'confusion_matrix': cm.tolist(),
        'labels': all_labels,
        'valid_samples': len(valid_indices),
        'total_samples': len(y_true)
    }


def evaluate_predictions(predictions, test_metadata, test_df, data_dir, label_encoder):
    """
    予測結果を評価する
    
    Args:
        predictions: 予測結果（確率分布）
        test_metadata: テストデータのメタデータ
        test_df: テストデータのDataFrame
        data_dir: データディレクトリ
        label_encoder: LabelEncoderインスタンス
    
    Returns:
        dict: 評価結果の辞書
    """
    print("\n" + "=" * 60)
    print("予測結果の評価")
    print("=" * 60)
    
    # 予測ラベルを取得
    if predictions is None or len(predictions) == 0:
        print("エラー: 予測結果がありません")
        return None
    
    predicted_classes = np.argmax(predictions, axis=1)
    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
    else:
        predicted_labels = [f'class_{cls}' for cls in predicted_classes]
    
    # ラベルを正規化
    predicted_labels = [normalize_action_label(label) for label in predicted_labels]
    
    # 正解ラベルを取得
    ground_truth_labels = get_ground_truth_labels(test_df, data_dir)
    
    # 予測ラベルと正解ラベルの順序を合わせる
    # test_metadataの順序に基づいてマッピング
    aligned_predicted = []
    aligned_ground_truth = []
    
    for i, metadata in enumerate(test_metadata):
        row_id = metadata.get('row_id', i)
        # test_dfから該当する行を検索
        if 'row_id' in test_df.columns:
            test_rows = test_df[test_df['row_id'] == row_id]
        else:
            # row_idカラムがない場合はインデックスで対応
            if i < len(test_df):
                test_rows = test_df.iloc[[i]]
            else:
                continue
        
        if len(test_rows) > 0:
            test_idx = test_rows.index[0]
            if test_idx < len(ground_truth_labels):
                aligned_predicted.append(predicted_labels[i] if i < len(predicted_labels) else 'unknown')
                aligned_ground_truth.append(ground_truth_labels[test_idx])
        else:
            # test_dfに見つからない場合は、インデックスで対応
            if i < len(ground_truth_labels):
                aligned_predicted.append(predicted_labels[i] if i < len(predicted_labels) else 'unknown')
                aligned_ground_truth.append(ground_truth_labels[i])
    
    if len(aligned_predicted) == 0:
        print("エラー: 予測ラベルと正解ラベルのマッピングに失敗しました")
        return None
    
    # F1スコアを計算
    evaluation_results = calculate_f1_score_metrics(aligned_ground_truth, aligned_predicted, label_encoder)
    
    # 結果を表示
    print("\n評価結果:")
    print(f"  有効サンプル数: {evaluation_results['valid_samples']}/{evaluation_results['total_samples']}")
    print(f"  精度 (Accuracy): {evaluation_results['accuracy']:.4f}")
    print(f"  F1スコア (マクロ平均): {evaluation_results['f1_macro']:.4f}")
    print(f"  F1スコア (マイクロ平均): {evaluation_results['f1_micro']:.4f}")
    print(f"  F1スコア (重み付き平均): {evaluation_results['f1_weighted']:.4f}")
    print(f"  適合率 (マクロ平均): {evaluation_results['precision_macro']:.4f}")
    print(f"  再現率 (マクロ平均): {evaluation_results['recall_macro']:.4f}")
    
    # クラスごとのF1スコアを表示
    if evaluation_results['f1_per_class']:
        print("\nクラスごとのF1スコア:")
        sorted_f1 = sorted(evaluation_results['f1_per_class'].items(), key=lambda x: x[1], reverse=True)
        for label, score in sorted_f1[:10]:  # 上位10件を表示
            print(f"  {label}: {score:.4f}")
    
    # 混同行列を表示（小さい場合のみ）
    if evaluation_results['confusion_matrix'] and len(evaluation_results['labels']) <= 15:
        print("\n混同行列（最初の10クラス）:")
        cm = np.array(evaluation_results['confusion_matrix'])
        labels = evaluation_results['labels'][:10]
        cm_subset = cm[:10, :10]
        print("  予測\\正解", end="")
        for label in labels:
            print(f"\t{label[:8]}", end="")
        print()
        for i, label in enumerate(labels):
            print(f"  {label[:8]}", end="")
            for j in range(len(labels)):
                print(f"\t{cm_subset[i, j]}", end="")
            print()
    
    return evaluation_results


def check_frame_overlaps(test_df):
    """
    スタートストップフレームが重複していないかを調べる
    
    Args:
        test_df: テストデータのDataFrame
    
    Returns:
        overlap_info: 重複情報の辞書 {row_id: [重複しているrow_idのリスト]}
    """
    overlap_info = {}
    
    if 'start_frame' not in test_df.columns or 'stop_frame' not in test_df.columns:
        return overlap_info
    
    # video_idごとにグループ化して重複をチェック
    if 'video_id' in test_df.columns:
        for video_id in test_df['video_id'].unique():
            video_rows = test_df[test_df['video_id'] == video_id].copy()
            
            for idx1, row1 in video_rows.iterrows():
                row_id1 = row1.get('row_id', idx1)
                start1 = row1['start_frame']
                stop1 = row1['stop_frame']
                
                overlapping_rows = []
                for idx2, row2 in video_rows.iterrows():
                    if idx1 == idx2:
                        continue
                    
                    row_id2 = row2.get('row_id', idx2)
                    start2 = row2['start_frame']
                    stop2 = row2['stop_frame']
                    
                    # フレーム範囲が重複しているかチェック
                    if not (stop1 < start2 or stop2 < start1):
                        overlapping_rows.append(row_id2)
                
                if overlapping_rows:
                    overlap_info[row_id1] = overlapping_rows
    else:
        # video_idがない場合は全体でチェック
        for idx1, row1 in test_df.iterrows():
            row_id1 = row1.get('row_id', idx1)
            start1 = row1['start_frame']
            stop1 = row1['stop_frame']
            
            overlapping_rows = []
            for idx2, row2 in test_df.iterrows():
                if idx1 == idx2:
                    continue
                
                row_id2 = row2.get('row_id', idx2)
                start2 = row2['start_frame']
                stop2 = row2['stop_frame']
                
                if not (stop1 < start2 or stop2 < start1):
                    overlapping_rows.append(row_id2)
            
            if overlapping_rows:
                overlap_info[row_id1] = overlapping_rows
    
    return overlap_info


def resolve_overlapping_predictions(predictions_dict, overlap_info, threshold=0.5):
    """
    重複しているフレーム範囲に対して、確率が閾値以上で大きい方を残す
    
    Args:
        predictions_dict: {row_id: prediction_probability}の形式
        overlap_info: 重複情報の辞書
        threshold: 確率の閾値（デフォルト: 0.5）
    
    Returns:
        resolved_predictions: 重複解決後の予測辞書
    """
    resolved_predictions = predictions_dict.copy()
    processed = set()
    
    for row_id, overlapping_rows in overlap_info.items():
        if row_id in processed:
            continue
        
        # 重複している行の予測確率を比較
        candidates = [row_id] + overlapping_rows
        valid_candidates = []
        
        for candidate_id in candidates:
            if candidate_id in predictions_dict:
                pred_prob = predictions_dict[candidate_id]
                # 最大確率を取得
                max_prob = np.max(pred_prob) if isinstance(pred_prob, np.ndarray) else pred_prob
                if max_prob >= threshold:
                    valid_candidates.append((candidate_id, max_prob))
        
        if len(valid_candidates) > 1:
            # 確率が最大のものを残す
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            winner_id = valid_candidates[0][0]
            
            # 他の重複行の予測を削除または警告
            for candidate_id, _ in valid_candidates[1:]:
                if candidate_id in resolved_predictions:
                    # 確率が低い方は削除（またはデフォルト値に設定）
                    del resolved_predictions[candidate_id]
                    processed.add(candidate_id)
        
        processed.add(row_id)
    
    return resolved_predictions


def validate_target_id(test_row, annotation_df=None, data_dir=None, video_id=None):
    """
    target_idがセルフかどうかが一致しているか確認
    
    Args:
        test_row: テストデータの1行（Seriesまたはdict）
        annotation_df: アノテーションデータのDataFrame（オプション）
        data_dir: データディレクトリ（オプション、アノテーションファイルを読み込むため）
        video_id: 動画ID（オプション）
    
    Returns:
        tuple: (is_valid, warning_message)
            is_valid: 検証が成功したかどうか
            warning_message: 警告メッセージ（検証失敗時）
    """
    test_target_id = test_row.get('target_id', None)
    
    # target_idがない場合は検証をスキップ
    if test_target_id is None:
        return True, None
    
    # アノテーションデータを取得
    if annotation_df is None and data_dir is not None and video_id is not None:
        # アノテーションファイルを読み込む（簡易版）
        # 実際の実装では、load_annotation_data関数を使用することを推奨
        pass
    
    if annotation_df is not None and len(annotation_df) > 0:
        # アノテーションデータのtarget_idを確認
        annotation_target_ids = annotation_df['target_id'].unique() if 'target_id' in annotation_df.columns else []
        
        # test_target_idがselfかどうかを確認
        test_is_self = (test_target_id == 'self' or str(test_target_id).lower() == 'self')
        
        # アノテーションデータのtarget_idがselfかどうかを確認
        annotation_has_self = any(
            str(tid).lower() == 'self' or tid == 'self' 
            for tid in annotation_target_ids
        )
        
        # 一致していない場合は警告
        if test_is_self != annotation_has_self:
            warning_msg = (
                f"target_idの不一致を検出: "
                f"test_rowのtarget_id={test_target_id} (self={test_is_self}), "
                f"annotationのtarget_idにselfが含まれる={annotation_has_self}"
            )
            return False, warning_msg
    
    return True, None


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
            
            # CSVまたはParquetファイルを読み込む
            if tracking_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(tracking_file)
            else:
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
                    padding = np.zeros((sequence_length, len(feature_cols)), dtype=np.float32)
                    sequences = np.array([padding], dtype=np.float32)
                else:
                    test_metadata.append(row.to_dict())
                    continue
            
            # 正規化
            if scaler is not None:
                n_samples, seq_len, n_features = sequences.shape
                sequences_reshaped = sequences.reshape(-1, n_features)
                sequences_scaled = scaler.transform(sequences_reshaped).astype(np.float32)
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


def create_submission(predictions, test_metadata, label_encoder, test_df=None, output_path=None):
    """
    提出ファイルを作成（sample_submission.csvと完全に同じ形式・順序）
    
    Args:
        predictions: 予測結果（確率分布）、Noneの場合はデフォルト予測を使用
        test_metadata: テストデータのメタデータ
        label_encoder: LabelEncoderインスタンス
        test_df: テストデータのDataFrame（row_idの順序を保証するため）
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
    
    # sample_submission.csvの形式を確認（必須）
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
    
    if sample_submission_path is None or not sample_submission_path.exists():
        raise FileNotFoundError(
            f"sample_submission.csvが見つかりません。提出形式を確認できません。\n"
            f"確認したパス: {possible_sample_paths}"
        )
    
    # sample_submission.csvを読み込んで形式を確認
    sample_df = pd.read_csv(sample_submission_path)
    print(f"\nsample_submission.csvの形式を確認:")
    print(f"  カラム: {list(sample_df.columns)}")
    print(f"  行数: {len(sample_df)}")
    print(f"  データ型:")
    for col in sample_df.columns:
        print(f"    {col}: {sample_df[col].dtype}")
    
    # 予測がない場合の処理
    if predictions is None or len(predictions) == 0:
        print("Warning: No predictions provided, using default action")
        if label_encoder is not None and hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
            default_action = label_encoder.classes_[0]
        else:
            default_action = 'unknown'
        predicted_labels = [default_action] * len(sample_df)
    else:
        # 重複フレームの検出
        print("\n重複フレームの検出中...")
        overlap_info = check_frame_overlaps(test_df)
        if overlap_info:
            print(f"  重複フレームを検出: {len(overlap_info)}件")
            for row_id, overlapping_rows in list(overlap_info.items())[:5]:
                print(f"    row_id={row_id}: {overlapping_rows}と重複")
        else:
            print("  重複フレームは検出されませんでした")
        
        # 予測クラスを取得
        predicted_classes = np.argmax(predictions, axis=1)
        
        # ラベルをデコード（文字列に変換）
        if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
        else:
            predicted_labels = [f'class_{cls}' for cls in predicted_classes]
        
        # ラベル名を正規化
        print("\nラベル名の正規化中...")
        normalized_count = 0
        for i, label in enumerate(predicted_labels):
            normalized = normalize_action_label(label)
            if normalized != label:
                normalized_count += 1
                if normalized_count <= 5:
                    print(f"  {label} → {normalized}")
            predicted_labels[i] = normalized
        
        if normalized_count > 0:
            print(f"  正規化されたラベル数: {normalized_count}/{len(predicted_labels)}")
        
        # 重複フレームの解決（確率が閾値以上で大きい方を残す）
        if overlap_info:
            print("\n重複フレームの解決中...")
            # 予測確率の辞書を作成
            predictions_dict = {}
            row_id_to_index = {}  # row_id -> index のマッピング
            for i, metadata in enumerate(test_metadata):
                row_id = metadata.get('row_id', i)
                row_id_to_index[row_id] = i
                if i < len(predictions):
                    predictions_dict[row_id] = predictions[i]
            
            # 重複解決
            resolved_predictions = resolve_overlapping_predictions(
                predictions_dict, overlap_info, threshold=0.5
            )
            
            # 解決後の予測を反映
            resolved_labels = {}
            removed_row_ids = set()
            for row_id, pred_prob in resolved_predictions.items():
                pred_class = np.argmax(pred_prob)
                if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
                    label = label_encoder.inverse_transform([pred_class])[0]
                else:
                    label = f'class_{pred_class}'
                resolved_labels[row_id] = normalize_action_label(label)
            
            # 削除されたrow_idを記録
            for row_id in predictions_dict:
                if row_id not in resolved_predictions:
                    removed_row_ids.add(row_id)
            
            if removed_row_ids:
                print(f"  重複により削除されたrow_id: {list(removed_row_ids)[:10]}")
            
            # 解決後のラベルで更新
            updated_count = 0
            for i, metadata in enumerate(test_metadata):
                row_id = metadata.get('row_id', i)
                if row_id in resolved_labels:
                    predicted_labels[i] = resolved_labels[row_id]
                    updated_count += 1
                elif row_id in removed_row_ids:
                    # 削除された行はデフォルト値を使用
                    default_action = predicted_labels[0] if len(predicted_labels) > 0 else 'unknown'
                    predicted_labels[i] = normalize_action_label(default_action)
            
            print(f"  重複解決により更新された予測数: {updated_count}")
    
    # test_dfがある場合は、row_idの順序を保証
    if test_df is not None and 'row_id' in test_df.columns:
        # test_dfのrow_id順序を使用
        submission_df = sample_df.copy()
        
        # 予測結果をマッピング（test_metadataの順序に基づく）
        prediction_dict = {}
        for i, metadata in enumerate(test_metadata):
            row_id = metadata.get('row_id', i)
            pred_idx = min(i, len(predicted_labels) - 1) if len(predicted_labels) > 0 else 0
            action = predicted_labels[pred_idx] if len(predicted_labels) > 0 else 'unknown'
            prediction_dict[row_id] = action
        
        # sample_dfのrow_idに基づいて予測を設定
        if 'action' in submission_df.columns:
            submission_df['action'] = submission_df['row_id'].map(prediction_dict)
            # マッピングできなかった行はデフォルト値を使用
            if submission_df['action'].isna().any():
                default_action = predicted_labels[0] if len(predicted_labels) > 0 else 'unknown'
                submission_df['action'] = submission_df['action'].fillna(default_action)
            # 最終的なラベル正規化を適用
            submission_df['action'] = submission_df['action'].apply(normalize_action_label)
        else:
            # actionカラムがない場合は追加
            submission_df['action'] = submission_df['row_id'].map(prediction_dict)
            if submission_df['action'].isna().any():
                default_action = predicted_labels[0] if len(predicted_labels) > 0 else 'unknown'
                submission_df['action'] = submission_df['action'].fillna(default_action)
            # 最終的なラベル正規化を適用
            submission_df['action'] = submission_df['action'].apply(normalize_action_label)
    else:
        # test_dfがない場合は、test_metadataの順序を使用
        submission_data = []
        for i, metadata in enumerate(test_metadata):
            row_id = metadata.get('row_id', i)
            pred_idx = min(i, len(predicted_labels) - 1) if len(predicted_labels) > 0 else 0
            action = predicted_labels[pred_idx] if len(predicted_labels) > 0 else 'unknown'
            
            # ラベル正規化を適用
            normalized_action = normalize_action_label(action)
            
            row = {
                'row_id': row_id,
                'action': normalized_action,
            }
            submission_data.append(row)
        
        submission_df = pd.DataFrame(submission_data)
        
        # sample_dfのrow_id順序に合わせる
        if 'row_id' in sample_df.columns:
            submission_df = sample_df[['row_id']].merge(
                submission_df[['row_id', 'action']],
                on='row_id',
                how='left',
                suffixes=('', '_pred')
            )
            # actionカラムを統合
            if 'action_pred' in submission_df.columns:
                submission_df['action'] = submission_df['action_pred'].fillna(submission_df.get('action', 'unknown'))
                submission_df = submission_df.drop(columns=['action_pred'])
            # 欠損値の処理
            if submission_df['action'].isna().any():
                default_action = predicted_labels[0] if len(predicted_labels) > 0 else 'unknown'
                submission_df['action'] = submission_df['action'].fillna(default_action)
    
    # sample_dfの列順序と完全に一致させる
    submission_df = submission_df[sample_df.columns.tolist()]
    
    # データ型をsample_dfと完全に一致させる
    for col in submission_df.columns:
        if col in sample_df.columns:
            submission_df[col] = submission_df[col].astype(sample_df[col].dtype)
    
    # 行数が一致することを確認
    if len(submission_df) != len(sample_df):
        print(f"警告: 提出ファイルの行数({len(submission_df)})がsample_submission.csvの行数({len(sample_df)})と一致しません")
        # sample_dfの行数に合わせる
        if len(submission_df) < len(sample_df):
            # 不足分を追加（デフォルト値で）
            default_action = predicted_labels[0] if len(predicted_labels) > 0 else 'unknown'
            missing_rows = sample_df[~sample_df['row_id'].isin(submission_df['row_id'])].copy()
            if 'action' in missing_rows.columns:
                missing_rows['action'] = default_action
            submission_df = pd.concat([submission_df, missing_rows], ignore_index=True)
            submission_df = submission_df.sort_values('row_id').reset_index(drop=True)
            submission_df = submission_df[sample_df.columns.tolist()]
        else:
            # 余分な行を削除
            submission_df = submission_df[submission_df['row_id'].isin(sample_df['row_id'])]
            submission_df = submission_df.sort_values('row_id').reset_index(drop=True)
    
    # CSVファイルに保存
    submission_df.to_csv(output_path, index=False)
    print(f"\n提出ファイルを保存しました: {output_path}")
    print(f"提出データ形状: {submission_df.shape}")
    print(f"提出ファイルのカラム: {list(submission_df.columns)}")
    print(f"sample_submission.csvのカラム: {list(sample_df.columns)}")
    print(f"カラム順序が一致: {list(submission_df.columns) == list(sample_df.columns)}")
    print("\n提出ファイルの最初の5行:")
    print(submission_df.head())
    print("\n提出ファイルの最後の5行:")
    print(submission_df.tail())
    
    return submission_df


def main():
    """
    メイン実行関数（推論専用）
    """
    total_start_time = time.time()
    
    print("=" * 60)
    print("MABe Mouse Behavior Detection - 1D CNN (推論用)")
    print("=" * 60)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Kaggle環境: {IS_KAGGLE}")
    print(f"データディレクトリ: {DATA_DIR}")
    print(f"モデルディレクトリ: {MODEL_DIR}")
    
    # GPU情報を表示
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU検出: {len(gpus)}基")
        else:
            print("GPU: 検出されませんでした（CPUで実行）")
    except Exception as e:
        print(f"GPU情報取得エラー: {e}")
    
    try:
        # モデルと前処理器の読み込み
        print("\n[1/3] モデルと前処理器の読み込み")
        model_load_start = time.time()
        
        model, scaler, label_encoder = load_model_and_preprocessors(MODEL_DIR)
        
        if model is None:
            print("エラー: モデルの読み込みに失敗しました。学習済みモデルが必要です。")
            return
        
        print(f"モデル読み込み時間: {timedelta(seconds=int(time.time() - model_load_start))}")
        
        # テストデータの読み込み
        print("\n[2/3] テストデータの読み込み")
        data_load_start = time.time()
        
        test_csv = Path(DATA_DIR) / 'test.csv'
        
        # 代替パスも試行
        if not test_csv.exists():
            for subdir in ['test', 'data']:
                alt_path = Path(DATA_DIR) / subdir / 'test.csv'
                if alt_path.exists():
                    test_csv = alt_path
                    break
        
        if not test_csv.exists():
            print(f"エラー: test.csvが見つかりません: {test_csv}")
            print(f"データディレクトリの内容を確認中: {DATA_DIR}")
            data_path = Path(DATA_DIR)
            if data_path.exists():
                files = list(data_path.glob('*.csv'))
                dirs = [d.name for d in data_path.iterdir() if d.is_dir()]
                print(f"  見つかったCSVファイル: {[f.name for f in files[:5]]}")
                print(f"  見つかったディレクトリ: {dirs[:5]}")
            return
        
        test_df = pd.read_csv(test_csv)
        print(f"テストデータ: {len(test_df)}行")
        print(f"テストデータのカラム: {list(test_df.columns)[:10]}")
        print(f"データ読み込み時間: {timedelta(seconds=int(time.time() - data_load_start))}")
        
        # 予測の実行
        print("\n[3/3] 予測の実行")
        predict_start = time.time()
        
        predictions, test_metadata = predict_with_1dcnn(
            model, test_df, DATA_DIR, scaler, sequence_length=SEQUENCE_LENGTH
        )
        
        predict_elapsed = time.time() - predict_start
        print(f"予測時間: {timedelta(seconds=int(predict_elapsed))}")
        print(f"予測結果数: {len(predictions) if predictions is not None else 0}")
        print(f"テストメタデータ数: {len(test_metadata)}")
        
        # 提出ファイルの生成
        print("\n提出ファイルの生成")
        submission_path = '/kaggle/working/submission.csv' if IS_KAGGLE else os.path.join(OUTPUT_DIR, 'submission.csv')
        
        submission_df = create_submission(
            predictions, test_metadata, label_encoder, test_df=test_df, output_path=submission_path
        )
        
        # 評価の実行（オプション、正解データがある場合のみ）
        print("\n予測結果の評価（オプション）")
        try:
            evaluation_results = evaluate_predictions(
                predictions, test_metadata, test_df, DATA_DIR, label_encoder
            )
            
            # 評価結果を保存
            if evaluation_results is not None:
                eval_file = Path(OUTPUT_DIR) / 'evaluation_results.json'
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n評価結果を保存しました: {eval_file}")
        except Exception as e:
            print(f"評価の実行中にエラーが発生しました: {e}")
            print("（正解データがない場合は正常です）")
            import traceback
            if not IS_KAGGLE:  # ローカル環境でのみ詳細なエラーを表示
                traceback.print_exc()
        
        # 総実行時間を表示
        total_elapsed = time.time() - total_start_time
        print("\n" + "=" * 60)
        print("推論完了")
        print("=" * 60)
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"総実行時間: {timedelta(seconds=int(total_elapsed))}")
        print(f"提出ファイル: {submission_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

