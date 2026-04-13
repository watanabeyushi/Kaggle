import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetB0
from keras.optimizers import Adam, SGD
from keras.models import Model

# モデル構成パラメーター
SEQUENCE_LENGTH = 32  # タイムウィンドウのフレーム数
IMG_SIZE = 224        # EfficientNetの標準入力サイズ (特徴量マップとして整形した場合)
NUM_CLASSES = 15      # MABeの行動クラス数

def build_cnn_extractor(input_shape):
    """
    EfficientNet Backboneを特徴抽出器として定義する。
    """
    cnn_backbone = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape,
        # 正則化強化のためにdrop_connect_rateを調整可能 
        drop_connect_rate=0.4 
    )
    
    # フェーズ1: 転移学習の初期段階ではBackboneを完全にフリーズする
    cnn_backbone.trainable = False 
    
    inputs = keras.Input(shape=input_shape)#入力層の定義
    x = cnn_backbone(inputs, training=False)#定義したテンソルをバックボーンに入力
    
    # 空間情報を固定長ベクトルに集約 (CNNの出力次元に依存)
    x = layers.GlobalAveragePooling2D()(x)#グローバル平均プーリングを行う
    return Model(inputs, x, name="EfficientNet_Feature_Extractor")

def build_mabe_hybrid_model():
    """
    CNN-LSTMハイブリッドモデルを構築する。
    """
    # 1. 入力層の定義: (シーケンス長, 高さ, 幅, チャンネル)
    # ここでは、Egocentric特徴量マップを想定 (T, H, W, C)
    input_shape = (SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)
    sequence_input = layers.Input(shape=input_shape)
    
    # 2. TimeDistributed CNN: 各タイムステップで空間特徴を独立して抽出
    feature_extractor = build_cnn_extractor((IMG_SIZE, IMG_SIZE, 3))
    extracted_features = layers.TimeDistributed(feature_extractor)(sequence_input)
    
    # 3. Temporal Stream: LSTM/GRUで時間依存性を学習
    # 256ユニットを例とする。より複雑なタスクには双方向LSTM (Bidirectional LSTM) も有効 。
    # return_sequences=False: シーケンス全体から単一の行動予測を導出
    x = layers.LSTM(256, return_sequences=False, activation='tanh')(extracted_features)
    
    # 4. Custom Classification Head
    # 強い正則化 (例: 0.5) を適用 
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(sequence_input, outputs, name="MABe_EfficientNet_Hybrid")
    return model


    