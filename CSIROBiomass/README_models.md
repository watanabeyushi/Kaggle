# モデルファイルのダウンロードと使用方法

## 概要

このプロジェクトでは、EfficientNet-B0モデルをローカルに保存して使用できます。これにより、インターネット接続がない環境でも実行可能になります。

## モデルのダウンロード

### 方法1: ダウンロードスクリプトを使用（推奨）

```bash
cd CSIROBiomass
python download_models.py
```

このスクリプトは、EfficientNet-B0モデルを自動的にダウンロードして`models/`ディレクトリに保存します。

### 方法2: 手動ダウンロード

以下のURLからモデルファイルをダウンロードして、`models/`ディレクトリに保存してください：

- URL: `https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth`
- 保存先: `CSIROBiomass/models/efficientnet_b0_rwightman-7f5810bc.pth`

## ディレクトリ構造

```
CSIROBiomass/
├── csiro.ipynb
├── csiro_解説.md
├── download_models.py
├── models/
│   └── efficientnet_b0_rwightman-7f5810bc.pth  (ダウンロード後)
└── README_models.md
```

## モデル読み込みの優先順位

`csiro.ipynb`では、以下の順序でモデルを読み込みます：

1. **ローカルファイル** (`./models/efficientnet_b0_rwightman-7f5810bc.pth`)
2. **Kaggle入力データ** (`/kaggle/input/efficientnet-b0/...`)
3. **キャッシュ** (`~/.cache/torch/hub/checkpoints/...`)
4. **オンラインダウンロード** (インターネット接続が必要)
5. **重みなしモデル** (フォールバック、精度が低下する可能性)

## ローカル環境での実行

### 前提条件

1. EfficientNet-B0モデルをダウンロード済み
2. DINOv2モデルが利用可能（ローカル環境では別途ダウンロードが必要）

### 実行手順

1. モデルをダウンロード：
   ```bash
   python download_models.py
   ```

2. データを配置：
   - 訓練データを`data/train.csv`に配置
   - テストデータを`data/test.csv`に配置
   - 画像ファイルを適切な場所に配置

3. ノートブックを実行：
   - `csiro.ipynb`を開いて実行

## Kaggle環境での実行

Kaggle環境では、以下のいずれかの方法でモデルを利用できます：

1. **Kaggle入力データセット**: EfficientNet-B0モデルをKaggleの入力データセットとして追加
2. **自動ダウンロード**: インターネット接続が有効な場合、自動的にダウンロード
3. **キャッシュ**: 以前にダウンロードしたモデルがキャッシュに残っている場合

## トラブルシューティング

### モデルが見つからない

- `download_models.py`を実行してモデルをダウンロードしてください
- モデルファイルが`models/`ディレクトリに正しく配置されているか確認してください

### インターネット接続エラー

- ローカルファイルから読み込むように設定されているため、モデルをダウンロード済みであれば問題ありません
- 初回実行時のみインターネット接続が必要です

### パスの問題

- ノートブックは自動的にローカル環境とKaggle環境を検出します
- カスタムパスを使用する場合は、コード内のパス設定を変更してください

## 注意事項

- モデルファイルのサイズは約20MBです
- Gitリポジトリに含める場合は、`.gitignore`の設定を確認してください
- 大きなモデルファイルは通常Gitに含めませんが、このプロジェクトでは含めることを想定しています

