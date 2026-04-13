"""
EfficientNet-B0モデルをダウンロードしてローカルに保存するスクリプト
"""
import os
import torch
from pathlib import Path
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def download_efficientnet_model():
    """EfficientNet-B0モデルをダウンロードして保存"""
    # プロジェクトのルートディレクトリを取得
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    
    # modelsディレクトリが存在しない場合は作成
    models_dir.mkdir(exist_ok=True)
    
    # モデルファイルのパス
    model_path = models_dir / "efficientnet_b0_rwightman-7f5810bc.pth"
    
    # 既にダウンロード済みの場合はスキップ
    if model_path.exists():
        print(f"✓ Model already exists at: {model_path}")
        print("  Skipping download.")
        return str(model_path)
    
    print("Downloading EfficientNet-B0 model...")
    try:
        # モデルをダウンロードして読み込む
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 重みを保存
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model downloaded and saved to: {model_path}")
        return str(model_path)
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("  You may need to download it manually or use online download.")
        return None

if __name__ == "__main__":
    model_path = download_efficientnet_model()
    if model_path:
        print(f"\nModel is ready at: {model_path}")
        print("You can now use this model in your notebook.")
    else:
        print("\nFailed to download model. Please check your internet connection.")

