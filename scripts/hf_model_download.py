import argparse
import os

from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login, snapshot_download
from typing import Literal

DATA_PATH = os.getenv('DATA_PATH', os.getcwd())

def download_model(model_name, save_dir=None, mode='model'):
    """
    Downloads a model from Hugging Face to a local folder.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub
        save_dir (str, optional): The directory to save the model to. If None, 
                                 creates a directory with the model name.
        mode (str): Download mode - 'model' uses AutoModel, 'snapshot' uses snapshot_download
    
    Returns:
        str: Path to the saved model
    """
    if save_dir is None:
        model_short_name = model_name.split('/')[-1]
        save_dir = os.path.join(DATA_PATH, model_short_name)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}. Skipping download.")
        return save_path
        
    print(f"Downloading model {model_name} to {save_path} using mode: {mode}...")
    
    if mode == 'snapshot':
        print(f"Using snapshot download mode...")
        snapshot_download(
            model_name,
            revision='main',
            ignore_patterns=['*.git*', '*README.md'],
            local_dir=save_path
        )
        print(f"Snapshot download completed to {save_path}")
    else:
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_path)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(save_path)
            print("Tokenizer downloaded and saved.")
        except Exception as e:
            print(f"Could not download tokenizer: {e}")
    
    print(f"Model downloaded successfully to {save_path}")
    return save_dir

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("--model", type=str, required=True, help="Model name from Hugging Face Hub")
    parser.add_argument("--output", type=str, default=None, help="Directory to save the model")
    parser.add_argument("--mode", type=str, choices=['model', 'snapshot'], default='model', 
                       help="Download mode: 'model' uses AutoModel, 'snapshot' downloads all files")
    
    args = parser.parse_args()
    
    download_model(args.model, args.output, args.mode)