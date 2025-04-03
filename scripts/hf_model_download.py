import argparse
import os

from transformers import AutoModel, AutoTokenizer


def download_model(model_name, save_dir=None):
    """
    Downloads a model from Hugging Face to a local folder.
    
    Args:
        model_name (str): The name of the model on Hugging Face Hub
        save_dir (str, optional): The directory to save the model to. If None, 
                                 creates a directory with the model name.
    
    Returns:
        str: Path to the saved model
    """
    if save_dir is None:
        model_short_name = model_name.split('/')[-1]
        save_dir = os.path.join(os.getcwd(), model_short_name)
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading model {model_name} to {save_dir}...")
    
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_dir)
        print("Tokenizer downloaded and saved.")
    except Exception as e:
        print(f"Could not download tokenizer: {e}")
    
    print(f"Model downloaded successfully to {save_dir}")
    return save_dir

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument("--model", type=str, required=True, help="Model name from Hugging Face Hub")
    parser.add_argument("--output", type=str, default=None, help="Directory to save the model")
    
    args = parser.parse_args()
    
    download_model(args.model, args.output)