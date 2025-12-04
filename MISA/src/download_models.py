# -*- coding: utf-8 -*-
import os
from transformers import RobertaTokenizer, RobertaModel, ViTImageProcessor, ViTModel

# Define save path: saved_models folder in the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_and_save(model_name, folder_name):
    print(f"Downloading: {model_name} ...")
    save_path = os.path.join(SAVE_DIR, folder_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        if 'roberta' in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
        else:
            processor = ViTImageProcessor.from_pretrained(model_name)
            model = ViTModel.from_pretrained(model_name)
            processor.save_pretrained(save_path)
            model.save_pretrained(save_path)
        
        print(f"Successfully saved to: {save_path}")
    except Exception as e:
        print(f"Download failed for {model_name}: {e}")

if __name__ == "__main__":
    print(f"Models will be saved to: {SAVE_DIR}")
    
    # 1. Download RoBERTa
    download_and_save('roberta-base', 'roberta-base')
    
    # 2. Download ViT
    download_and_save('google/vit-base-patch16-224-in21k', 'vit-base')
    
    print("\nAll models downloaded!")
    print(f"Please upload the '{SAVE_DIR}' folder to 'MISA/src/' on your server.")
