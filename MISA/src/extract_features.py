# -*- coding: utf-8 -*-
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel, ViTImageProcessor, ViTModel
from PIL import Image
from data_loader import RedditUserDataset

# 配置路径
# 假设脚本运行在 MISA/src/ 目录下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'user_timelines_20260305_0113.json')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'processed_features.pkl')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor:
    def __init__(self):
        print(f"Initializing feature extractor (Device: {DEVICE})...")
        
        # 检查本地模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_roberta = os.path.join(current_dir, 'saved_models', 'roberta-base')
        local_vit = os.path.join(current_dir, 'saved_models', 'vit-base')
        
        # 确定模型来源
        roberta_source = local_roberta if os.path.exists(local_roberta) else 'roberta-base'
        vit_source = local_vit if os.path.exists(local_vit) else 'google/vit-base-patch16-224-in21k'
        
        print(f"Using RoBERTa from: {roberta_source}")
        print(f"Using ViT from: {vit_source}")

        # 1. Text Model (RoBERTa)
        print("Loading RoBERTa model...")
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(roberta_source)
            self.text_model = RobertaModel.from_pretrained(roberta_source).to(DEVICE)
            self.text_model.eval()
        except Exception as e:
            print(f"Failed to load RoBERTa: {e}")
            raise

        # 2. Image Model (ViT)
        print("Loading ViT model...")
        try:
            self.image_processor = ViTImageProcessor.from_pretrained(vit_source)
            self.vision_model = ViTModel.from_pretrained(vit_source).to(DEVICE)
            self.vision_model.eval()
        except Exception as e:
            print(f"Failed to load ViT: {e}")
            raise

    def extract_text_features(self, text):
        """Extract text features (768 dim)"""
        if not text or not text.strip():
            return torch.zeros(768).to(DEVICE)
            
        # Truncation length set to 512
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        # Use [CLS] token embedding as sentence representation
        return outputs.last_hidden_state[:, 0, :].squeeze(0)

    def extract_image_features(self, image_paths):
        """Extract image features (768 dim)"""
        if not image_paths:
            return torch.zeros(768).to(DEVICE)

        valid_images = []
        for path in image_paths:
            # Normalize path separators
            path = path.replace('\\', '/')
            # Fix path: dataset paths are relative, need to join with project root
            full_path = os.path.join(PROJECT_ROOT, path)
            # Normalize again for OS
            full_path = os.path.normpath(full_path)
            
            if os.path.exists(full_path):
                try:
                    image = Image.open(full_path).convert("RGB")
                    valid_images.append(image)
                except Exception as e:
                    print(f"Cannot read image {full_path}: {e}")
            else:
                # DEBUG: Print first few missing files to help debug
                if not hasattr(self, '_missing_file_printed'):
                    print(f"WARNING: Image file not found: {full_path}")
                    self._missing_file_printed = 0
                if self._missing_file_printed < 5:
                    print(f"MISSING: {full_path}")
                    self._missing_file_printed += 1
        
        if not valid_images:
            return torch.zeros(768).to(DEVICE)

        try:
            inputs = self.image_processor(images=valid_images, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
            
            # If multiple images, take mean
            # outputs.last_hidden_state shape: [batch_size, sequence_length, hidden_size]
            # Take [CLS] token (index 0)
            features = outputs.last_hidden_state[:, 0, :] 
            return torch.mean(features, dim=0)
        except Exception as e:
            print(f"Image feature extraction error: {e}")
            return torch.zeros(768).to(DEVICE)

    def process_dataset(self, dataset):
        processed_data = []
        
        print(f"Start processing {len(dataset)} users...")
        
        for i in tqdm(range(len(dataset))):
            user_data = dataset[i]
            username = user_data['username']
            timeline = user_data['timeline']
            
            user_features = []
            
            for week_data in timeline:
                # 1. Extract text features
                text_feat = self.extract_text_features(week_data['text_content'])
                
                # 2. Extract image features
                img_feat = self.extract_image_features(week_data['image_paths'])
                
                # 3. Behavior features (convert to Tensor)
                beh_dict = week_data['behavior_feats']
                beh_feat = torch.tensor([
                    beh_dict['avg_score'],
                    beh_dict['total_comments'],
                    beh_dict['post_count'],
                    beh_dict['active_hours']
                ], dtype=torch.float32).to(DEVICE)
                
                user_features.append({
                    'week_id': week_data['week_id'],
                    'text_feat': text_feat.cpu().numpy(), # Convert back to CPU and numpy for saving
                    'image_feat': img_feat.cpu().numpy(),
                    'behavior_feat': beh_feat.cpu().numpy()
                })
            
            processed_data.append({
                'username': username,
                'timeline_features': user_features
            })
            
        return processed_data

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    # 1. Load dataset
    print("Loading dataset...")
    dataset = RedditUserDataset(timeline_file_path=DATA_PATH)
    
    # 2. Initialize extractor
    extractor = FeatureExtractor()
    
    # 3. Process data
    final_data = extractor.process_dataset(dataset)
    
    # 4. Save results
    print(f"Saving processed features to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(final_data, f)
    print("Done!")

if __name__ == "__main__":
    main()
