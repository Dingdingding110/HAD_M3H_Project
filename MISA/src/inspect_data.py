import pickle
import numpy as np
import os
import torch

def inspect_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_path = os.path.join(project_root, 'enhanced_reddit_data', 'processed_features.pkl')
    
    if not os.path.exists(features_path):
        print(f"File not found: {features_path}")
        return

    print(f"Loading {features_path}...")
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} users.")
    
    # Check first 5 users
    print("\n--- Checking Feature Variance ---")
    
    first_user = data[0]
    print(f"User 1: {first_user['username']}")
    u1_timeline = first_user['timeline_features']
    u1_text = u1_timeline[0]['text_feat']
    u1_img = u1_timeline[0]['image_feat']
    
    print(f"User 1 Week 1 Text Mean: {np.mean(u1_text):.6f}")
    print(f"User 1 Week 1 Image Mean: {np.mean(u1_img):.6f}")
    
    for i in range(1, 5):
        user = data[i]
        print(f"\nUser {i+1}: {user['username']}")
        timeline = user['timeline_features']
        text = timeline[0]['text_feat']
        img = timeline[0]['image_feat']
        
        print(f"Week 1 Text Mean: {np.mean(text):.6f}")
        print(f"Week 1 Image Mean: {np.mean(img):.6f}")
        
        # Check if identical to User 1
        if np.allclose(text, u1_text):
            print("WARNING: Text features identical to User 1!")
        else:
            print("Text features distinct.")
            
        if np.allclose(img, u1_img):
            print("WARNING: Image features identical to User 1!")
        else:
            print("Image features distinct.")

if __name__ == "__main__":
    inspect_data()
