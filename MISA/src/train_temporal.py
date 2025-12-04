import os
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse

# Import our modules
from config import get_config
from models import TemporalMISA
from utils import DiffLoss, MSE, CMD

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TemporalDataset(Dataset):
    def __init__(self, features_path, label_map=None):
        """
        Args:
            features_path: Path to processed_features.pkl
            label_map: Dictionary mapping username to label (int). 
                       If None, generates dummy labels for testing.
        """
        with open(features_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        self.label_map = label_map
        
        # Filter data if label_map is provided
        if self.label_map:
            self.data = []
            skipped = 0
            for user_data in raw_data:
                if user_data['username'] in self.label_map:
                    self.data.append(user_data)
                else:
                    skipped += 1
            print(f"Loaded {len(self.data)} users with labels. Skipped {skipped} users without labels.")
        else:
            self.data = raw_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_data = self.data[idx]
        username = user_data['username']
        timeline = user_data['timeline_features']
        
        # Extract sequences
        text_seq = [torch.tensor(t['text_feat']) for t in timeline]
        image_seq = [torch.tensor(t['image_feat']) for t in timeline]
        behavior_seq = [torch.tensor(t['behavior_feat']) for t in timeline]
        
        # Stack into tensors: [seq_len, feat_dim]
        text_seq = torch.stack(text_seq)
        image_seq = torch.stack(image_seq)
        behavior_seq = torch.stack(behavior_seq)
        
        # Get Label
        if self.label_map and username in self.label_map:
            label = self.label_map[username]
        else:
            # Dummy label: 0 or 1 based on hash of username
            label = hash(username) % 2
            
        return text_seq, image_seq, behavior_seq, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    # batch is a list of tuples (text, image, behavior, label)
    text_seqs, image_seqs, behavior_seqs, labels = zip(*batch)
    
    # Pad sequences
    # batch_first=True -> [batch_size, max_seq_len, feat_dim]
    text_padded = pad_sequence(text_seqs, batch_first=True)
    image_padded = pad_sequence(image_seqs, batch_first=True)
    behavior_padded = pad_sequence(behavior_seqs, batch_first=True)
    
    labels = torch.stack(labels)
    
    return text_padded, image_padded, behavior_padded, labels

def get_misa_losses(model, config, diff_loss_fn, recon_loss_fn, cmd_loss_fn):
    """Calculate auxiliary losses for MISA (Diff, Recon, CMD)"""
    misa = model.misa
    
    # 1. Diff Loss
    diff_loss = 0
    # Between private and shared
    diff_loss += diff_loss_fn(misa.utt_private_t, misa.utt_shared_t)
    diff_loss += diff_loss_fn(misa.utt_private_v, misa.utt_shared_v)
    diff_loss += diff_loss_fn(misa.utt_private_a, misa.utt_shared_a) # 'a' is behavior here
    # Across privates
    diff_loss += diff_loss_fn(misa.utt_private_a, misa.utt_private_t)
    diff_loss += diff_loss_fn(misa.utt_private_a, misa.utt_private_v)
    diff_loss += diff_loss_fn(misa.utt_private_t, misa.utt_private_v)
    
    # 2. Recon Loss
    recon_loss = 0
    recon_loss += recon_loss_fn(misa.utt_t_recon, misa.utt_t_orig)
    recon_loss += recon_loss_fn(misa.utt_v_recon, misa.utt_v_orig)
    recon_loss += recon_loss_fn(misa.utt_a_recon, misa.utt_a_orig)
    recon_loss = recon_loss / 3.0
    
    # 3. CMD Loss (Similarity)
    if config.use_cmd_sim:
        cmd_loss = 0
        cmd_loss += cmd_loss_fn(misa.utt_shared_t, misa.utt_shared_v, 5)
        cmd_loss += cmd_loss_fn(misa.utt_shared_t, misa.utt_shared_a, 5)
        cmd_loss += cmd_loss_fn(misa.utt_shared_a, misa.utt_shared_v, 5)
        cmd_loss = cmd_loss / 3.0
    else:
        cmd_loss = 0.0
        
    return diff_loss, recon_loss, cmd_loss

def train(args):
    # 1. Config
    # Manually set config values since we are not using the full original config system
    class Config:
        def __init__(self):
            self.embedding_size = 768 # Text (RoBERTa)
            self.visual_size = 768    # Image (ViT)
            self.acoustic_size = 4    # Behavior
            self.hidden_size = 128    # MISA hidden size
            self.dropout = 0.5
            self.num_classes = 2      # Binary classification (Risk vs Non-Risk)
            self.activation = nn.ReLU
            self.rnncell = 'lstm'
            self.use_bert = False     # We use pre-extracted features
            self.use_cmd_sim = True
            self.reverse_grad_weight = 1.0
            
            # Loss weights
            self.diff_weight = 0.3
            self.sim_weight = 1.0
            self.recon_weight = 1.0
            
    config = Config()
    
    # 2. Data
    print("Loading data...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'enhanced_reddit_data', 'processed_features.pkl')
    labels_path = os.path.join(project_root, 'enhanced_reddit_data', 'user_labels.json')
    
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Please run extract_features.py first.")
        return

    # Load labels
    label_map = None
    if os.path.exists(labels_path):
        print(f"Loading labels from {labels_path}...")
        with open(labels_path, 'r') as f:
            label_map = json.load(f)
    else:
        print("Warning: Labels file not found. Using dummy labels.")

    dataset = TemporalDataset(features_path, label_map=label_map)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model
    print("Initializing model...")
    model = TemporalMISA(config).to(DEVICE)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Calculate class weights if labels exist
    if label_map:
        # Get all labels from the dataset
        all_labels = [label_map[d['username']] for d in dataset.data]
        class_counts = np.bincount(all_labels)
        print(f"Class distribution: {class_counts}")
        
        # Simple inverse frequency weights
        total_samples = sum(class_counts)
        weights = [total_samples / c for c in class_counts]
        # Normalize
        weights = torch.FloatTensor(weights).to(DEVICE)
        weights = weights / weights.sum()
        print(f"Using class weights: {weights}")
        
        task_criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        task_criterion = nn.CrossEntropyLoss()
    
    # Auxiliary losses
    diff_loss_fn = DiffLoss()
    recon_loss_fn = MSE()
    cmd_loss_fn = CMD()
    
    # 5. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (text, image, behavior, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            text, image, behavior, labels = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(text, image, behavior)
            future_risk_pred = outputs['future_risk']
            
            # Calculate Losses
            # 1. Task Loss (Future Risk Prediction)
            task_loss = task_criterion(future_risk_pred, labels)
            
            # 2. MISA Auxiliary Losses
            diff_loss, recon_loss, cmd_loss = get_misa_losses(model, config, diff_loss_fn, recon_loss_fn, cmd_loss_fn)
            
            # Total Loss
            # INCREASED TASK WEIGHT: Multiply task_loss by 10.0 to force model to learn classification
            loss = 10.0 * task_loss + \
                   config.diff_weight * diff_loss + \
                   config.sim_weight * cmd_loss + \
                   config.recon_weight * recon_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(future_risk_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        # Print detailed loss info for the last batch to debug
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1} Debug | Task Loss: {task_loss.item():.4f} | Recon: {recon_loss.item():.4f} | Diff: {diff_loss.item():.4f}")
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
        
        # Validation
        validate(model, val_loader, task_criterion)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text, image, behavior, labels in val_loader:
            text, image, behavior, labels = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(text, image, behavior)
            future_risk_pred = outputs['future_risk']
            
            loss = criterion(future_risk_pred, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(future_risk_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = total_loss / len(val_loader)
    acc = 100 * correct / total
    print(f"Validation | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
