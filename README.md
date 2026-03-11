# -*- coding: utf-8 -*-
import os
import math
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import argparse
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Import our modules
from config import get_config
from models import TemporalMISA
from utils import DiffLoss, MSE, CMD

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TemporalDataset(Dataset):
    def __init__(self, features_path, label_map=None, behavior_stats=None):
        """
        Args:
            features_path: Path to processed_features.pkl
            label_map: Dictionary mapping username to label (int).
                       If None, generates dummy labels for testing.
            behavior_stats: (mean, std) tuple for behavior normalization.
                            If None, computed from this dataset.
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

        # Compute global behavior normalization stats (log1p + Z-score)
        # Behavior features are raw counts (e.g. post counts, scores) with huge range [0, 184923]
        # We must normalize or MISA will be dominated by these values
        if behavior_stats is None:
            all_behavior = []
            for user_data in self.data:
                for week in user_data['timeline_features']:
                    b = np.array(week['behavior_feat'], dtype=np.float32)
                    all_behavior.append(np.log1p(np.abs(b)))  # log1p-transform
            all_behavior = np.array(all_behavior)
            self.behavior_mean = torch.FloatTensor(all_behavior.mean(axis=0))
            self.behavior_std  = torch.FloatTensor(all_behavior.std(axis=0) + 1e-8)
            print(f"Behavior norm stats: mean={self.behavior_mean.tolist()}, "
                  f"std={self.behavior_std.tolist()}")
        else:
            self.behavior_mean, self.behavior_std = behavior_stats

    def get_behavior_stats(self):
        return (self.behavior_mean, self.behavior_std)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_data = self.data[idx]
        username = user_data['username']
        timeline = user_data['timeline_features']
        
        # Extract week-level sequences
        text_seq  = [torch.tensor(t['text_feat'],  dtype=torch.float32) for t in timeline]
        image_seq = [torch.tensor(t['image_feat'], dtype=torch.float32) for t in timeline]

        # 锟斤拷锟斤拷 Rich 16-dim temporal behavior features (from per-post timestamps) 锟斤拷锟斤拷锟斤拷
        # Computed offline in update_behavior_features.py from raw post data.
        # Includes: circadian patterns, late-night ratio, posting rhythm, etc.
        behavior_seq = []
        for t in timeline:
            if 'behavior_feat_rich' in t:
                b = torch.tensor(t['behavior_feat_rich'], dtype=torch.float32)  # [16]
            else:
                # Fallback: old 4-dim stats (log1p + Z-score)
                b_raw = torch.tensor(t['behavior_feat'], dtype=torch.float32)
                b_raw = torch.log1p(torch.abs(b_raw))
                b = (b_raw - self.behavior_mean) / self.behavior_std
            behavior_seq.append(b)

        # Stack into tensors: [seq_len, feat_dim]
        text_seq     = torch.stack(text_seq)
        image_seq    = torch.stack(image_seq)
        behavior_seq = torch.stack(behavior_seq)   # [seq_len, 16]
        
        # Get Label
        if self.label_map and username in self.label_map:
            label = self.label_map[username]
        else:
            label = hash(username) % 2
            
        return text_seq, image_seq, behavior_seq, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Collates variable-length week sequences; behavior is now a fixed 16-dim vector per week."""
    text_seqs, image_seqs, behavior_seqs, labels = zip(*batch)
    
    # Record actual sequence lengths BEFORE padding
    lengths = torch.tensor([len(s) for s in text_seqs], dtype=torch.long)
    
    # Pad sequences: [B, max_seq_len, feat_dim]
    text_padded     = pad_sequence(text_seqs,     batch_first=True)
    image_padded    = pad_sequence(image_seqs,    batch_first=True)
    behavior_padded = pad_sequence(behavior_seqs, batch_first=True)
    
    labels = torch.stack(labels)
    return text_padded, image_padded, behavior_padded, labels, lengths

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
    recon_loss += recon_loss_fn(misa.utt_v_recon, misa.utt_v_orig)  # Re-enabled: images valid
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def confidence_diversity_loss(modal_confidences):
    """
    Encourages the model to make decisive modal choices rather than uniform 1/3 weights.
    Minimizing entropy -> weights become more peaked (concentrate on informative modalities).

    Args:
        modal_confidences: [batch_size, seq_len, 3]  (softmaxed modal weights)
    Returns:
        scalar loss
    """
    weights = modal_confidences.view(-1, 3)   # [N, 3]
    eps = 1e-8
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)   # [N]
    return entropy.mean()

def train(args):
    set_seed(args.seed)
    print(f"[Seed={args.seed}] Starting run...")
    
    # 1. Config
    # Manually set config values since we are not using the full original config system
    class Config:
        def __init__(self):
            self.embedding_size = 768 # Text (RoBERTa)
            self.visual_size = 768    # Image (ViT)
            self.acoustic_size = 16   # rich 16-dim temporal behavior (was 4 simple stats)
            self.hidden_size = 64     # Reduced: 128鈫�64 to combat overfitting on 480 samples
            self.dropout = 0.65       # Restored: best from Round 1
            self.num_classes = 2      # Binary classification (Risk vs Non-Risk)
            self.activation = nn.ReLU
            self.rnncell = 'lstm'
            self.use_bert = False     # We use pre-extracted features
            # CMD off; keep diff_weight for orthogonality
            self.use_cmd_sim = False
            self.reverse_grad_weight = 1.0
            
            # Auxiliary loss weights
            self.diff_weight  = 0.005  # slight orthogonality pressure
            self.recon_weight = 0.0    # disabled
            self.sim_weight   = 0.0    # CMD disabled
            
    config = Config()
    
    # 2. Data
    print("Loading data...")
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(project_root, 'temporal_reddit_data', 'processed_features.pkl')
    labels_path = os.path.join(project_root, 'temporal_reddit_data', 'user_labels.json')
    
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
    # Use generator for reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model
    print("Initializing model...")
    model = TemporalMISA(config).to(DEVICE)
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)  # Restored: Adam (Round1 best)
    # CosineAnnealing: smooth LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
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
        
        # label smoothing: 0.15鈫�0.25 for stronger confidence suppression
        task_criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
        print("Using CrossEntropyLoss with label_smoothing=0.25")
    else:
        task_criterion = nn.CrossEntropyLoss()
    
    # Auxiliary losses
    diff_loss_fn = DiffLoss()
    recon_loss_fn = MSE()
    cmd_loss_fn = CMD()
    
    # 5. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    best_val_acc = 0.0
    early_stop_counter = 0
    EARLY_STOP_PATIENCE = 20  # Increased: 15鈫�20 for more exploration
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.6f}")
        
        for batch_idx, (text, image, behavior, labels, lengths) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            text, image, behavior, labels = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE), labels.to(DEVICE)
            # lengths stays on CPU (pack_padded_sequence requires CPU lengths)

            # Feature Noise Augmentation (training only) 鈥� strengthened to fight memorization
            text     = text     + torch.randn_like(text)     * 0.08
            image    = image    + torch.randn_like(image)    * 0.08
            behavior = behavior + torch.randn_like(behavior) * 0.04

            optimizer.zero_grad()
            
            # Forward
            outputs = model(text, image, behavior, lengths=lengths)
            future_risk_pred = outputs['future_risk']
            modal_conf = outputs['modal_confidences']   # [B, seq_len, 3]

            # Calculate Losses
            # 1. Task Loss
            task_loss = task_criterion(future_risk_pred, labels)

            # 2. Confidence Diversity Loss (encourage decisive modal weighting)
            conf_loss = confidence_diversity_loss(modal_conf)

            # 3. MISA Auxiliary Losses
            diff_loss, recon_loss, cmd_loss = get_misa_losses(model, config, diff_loss_fn, recon_loss_fn, cmd_loss_fn)

            # Total Loss
            # diff_loss:  private 锟斤拷 shared orthogonality  (~5% of task)
            # recon_loss: information reconstruction fidelity (~3% of task)
            # conf_loss:  modality confidence diversity    (fixed 5%)
            loss = (task_loss
                    + config.diff_weight  * diff_loss
                    + config.recon_weight * recon_loss
                    + config.sim_weight   * cmd_loss
                    + 0.05 * conf_loss)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(future_risk_pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        cmd_val = cmd_loss if isinstance(cmd_loss, float) else cmd_loss.item()
        diff_contrib  = config.diff_weight  * diff_loss.item()
        recon_contrib = config.recon_weight * recon_loss.item()
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}% | "
              f"Task: {task_loss.item():.4f} | "
              f"Diff: {diff_loss.item():.4f}(w={diff_contrib:.4f}) | "
              f"Recon: {recon_loss.item():.4f}(w={recon_contrib:.4f}) | "
              f"CMD: {cmd_val:.4f}")

        # 锟斤拷锟斤拷 Print avg modal confidence weights (text / image / behavior) 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
        model.eval()
        conf_accum = []
        with torch.no_grad():
            for text_b, image_b, behavior_b, _, lengths_b in val_loader:
                text_b     = text_b.to(DEVICE)
                image_b    = image_b.to(DEVICE)
                behavior_b = behavior_b.to(DEVICE)
                out_b = model(text_b, image_b, behavior_b, lengths=lengths_b)
                mc = out_b['modal_confidences'].mean(dim=1)   # [B, 3]
                conf_accum.append(mc.cpu())
        conf_all = torch.cat(conf_accum, dim=0).mean(dim=0)   # [3]
        print(f"  Modal Weights (val avg) | text={conf_all[0]:.4f}  image={conf_all[1]:.4f}  behavior={conf_all[2]:.4f}")
        model.train()
        # 锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷

        # Validation
        val_loss, val_acc, val_f1 = validate(model, val_loader, task_criterion)
        
        # Scheduler step (CosineAnnealing uses epoch count, not val metric)
        scheduler.step()  # CosineAnnealing by epoch
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            save_path = os.path.join(project_root, 'MISA', f'best_model_seed{args.seed}.pth')
            torch.save(model.state_dict(), save_path)
            # Also save as best_model.pth if it's the overall best (handled externally)
            print(f"New best model saved with Val Acc: {val_acc:.2f}%")
        else:
            early_stop_counter += 1
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2f}%")
    return best_val_acc

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = [] # Store probabilities
    
    # Threshold for classification (Lowering it helps with imbalanced data)
    THRESHOLD = 0.5 
    
    with torch.no_grad():
        for text, image, behavior, labels, lengths in val_loader:
            text, image, labels = text.to(DEVICE), image.to(DEVICE), labels.to(DEVICE)
            behavior = behavior.to(DEVICE)
            
            outputs = model(text, image, behavior, lengths=lengths)
            future_risk_pred = outputs['future_risk']
            
            loss = criterion(future_risk_pred, labels)
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.softmax(future_risk_pred, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Use custom threshold instead of argmax
            # predicted = torch.max(future_risk_pred.data, 1)[1]
            predicted = (probs[:, 1] > THRESHOLD).long()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(val_loader)
    acc = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Validation | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}% | F1: {f1:.4f} | Threshold: {THRESHOLD}")
    
    # Debug: Print unique predictions and PROBABILITIES
    unique_preds = np.unique(all_preds)
    print(f"Unique Predictions: {unique_preds}")
    
    # Convert to numpy array for easier slicing
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Print average probability assigned to the correct class
    print("--- Probability Analysis ---")
    print(f"Avg Prob for Class 0: {np.mean(all_probs[:, 0]):.4f}")
    print(f"Avg Prob for Class 1: {np.mean(all_probs[:, 1]):.4f}")
    
    # Show a few examples of Class 1 samples (Risk Users)
    risk_indices = np.where(all_labels == 1)[0]
    if len(risk_indices) > 0:
        print(f"Sample Probabilities for True Risk Users (Class 1):")
        # Show first 5 risk users
        for idx in risk_indices[:5]:
            print(f"  True: 1 | Pred: {all_preds[idx]} | Probs: [0: {all_probs[idx][0]:.4f}, 1: {all_probs[idx][1]:.4f}]")
    
    print("-" * 50)
    
    return avg_loss, acc, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)  # Back to 8: more gradient updates, more stochastic regularization
    parser.add_argument('--lr', type=float, default=5e-4)  # Restored: best LR from Round 1
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi_seed', action='store_true', help='Run with seeds 42, 123, 777 and keep best')
    args = parser.parse_args()
    
    if args.multi_seed:
        seeds = [42, 123, 777]
        overall_best = 0.0
        for s in seeds:
            args.seed = s
            print(f"\n{'='*60}\nRunning with seed={s}\n{'='*60}")
            best_acc = train(args)
            if best_acc > overall_best:
                overall_best = best_acc
        print(f"\nBest across all seeds: {overall_best:.2f}%")
    else:
        train(args)
