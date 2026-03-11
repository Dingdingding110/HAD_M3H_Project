# -*- coding: utf-8 -*-
"""
Diagnostic script to determine why TemporalMISA fails to learn.
Tests:
1. Feature statistics (NaN, Inf, norms)
2. LogisticRegression on mean-pooled features -> measures separability
3. Simple MLP baseline -> confirms PyTorch can learn on this data
4. MISA gradient flow check
"""
import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'processed_features.pkl')
LABELS_PATH = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'user_labels.json')

def load_data():
    with open(FEATURES_PATH, 'rb') as f:
        raw_data = pickle.load(f)
    with open(LABELS_PATH, 'r') as f:
        label_map = json.load(f)
    return raw_data, label_map

def mean_pool_user(user_data):
    """Mean-pool all weekly features into a single vector per user."""
    timeline = user_data['timeline_features']
    text_feats = np.array([t['text_feat'] for t in timeline])
    image_feats = np.array([t['image_feat'] for t in timeline])
    behavior_feats = np.array([t['behavior_feat'] for t in timeline])
    
    return np.concatenate([
        text_feats.mean(axis=0),
        image_feats.mean(axis=0),
        behavior_feats.mean(axis=0)
    ])

def check_feature_stats(raw_data, label_map):
    print("\n" + "="*60)
    print("1. FEATURE STATISTICS CHECK")
    print("="*60)
    
    all_text, all_image, all_behavior = [], [], []
    nan_users, inf_users, zero_users = 0, 0, 0
    
    for user_data in raw_data:
        if user_data['username'] not in label_map:
            continue
        for week in user_data['timeline_features']:
            t = np.array(week['text_feat'])
            v = np.array(week['image_feat'])
            a = np.array(week['behavior_feat'])
            
            all_text.append(t)
            all_image.append(v)
            all_behavior.append(a)
            
            if np.any(np.isnan(t)) or np.any(np.isnan(v)) or np.any(np.isnan(a)):
                nan_users += 1
            if np.any(np.isinf(t)) or np.any(np.isinf(v)) or np.any(np.isinf(a)):
                inf_users += 1
            if np.linalg.norm(t) < 1e-6 or np.linalg.norm(v) < 1e-6:
                zero_users += 1
    
    all_text = np.array(all_text)
    all_image = np.array(all_image)
    all_behavior = np.array(all_behavior)
    
    print(f"Total weeks: {len(all_text)}")
    print(f"NaN weeks: {nan_users}, Inf weeks: {inf_users}, Near-zero weeks: {zero_users}")
    print(f"\nText features:    mean norm={np.linalg.norm(all_text, axis=1).mean():.4f}, "
          f"std={all_text.std():.4f}, range=[{all_text.min():.4f}, {all_text.max():.4f}]")
    print(f"Image features:   mean norm={np.linalg.norm(all_image, axis=1).mean():.4f}, "
          f"std={all_image.std():.4f}, range=[{all_image.min():.4f}, {all_image.max():.4f}]")
    print(f"Behavior features: mean norm={np.linalg.norm(all_behavior, axis=1).mean():.4f}, "
          f"std={all_behavior.std():.4f}, range=[{all_behavior.min():.4f}, {all_behavior.max():.4f}]")
    
    return nan_users == 0 and inf_users == 0

def test_sklearn_baseline(raw_data, label_map):
    print("\n" + "="*60)
    print("2. SKLEARN LOGISTIC REGRESSION BASELINE (mean-pooled features)")
    print("="*60)
    
    X, y = [], []
    for user_data in raw_data:
        uname = user_data['username']
        if uname not in label_map:
            continue
        X.append(mean_pool_user(user_data))
        y.append(label_map[uname])
    
    X = np.array(X)
    y = np.array(y)
    print(f"Dataset: {X.shape[0]} users, {X.shape[1]} features, class balance={y.mean():.3f}")
    
    # Check for NaN/Inf in mean-pooled features
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("ERROR: NaN/Inf found in mean-pooled X!")
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different feature subsets
    text_dim = 768
    image_dim = 768
    behavior_dim = X.shape[1] - text_dim - image_dim
    
    for name, idx in [
        ("Text only", slice(0, text_dim)),
        ("Image only", slice(text_dim, text_dim + image_dim)),
        ("Behavior only", slice(text_dim + image_dim, None)),
        ("Text+Behavior", [*range(0, text_dim), *range(text_dim+image_dim, X.shape[1])]),
        ("All features", slice(None)),
    ]:
        X_sub = X_scaled[:, idx] if isinstance(idx, slice) else X_scaled[:, idx]
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0),
                X_sub, y, cv=5, scoring='accuracy'
            )
            print(f"  {name:20s}: CV Acc = {scores.mean():.4f} +/- {scores.std():.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

def test_pytorch_mlp(raw_data, label_map):
    print("\n" + "="*60)
    print("3. PYTORCH SIMPLE MLP BASELINE (mean-pooled, no MISA)")
    print("="*60)
    
    X, y = [], []
    for user_data in raw_data:
        uname = user_data['username']
        if uname not in label_map:
            continue
        X.append(mean_pool_user(user_data))
        y.append(label_map[uname])
    
    X = torch.FloatTensor(np.array(X))
    y = torch.LongTensor(np.array(y))
    
    # Normalize
    mean = X.mean(0)
    std = X.std(0) + 1e-8
    X = (X - mean) / std
    
    # Train/val split
    n = len(X)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[:int(0.8*n)], idx[int(0.8*n):]
    
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Simple MLP: input -> 256 -> 64 -> 2
    model = nn.Sequential(
        nn.Linear(X.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    for epoch in range(30):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for xb, yb in val_loader:
                preds = model(xb).argmax(1)
                all_preds.extend(preds.numpy())
                all_labels.extend(yb.numpy())
            val_acc = accuracy_score(all_labels, all_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}: Val Acc = {val_acc:.4f} (best={best_val_acc:.4f})")
    
    print(f"\n  Final Best Val Acc (MLP): {best_val_acc:.4f}")

def test_misa_gradient_flow(raw_data, label_map):
    print("\n" + "="*60)
    print("4. MISA GRADIENT FLOW CHECK")
    print("="*60)
    
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'MISA', 'src'))
    from models import TemporalMISA
    
    class Config:
        embedding_size = 768
        visual_size = 768
        acoustic_size = 4
        hidden_size = 128
        dropout = 0.5
        num_classes = 2
        activation = nn.ReLU
        rnncell = 'lstm'
        use_bert = False
        use_cmd_sim = False
        reverse_grad_weight = 1.0
        diff_weight = 0.0
        sim_weight = 0.0
        recon_weight = 0.0
    
    config = Config()
    model = TemporalMISA(config)
    
    # Create a small fake batch
    batch_size = 4
    seq_len = 5
    text = torch.randn(batch_size, seq_len, 768)
    image = torch.randn(batch_size, seq_len, 768)
    behavior = torch.randn(batch_size, seq_len, 4)
    labels = torch.randint(0, 2, (batch_size,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Forward + backward
    optimizer.zero_grad()
    outputs = model(text, image, behavior)
    loss = criterion(outputs['future_risk'], labels)
    loss.backward()
    
    print(f"  Forward pass OK, loss={loss.item():.4f}")
    
    # Check gradient norms per module
    print("\n  Gradient norms per module:")
    has_zero_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < 1e-10:
                has_zero_grad = True
                print(f"    {name}: NEAR-ZERO GRAD ({grad_norm:.2e})")
            elif grad_norm > 10:
                print(f"    {name}: LARGE GRAD ({grad_norm:.4f}) *** WARNING")
        else:
            print(f"    {name}: NO GRAD (disconnected?)")
    
    if not has_zero_grad:
        print("  All parameters have non-zero gradients - gradient flow OK")
    
    # Check output variance
    with torch.no_grad():
        # Test with 20 different random inputs
        outputs_list = []
        for i in range(20):
            t = torch.randn(1, seq_len, 768)
            v = torch.randn(1, seq_len, 768)
            a = torch.randn(1, seq_len, 4)
            out = model(t, v, a)['future_risk']
            outputs_list.append(out.softmax(1)[0, 1].item())
        
        print(f"\n  Output variance check (20 random inputs):")
        print(f"  Prob(class=1): min={min(outputs_list):.4f}, max={max(outputs_list):.4f}, "
              f"std={np.std(outputs_list):.4f}")
        if np.std(outputs_list) < 0.01:
            print("  WARNING: Model outputs nearly identical for all inputs! -> Mode collapse in init.")
        else:
            print("  Model outputs are varied - initialization is OK.")

if __name__ == "__main__":
    print("Loading data...")
    raw_data, label_map = load_data()
    print(f"Loaded {len(raw_data)} users, {len(label_map)} with labels")
    
    ok = check_feature_stats(raw_data, label_map)
    test_sklearn_baseline(raw_data, label_map)
    test_pytorch_mlp(raw_data, label_map)
    test_misa_gradient_flow(raw_data, label_map)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
