# -*- coding: utf-8 -*-
"""
run_cv_experiments.py
=====================
HAD-M3H 五折分层交叉验证实验脚本（统一实验协议版）

所有模型共享：
  - 同一组 5 fold 划分（StratifiedKFold, seed=42）
  - 同一训练配置（epochs / patience / lr / wd / batch_size）
  - 同一随机种子初始化（每 fold 固定 seed）

输出：
  - 每 fold 详细结果（cv_results.json）
  - 终端打印 mean ± std 表格

运行方式（Linux 服务器）：
  cd ~/zzq/HAD_M3H_project
  CUDA_VISIBLE_DEVICES=0 python MISA/src/run_cv_experiments.py 2>&1 | tee cv_result.log
"""

import os, sys, json, pickle, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MISA, TemporalMISA
from utils import DiffLoss, MSE, CMD

# ════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ════════════════════════════════════════════════════════════════════
N_FOLDS    = 5
SEED       = 42
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

# ── 基线模型超参 ──────────────────────────────────────────────────
MAX_EPOCHS = 60      # 统一最大 epoch
PATIENCE   = 12      # 统一早停耐心值
LR         = 5e-4
WD         = 1e-2

# ── MISA 专用超参（复杂模型在小数据下需要更宽松的正则）────────────
MISA_MAX_EPOCHS = 100   # 给复杂模型足够训练轮数
MISA_PATIENCE   = 20    # 早停耐心值放宽
MISA_WD         = 1e-4  # L2 正则降低 100x（原 1e-2 对小数据 MISA 过强）
MISA_LABEL_SMOOTH = 0.1 # 标签平滑降低（原 0.25 使 loss 信噪比过低）

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_PATH = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'processed_features.pkl')
LABELS_PATH   = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'user_labels.json')
CKPT_PATH     = os.path.join(PROJECT_ROOT, 'MISA', 'cv_results.json')

print(f"[INFO] Device: {DEVICE}", flush=True)
print(f"[INFO] Project root: {PROJECT_ROOT}", flush=True)
print(f"[INFO] N_FOLDS={N_FOLDS}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}", flush=True)
print(f"[INFO] MISA: MAX_EPOCHS={MISA_MAX_EPOCHS}  PATIENCE={MISA_PATIENCE}  WD={MISA_WD}  LABEL_SMOOTH={MISA_LABEL_SMOOTH}", flush=True)


# ════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════
# MODEL CONFIG
# ════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self, acoustic_size=16, **kwargs):
        self.embedding_size      = 768
        self.visual_size         = 768
        self.acoustic_size       = acoustic_size
        self.hidden_size         = 64
        self.dropout             = 0.3   # 8.2: 0.65 → 0.3（小数据下 0.65 过强）
        self.num_classes         = 2
        self.activation          = nn.ReLU
        self.rnncell             = 'lstm'
        self.use_bert            = False
        self.use_cmd_sim         = False
        self.reverse_grad_weight = 1.0
        self.diff_weight         = 0.005
        self.recon_weight        = 0.0
        self.sim_weight          = 0.0
        self.no_confidence_fusion = False
        self.no_mc_uncertainty    = False
        self.no_image_masking     = False
        self.no_temporal_lstm     = False
        for k, v in kwargs.items():
            setattr(self, k, v)


# ════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════
class TemporalDataset(Dataset):
    def __init__(self, features_path, label_map, use_4dim_behavior=False):
        print(f"[Dataset] Opening pkl: {features_path}", flush=True)
        with open(features_path, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"[Dataset] pkl loaded ({len(raw_data)} raw users). Filtering...", flush=True)
        self.data      = [ud for ud in raw_data if ud['username'] in label_map]
        self.label_map = label_map
        self.use_4dim  = use_4dim_behavior
        print(f"[Dataset] {len(self.data)} users after filter. Computing behavior stats...", flush=True)

        # Compute behavior normalization stats from ALL data (not per-fold, avoid leakage)
        all_b = np.array([
            np.log1p(np.abs(np.array(wk['behavior_feat'], dtype=np.float32)))
            for ud in self.data for wk in ud['timeline_features']
        ])
        self.behavior_mean = torch.FloatTensor(all_b.mean(axis=0))
        self.behavior_std  = torch.FloatTensor(all_b.std(axis=0) + 1e-8)
        print(f"[Dataset] behavior_mean shape={self.behavior_mean.shape}, done.", flush=True)

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        """Return label array for all samples (needed by StratifiedKFold)."""
        return np.array([self.label_map[ud['username']] for ud in self.data])

    def __getitem__(self, idx):
        ud       = self.data[idx]
        timeline = ud['timeline_features']
        text_seq  = [torch.tensor(t['text_feat'],  dtype=torch.float32) for t in timeline]
        image_seq = [torch.tensor(t['image_feat'], dtype=torch.float32) for t in timeline]
        behav_seq = []
        for t in timeline:
            if not self.use_4dim and 'behavior_feat_rich' in t:
                b = torch.tensor(t['behavior_feat_rich'], dtype=torch.float32)
            else:
                b_raw = torch.tensor(t['behavior_feat'], dtype=torch.float32)
                b_raw = torch.log1p(torch.abs(b_raw))
                b = (b_raw - self.behavior_mean) / self.behavior_std
            behav_seq.append(b)
        label = self.label_map[ud['username']]
        return (torch.stack(text_seq), torch.stack(image_seq),
                torch.stack(behav_seq), torch.tensor(label, dtype=torch.long))

    def get_static_features(self):
        """返回用户级静态特征矩阵 (N, 768+768+16) 和标签数组，用于 sklearn 模型.
        各模态取时间轴均值（无时序建模），对应 Abdullah et al. 2024 的 mean-pooling 策略."""
        X_text, X_img, X_beh, y = [], [], [], []
        for ud in self.data:
            timeline = ud['timeline_features']
            t_feats = np.array([t['text_feat']  for t in timeline], dtype=np.float32)
            v_feats = np.array([t['image_feat'] for t in timeline], dtype=np.float32)
            b_feats = []
            for t in timeline:
                if not self.use_4dim and 'behavior_feat_rich' in t:
                    b_feats.append(np.array(t['behavior_feat_rich'], dtype=np.float32))
                else:
                    b = np.array(t['behavior_feat'], dtype=np.float32)
                    b = np.log1p(np.abs(b))
                    b_feats.append(b)
            b_feats = np.array(b_feats, dtype=np.float32)
            X_text.append(t_feats.mean(axis=0))
            X_img.append(v_feats.mean(axis=0))
            X_beh.append(b_feats.mean(axis=0))
            y.append(self.label_map[ud['username']])
        X = np.concatenate([X_text, X_img, X_beh], axis=1)  # (N, 768+768+16)
        return X, np.array(y)


def collate_fn(batch):
    text_seqs, img_seqs, beh_seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in text_seqs], dtype=torch.long)
    return (pad_sequence(text_seqs, batch_first=True),
            pad_sequence(img_seqs,  batch_first=True),
            pad_sequence(beh_seqs,  batch_first=True),
            torch.stack(labels),
            lengths)


# ════════════════════════════════════════════════════════════════════
# BASELINE MODELS
# ════════════════════════════════════════════════════════════════════
class StaticMLP(nn.Module):
    """RoBERTa-Mean + MLP baseline (Abdullah et al. 2024 复现).
    输入: 各模态时序均値 (768+768+16 = 1552) 或仅文本均値 (768).
    无任何时序建模, 直接分类."""
    def __init__(self, input_dim=768, hidden=128, dropout=0.3, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


    def __init__(self, input_dim, hidden=64, dropout=0.65, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden, num_classes)

    def forward(self, feat_seq, lengths):
        B, T, D = feat_seq.shape
        proj   = self.proj(feat_seq.reshape(B * T, D)).view(B, T, -1)
        packed = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


class ConcatLSTM(nn.Module):
    def __init__(self, text_dim=768, img_dim=768, beh_dim=16, hidden=64, dropout=0.65, nc=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + img_dim + beh_dim, hidden * 3), nn.ReLU(), nn.LayerNorm(hidden * 3))
        self.lstm = nn.LSTM(hidden * 3, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden, nc)

    def forward(self, text, image, behavior, lengths):
        B, T, _ = text.shape
        cat    = torch.cat([text, image, behavior], dim=-1)
        proj   = self.proj(cat.reshape(B * T, -1)).view(B, T, -1)
        packed = pack_padded_sequence(proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


class WeightedLSTM(nn.Module):
    def __init__(self, text_dim=768, img_dim=768, beh_dim=16, hidden=64, dropout=0.65, nc=2):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.proj_v = nn.Sequential(nn.Linear(img_dim,  hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.proj_a = nn.Sequential(nn.Linear(beh_dim,  hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop   = nn.Dropout(dropout)
        self.out    = nn.Linear(hidden, nc)

    def forward(self, text, image, behavior, lengths):
        B, T, _ = text.shape
        pt = self.proj_t(text.reshape(B * T, -1)).view(B, T, -1)
        pv = self.proj_v(image.reshape(B * T, -1)).view(B, T, -1)
        pa = self.proj_a(behavior.reshape(B * T, -1)).view(B, T, -1)
        fused  = (pt + pv + pa) / 3.0
        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


# ════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, is_misa=True, modality='full'):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        text, image, behavior, labels, lengths = batch
        text, image, behavior = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE)

        if is_misa:
            outputs = model(text, image, behavior, lengths=lengths)
            logits  = outputs['future_risk']
        else:
            if modality == 'text':
                logits = model(text, lengths)
            elif modality == 'image':
                logits = model(image, lengths)
            elif modality == 'behavior':
                logits = model(behavior, lengths)
            else:
                logits = model(text, image, behavior, lengths)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return acc, f1, auc


# ════════════════════════════════════════════════════════════════════
# AUXILIARY LOSSES
# ════════════════════════════════════════════════════════════════════
def get_misa_losses(model, diff_fn, recon_fn):
    misa = model.misa
    diff_loss = sum([diff_fn(misa.utt_private_t, misa.utt_shared_t),
                     diff_fn(misa.utt_private_v, misa.utt_shared_v),
                     diff_fn(misa.utt_private_a, misa.utt_shared_a),
                     diff_fn(misa.utt_private_a, misa.utt_private_t),
                     diff_fn(misa.utt_private_a, misa.utt_private_v),
                     diff_fn(misa.utt_private_t, misa.utt_private_v)])
    recon_loss = (recon_fn(misa.utt_t_recon, misa.utt_t_orig) +
                  recon_fn(misa.utt_v_recon, misa.utt_v_orig) +
                  recon_fn(misa.utt_a_recon, misa.utt_a_orig)) / 3.0
    return diff_loss, recon_loss


def conf_diversity_loss(modal_conf):
    w = modal_conf.reshape(-1, 3)
    return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()


# ════════════════════════════════════════════════════════════════════
# ABDULLAH ET AL. 2024 BASELINE TRAINERS
# ════════════════════════════════════════════════════════════════════
def train_lr_baseline(X_all, y_all, train_idx, val_idx, fold_seed=42):
    """复现 Abdullah et al. 2024: TF-IDF + Logistic Regression.
    我们用 RoBERTa 均値特征替代 TF-IDF（无原始文本可用）, 无时序建模."""
    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    clf = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000,
        random_state=fold_seed, solver='lbfgs'
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    probs = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, preds)
    f1  = f1_score(y_val, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_val, probs)
    except Exception:
        auc = 0.5
    return acc, f1, auc


def train_static_mlp_baseline(X_all, y_all, train_idx, val_idx, fold_seed=42):
    """复现 Abdullah et al. 2024: RoBERTa 均値 + MLP 分类器, 无时序建模."""
    set_seed(fold_seed)
    input_dim = X_all.shape[1]
    X_train_np, X_val_np = X_all[train_idx], X_all[val_idx]
    y_train_np, y_val_np = y_all[train_idx], y_all[val_idx]

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train_np)
    X_val_np   = scaler.transform(X_val_np)

    X_train = torch.FloatTensor(X_train_np).to(DEVICE)
    X_val   = torch.FloatTensor(X_val_np).to(DEVICE)
    y_train = torch.LongTensor(y_train_np).to(DEVICE)
    y_val_t = torch.LongTensor(y_val_np)

    model = StaticMLP(input_dim=input_dim, hidden=256, dropout=0.3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MISA_MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_metrics, no_improve = 0.0, (0., 0., 0.), 0

    for epoch in range(MISA_MAX_EPOCHS):
        model.train()
        # mini-batch
        perm = torch.randperm(len(y_train))
        for i in range(0, len(y_train), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_train[idx])
            loss   = criterion(logits, y_train[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = (probs > 0.5).astype(int)
        acc = accuracy_score(y_val_np, preds)
        f1  = f1_score(y_val_np, preds, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(y_val_np, probs)
        except Exception:
            auc = 0.5

        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"    ep{epoch+1:02d} val_acc={acc:.4f} best={best_acc:.4f} no_imp={no_improve}", flush=True)
        if acc > best_acc:
            best_acc, best_metrics, no_improve = acc, (acc, f1, auc), 0
        else:
            no_improve += 1
            if no_improve >= MISA_PATIENCE:
                break

    return best_metrics


# ════════════════════════════════════════════════════════════════════
# TRAINING LOOPS
# ════════════════════════════════════════════════════════════════════
def train_baseline(model, train_loader, val_loader, modality='text', fold_seed=42):
    set_seed(fold_seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

    best_acc, best_metrics, no_improve = 0.0, (0., 0., 0.), 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            text, image, behavior, labels, lengths = batch
            text, image, behavior = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE)
            labels = labels.to(DEVICE)

            text     = text     + torch.randn_like(text)     * 0.08
            image    = image    + torch.randn_like(image)    * 0.08
            behavior = behavior + torch.randn_like(behavior) * 0.04

            optimizer.zero_grad()
            if modality == 'text':
                logits = model(text, lengths)
            elif modality == 'image':
                logits = model(image, lengths)
            elif modality == 'behavior':
                logits = model(behavior, lengths)
            else:
                logits = model(text, image, behavior, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        acc, f1, auc = evaluate(model, val_loader, is_misa=False, modality=modality)
        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"    ep{epoch+1:02d} val_acc={acc:.4f} best={best_acc:.4f} no_imp={no_improve}", flush=True)
        if acc > best_acc:
            best_acc, best_metrics, no_improve = acc, (acc, f1, auc), 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    return best_metrics


def train_misa_variant(model, config, train_loader, val_loader, fold_seed=42):
    set_seed(fold_seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=MISA_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MISA_MAX_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=MISA_LABEL_SMOOTH)
    diff_fn, recon_fn = DiffLoss(), MSE()

    best_acc, best_metrics, no_improve = 0.0, (0., 0., 0.), 0

    for epoch in range(MISA_MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            text, image, behavior, labels, lengths = batch
            text, image, behavior = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE)
            labels = labels.to(DEVICE)

            text     = text     + torch.randn_like(text)     * 0.08
            image    = image    + torch.randn_like(image)    * 0.08
            behavior = behavior + torch.randn_like(behavior) * 0.04

            optimizer.zero_grad()
            outputs = model(text, image, behavior, lengths=lengths)
            logits  = outputs['future_risk']
            modal_c = outputs['modal_confidences']

            task_loss            = criterion(logits, labels)
            diff_l, recon_l      = get_misa_losses(model, diff_fn, recon_fn)
            cdiv_loss            = conf_diversity_loss(modal_c)

            loss = (task_loss
                    + config.diff_weight  * diff_l
                    + config.recon_weight * recon_l
                    + 0.05 * cdiv_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        acc, f1, auc = evaluate(model, val_loader, is_misa=True)
        if (epoch + 1) % 10 == 0 or no_improve == 0:
            print(f"    ep{epoch+1:02d} val_acc={acc:.4f} best={best_acc:.4f} no_imp={no_improve}", flush=True)
        if acc > best_acc:
            best_acc, best_metrics, no_improve = acc, (acc, f1, auc), 0
        else:
            no_improve += 1
            if no_improve >= MISA_PATIENCE:
                break

    return best_metrics


# ════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ════════════════════════════════════════════════════════════════════
def load_ckpt():
    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_ckpt(data):
    with open(CKPT_PATH, 'w') as f:
        json.dump(data, f, indent=2)


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()
    set_seed(SEED)

    # ── 加载数据 ─────────────────────────────────────────────────
    print("[INFO] Loading label map...", flush=True)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    print(f"[INFO] Label map loaded: {len(label_map)} users", flush=True)

    print("[INFO] Loading features (pkl)...", flush=True)
    dataset      = TemporalDataset(FEATURES_PATH, label_map)
    print("[INFO] Dataset (16-dim) loaded", flush=True)
    dataset_4dim = TemporalDataset(FEATURES_PATH, label_map, use_4dim_behavior=True)
    print("[INFO] Dataset (4-dim) loaded", flush=True)

    all_labels = dataset.get_labels()
    n_total    = len(dataset)
    print(f"[INFO] Total users: {n_total}  (risk={all_labels.sum()}, ctrl={n_total-all_labels.sum()})", flush=True)

    # ── 预计算静态特征（LR/MLP 基线用） ────────────────────────────────
    print("[INFO] Pre-computing static features for sklearn baselines...", flush=True)
    X_static, y_static = dataset.get_static_features()   # (N, 768+768+16)
    X_text_only = X_static[:, :768]                       # (N, 768) 仅文本，对应论文第一作者的最佳 LLM
    print(f"[INFO] Static features: X={X_static.shape}", flush=True)

    # ── 生成 5 折划分（所有模型共用） ────────────────────────────
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(np.arange(n_total), all_labels))

    # ── 断点续传 ─────────────────────────────────────────────────
    ckpt = load_ckpt()

    # 每个模型的结果格式: { model_name: [[fold0_acc, fold0_f1, fold0_auc], [fold1…], …] }
    model_names = [
        'RoBERTa-Mean+LR', 'RoBERTa-Mean+MLP',
        'Text-only', 'Image-only', 'Behavior-only',
        'Concat', 'Weighted', 'Standard-MISA', 'HAD-M3H',
        'wo-CF', 'wo-MC', 'wo-IM', 'wo-LSTM', 'wo-BF',
    ]
    if not ckpt:
        ckpt = {name: [] for name in model_names}

    print(f"\n[INFO] Starting {N_FOLDS}-fold CV for {len(model_names)} models...\n", flush=True)

    # ── 逐 fold 训练 ──────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*70}", flush=True)
        print(f"  FOLD {fold_idx+1} / {N_FOLDS}  (train={len(train_idx)}, val={len(val_idx)})", flush=True)
        print(f"{'='*70}", flush=True)

        fold_seed = SEED + fold_idx   # 每 fold 不同初始化种子

        # ── Abdullah et al. 2024 基线 ───────────────────────────────────────────
        if not already_done('RoBERTa-Mean+LR'):
            print(f"\n--- [F{fold_idx+1}] RoBERTa-Mean + LR  (Abdullah 2024 复现) ---", flush=True)
            r = train_lr_baseline(X_text_only, y_static, train_idx, val_idx, fold_seed)
            ckpt['RoBERTa-Mean+LR'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] RoBERTa-Mean+LR fold {fold_idx+1}", flush=True)

        if not already_done('RoBERTa-Mean+MLP'):
            print(f"\n--- [F{fold_idx+1}] RoBERTa-Mean + MLP  (Abdullah 2024 复现) ---", flush=True)
            r = train_static_mlp_baseline(X_text_only, y_static, train_idx, val_idx, fold_seed)
            ckpt['RoBERTa-Mean+MLP'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] RoBERTa-Mean+MLP fold {fold_idx+1}", flush=True)

        # DataLoaders for this fold
        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn)
        train_loader_4d = DataLoader(
            Subset(dataset_4dim, train_idx), batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn)
        val_loader_4d = DataLoader(
            Subset(dataset_4dim, val_idx), batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn)

        def already_done(name):
            return len(ckpt.get(name, [])) > fold_idx

        # ── Baselines ──────────────────────────────────────────────
        if not already_done('Text-only'):
            print(f"\n--- [F{fold_idx+1}] Text-only LSTM ---", flush=True)
            m = SingleModalLSTM(input_dim=768)
            r = train_baseline(m, train_loader, val_loader, modality='text', fold_seed=fold_seed)
            ckpt['Text-only'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Text-only fold {fold_idx+1}", flush=True)

        if not already_done('Image-only'):
            print(f"\n--- [F{fold_idx+1}] Image-only LSTM ---", flush=True)
            m = SingleModalLSTM(input_dim=768)
            r = train_baseline(m, train_loader, val_loader, modality='image', fold_seed=fold_seed)
            ckpt['Image-only'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Image-only fold {fold_idx+1}", flush=True)

        if not already_done('Behavior-only'):
            print(f"\n--- [F{fold_idx+1}] Behavior-only LSTM ---", flush=True)
            m = SingleModalLSTM(input_dim=16)
            r = train_baseline(m, train_loader, val_loader, modality='behavior', fold_seed=fold_seed)
            ckpt['Behavior-only'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Behavior-only fold {fold_idx+1}", flush=True)

        if not already_done('Concat'):
            print(f"\n--- [F{fold_idx+1}] Concat LSTM ---", flush=True)
            m = ConcatLSTM()
            r = train_baseline(m, train_loader, val_loader, modality='concat', fold_seed=fold_seed)
            ckpt['Concat'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Concat fold {fold_idx+1}", flush=True)

        if not already_done('Weighted'):
            print(f"\n--- [F{fold_idx+1}] Weighted LSTM ---", flush=True)
            m = WeightedLSTM()
            r = train_baseline(m, train_loader, val_loader, modality='weighted', fold_seed=fold_seed)
            ckpt['Weighted'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Weighted fold {fold_idx+1}", flush=True)

        if not already_done('Standard-MISA'):
            print(f"\n--- [F{fold_idx+1}] Standard-MISA ---", flush=True)
            cfg = Config(no_confidence_fusion=True, no_temporal_lstm=True)
            m   = TemporalMISA(cfg)
            r   = train_misa_variant(m, cfg, train_loader, val_loader, fold_seed=fold_seed)
            ckpt['Standard-MISA'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] Standard-MISA fold {fold_idx+1}", flush=True)

        # ── HAD-M3H ────────────────────────────────────────────────
        if not already_done('HAD-M3H'):
            print(f"\n--- [F{fold_idx+1}] HAD-M3H (full model) ---", flush=True)
            cfg = Config()
            m   = TemporalMISA(cfg)
            r   = train_misa_variant(m, cfg, train_loader, val_loader, fold_seed=fold_seed)
            ckpt['HAD-M3H'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] HAD-M3H fold {fold_idx+1}", flush=True)

        # ── Ablations ──────────────────────────────────────────────
        ablations = [
            ('wo-CF',   Config(no_confidence_fusion=True)),
            ('wo-MC',   Config(no_mc_uncertainty=True)),
            ('wo-IM',   Config(no_image_masking=True)),
            ('wo-LSTM', Config(no_temporal_lstm=True)),
        ]
        for name, cfg in ablations:
            if not already_done(name):
                print(f"\n--- [F{fold_idx+1}] {name} ---", flush=True)
                m = TemporalMISA(cfg)
                r = train_misa_variant(m, cfg, train_loader, val_loader, fold_seed=fold_seed)
                ckpt[name].append(list(r)); save_ckpt(ckpt)
                print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
            else:
                print(f"  [CACHED] {name} fold {fold_idx+1}", flush=True)

        if not already_done('wo-BF'):
            print(f"\n--- [F{fold_idx+1}] wo-BF (4-dim behavior) ---", flush=True)
            cfg = Config(acoustic_size=4)
            m   = TemporalMISA(cfg)
            r   = train_misa_variant(m, cfg, train_loader_4d, val_loader_4d, fold_seed=fold_seed)
            ckpt['wo-BF'].append(list(r)); save_ckpt(ckpt)
            print(f"  → acc={r[0]:.4f}  f1={r[1]:.4f}  auc={r[2]:.4f}", flush=True)
        else:
            print(f"  [CACHED] wo-BF fold {fold_idx+1}", flush=True)

        elapsed = (time.time() - t_start) / 60
        print(f"\n  [Fold {fold_idx+1} done, total elapsed: {elapsed:.1f} min]", flush=True)

    # ════════════════════════════════════════════════════════════════
    # FINAL RESULTS TABLE
    # ════════════════════════════════════════════════════════════════
    print("\n\n" + "="*90)
    print(f"{'5-Fold CV Results  (mean ± std)':^90}")
    print("="*90)
    print(f"{'Model':<26} {'Accuracy':>18} {'F1':>18} {'AUC':>18}")
    print("-"*90)

    baseline_models = ['RoBERTa-Mean+LR', 'RoBERTa-Mean+MLP',
                        'Text-only', 'Image-only', 'Behavior-only', 'Concat', 'Weighted', 'Standard-MISA', 'HAD-M3H']
    ablation_models = ['HAD-M3H', 'wo-CF', 'wo-MC', 'wo-IM', 'wo-LSTM', 'wo-BF']

    def fmt(vals):
        arr = np.array(vals)
        return f"{arr.mean():.4f} ± {arr.std():.4f}"

    print("\n【Table 4.1  Baseline Comparison】")
    for name in baseline_models:
        folds = ckpt.get(name, [])
        if not folds:
            print(f"  {name:<24}  (no data)")
            continue
        accs = [v[0] for v in folds]
        f1s  = [v[1] for v in folds]
        aucs = [v[2] for v in folds]
        print(f"  {name:<24}  {fmt(accs):>18}  {fmt(f1s):>18}  {fmt(aucs):>18}")

    print("\n【Table 4.2  Ablation Study】")
    base_accs = [v[0] for v in ckpt.get('HAD-M3H', [])]
    base_mean = np.mean(base_accs) if base_accs else 0.0
    for name in ablation_models:
        folds = ckpt.get(name, [])
        if not folds:
            print(f"  {name:<24}  (no data)")
            continue
        accs = [v[0] for v in folds]
        f1s  = [v[1] for v in folds]
        aucs = [v[2] for v in folds]
        drop = f"-{(base_mean - np.mean(accs))*100:.2f}%" if name != 'HAD-M3H' else "—"
        print(f"  {name:<24}  {fmt(accs):>18}  {fmt(f1s):>18}  {fmt(aucs):>18}  {drop:>8}")

    print("\n" + "="*90)
    total_min = (time.time() - t_start) / 60
    print(f"All done. Total time: {total_min:.1f} min")
    print(f"Full per-fold results saved to: {CKPT_PATH}")
    print("="*90)


if __name__ == '__main__':
    main()
