# -*- coding: utf-8 -*-
"""
run_experiments.py
==================
HAD-M3H 完整实验评估脚本（对应项目书第4章）

实验内容：
  Table 4.1  基准对比实验（Text-only / Image-only / Behavior-only / Concat /
              Weighted / Standard-MISA / HAD-M3H）
  Table 4.2  消融实验（5 个变体）
  Table 4.3  k 步预测实验（k = 1, 2, 3）
  Extra      AUC 指标 + 高风险用户模态权重统计

运行方式：
  cd D:\\PythonProject\\HAD_M3H_project
  python MISA/src/run_experiments.py
"""

import os, sys, pickle, json, random, copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm

# ── 路径 ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MISA, TemporalMISA
from utils import DiffLoss, MSE, CMD

SEED   = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Project root: {PROJECT_ROOT}")


# ════════════════════════════════════════════════════════════════════
# 1. SEED
# ════════════════════════════════════════════════════════════════════
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ════════════════════════════════════════════════════════════════════
# 2. CONFIG
# ════════════════════════════════════════════════════════════════════
class Config:
    def __init__(self, acoustic_size=16, **kwargs):
        self.embedding_size = 768
        self.visual_size    = 768
        self.acoustic_size  = acoustic_size
        self.hidden_size    = 64
        self.dropout        = 0.65
        self.num_classes    = 2
        self.activation     = nn.ReLU
        self.rnncell        = 'lstm'
        self.use_bert       = False
        self.use_cmd_sim    = False
        self.reverse_grad_weight = 1.0
        self.diff_weight  = 0.005
        self.recon_weight = 0.0
        self.sim_weight   = 0.0
        # Ablation flags (all False by default = original HAD-M3H behavior)
        self.no_confidence_fusion = False
        self.no_mc_uncertainty    = False
        self.no_image_masking     = False
        self.no_temporal_lstm     = False
        for k, v in kwargs.items():
            setattr(self, k, v)


# ════════════════════════════════════════════════════════════════════
# 3. DATASET
# ════════════════════════════════════════════════════════════════════
class TemporalDataset(Dataset):
    """支持 use_4dim_behavior 开关（消融：退化为4维行为特征）。"""
    def __init__(self, features_path, label_map, behavior_stats=None, use_4dim_behavior=False):
        with open(features_path, 'rb') as f:
            raw_data = pickle.load(f)
        self.data   = [ud for ud in raw_data if ud['username'] in label_map]
        self.label_map = label_map
        self.use_4dim  = use_4dim_behavior

        if behavior_stats is None:
            all_b = []
            for ud in self.data:
                for wk in ud['timeline_features']:
                    b = np.array(wk['behavior_feat'], dtype=np.float32)
                    all_b.append(np.log1p(np.abs(b)))
            all_b = np.array(all_b)
            self.behavior_mean = torch.FloatTensor(all_b.mean(axis=0))
            self.behavior_std  = torch.FloatTensor(all_b.std(axis=0) + 1e-8)
        else:
            self.behavior_mean, self.behavior_std = behavior_stats

    def get_behavior_stats(self):
        return (self.behavior_mean, self.behavior_std)

    def __len__(self):
        return len(self.data)

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


def collate_fn(batch):
    text_seqs, img_seqs, beh_seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in text_seqs], dtype=torch.long)
    return (pad_sequence(text_seqs, batch_first=True),
            pad_sequence(img_seqs,  batch_first=True),
            pad_sequence(beh_seqs,  batch_first=True),
            torch.stack(labels),
            lengths)


def make_loaders(features_path, label_map, batch_size=8, use_4dim=False):
    """返回 (train_loader, val_loader, behavior_stats)。固定 seed=42 划分。"""
    set_seed(SEED)
    dataset = TemporalDataset(features_path, label_map, use_4dim_behavior=use_4dim)
    n       = len(dataset)
    train_n = int(0.8 * n)
    val_n   = n - train_n
    gen     = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_n, val_n], generator=gen)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, dataset.get_behavior_stats()


# ════════════════════════════════════════════════════════════════════
# 4. BASELINE MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════════════
class SingleModalLSTM(nn.Module):
    """Text-only / Image-only / Behavior-only LSTM baseline."""
    def __init__(self, input_dim, hidden=64, dropout=0.65, num_classes=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden, num_classes)

    def forward(self, feat_seq, lengths):
        B, T, D = feat_seq.shape
        proj   = self.proj(feat_seq.view(B * T, D)).view(B, T, -1)
        packed = pack_padded_sequence(proj, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


class ConcatLSTM(nn.Module):
    """Concat all modality features → LSTM baseline."""
    def __init__(self, text_dim=768, img_dim=768, beh_dim=16, hidden=64, dropout=0.65, nc=2):
        super().__init__()
        concat_dim = text_dim + img_dim + beh_dim
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, hidden * 3), nn.ReLU(), nn.LayerNorm(hidden * 3))
        self.lstm = nn.LSTM(hidden * 3, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden, nc)

    def forward(self, text, image, behavior, lengths):
        B, T, _ = text.shape
        cat    = torch.cat([text, image, behavior], dim=-1)
        proj   = self.proj(cat.view(B * T, -1)).view(B, T, -1)
        packed = pack_padded_sequence(proj, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


class WeightedLSTM(nn.Module):
    """Equal-weight projection + sum + LSTM baseline."""
    def __init__(self, text_dim=768, img_dim=768, beh_dim=16, hidden=64, dropout=0.65, nc=2):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.proj_v = nn.Sequential(nn.Linear(img_dim,  hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.proj_a = nn.Sequential(nn.Linear(beh_dim,  hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(hidden, nc)

    def forward(self, text, image, behavior, lengths):
        B, T, _ = text.shape
        pt = self.proj_t(text.view(B * T, -1)).view(B, T, -1)
        pv = self.proj_v(image.view(B * T, -1)).view(B, T, -1)
        pa = self.proj_a(behavior.view(B * T, -1)).view(B, T, -1)
        fused  = (pt + pv + pa) / 3.0
        packed = pack_padded_sequence(fused, lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.out(self.drop(h_n.squeeze(0)))


# ════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, is_misa=True, modality='full', k_step=0):
    """
    返回 (accuracy, f1, auc)。
    k_step > 0 时对序列做截断（k步预测评估）。
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        text, image, behavior, labels, lengths = batch
        text     = text.to(DEVICE)
        image    = image.to(DEVICE)
        behavior = behavior.to(DEVICE)
        labels   = labels.to(DEVICE)

        # k-step truncation
        if k_step > 0:
            new_len = (lengths - k_step).clamp(min=1)
            max_t   = int(new_len.max().item())
            text     = text[:, :max_t]
            image    = image[:, :max_t]
            behavior = behavior[:, :max_t]
            lengths  = new_len

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
            elif modality in ('concat', 'weighted'):
                logits = model(text, image, behavior, lengths)
            else:
                raise ValueError(f"Unknown modality: {modality}")

        probs   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds   = (probs > 0.5).astype(int)
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
# 6. MISA AUXILIARY LOSSES
# ════════════════════════════════════════════════════════════════════
def get_misa_losses(model, config, diff_fn, recon_fn, cmd_fn):
    misa = model.misa
    diff_loss  = sum([diff_fn(misa.utt_private_t, misa.utt_shared_t),
                      diff_fn(misa.utt_private_v, misa.utt_shared_v),
                      diff_fn(misa.utt_private_a, misa.utt_shared_a),
                      diff_fn(misa.utt_private_a, misa.utt_private_t),
                      diff_fn(misa.utt_private_a, misa.utt_private_v),
                      diff_fn(misa.utt_private_t, misa.utt_private_v)])
    recon_loss = (recon_fn(misa.utt_t_recon, misa.utt_t_orig) +
                  recon_fn(misa.utt_v_recon, misa.utt_v_orig) +
                  recon_fn(misa.utt_a_recon, misa.utt_a_orig)) / 3.0
    cmd_loss   = 0.0
    return diff_loss, recon_loss, cmd_loss


def conf_diversity_loss(modal_conf):
    w = modal_conf.view(-1, 3)
    return -(w * torch.log(w + 1e-8)).sum(dim=-1).mean()


# ════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOPS
# ════════════════════════════════════════════════════════════════════
def train_baseline(model, train_loader, val_loader,
                   modality='text', epochs=25, patience=7,
                   lr=5e-4, wd=1e-2):
    """简单基线模型训练循环（无 MISA 辅助损失）。"""
    set_seed(SEED)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25)

    best_acc, best_metrics, no_improve = 0.0, (0., 0., 0.), 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            text, image, behavior, labels, lengths = batch
            text, image, behavior = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE)
            labels = labels.to(DEVICE)

            # 特征噪声增强（与主模型保持一致）
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
            elif modality in ('concat', 'weighted'):
                logits = model(text, image, behavior, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        acc, f1, auc = evaluate(model, val_loader, is_misa=False, modality=modality)
        print(f"  [E{epoch+1:02d}] acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        if acc > best_acc:
            best_acc     = acc
            best_metrics = (acc, f1, auc)
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    return best_metrics


def train_misa_variant(model, config, train_loader, val_loader,
                       epochs=35, patience=9, lr=5e-4, wd=1e-2):
    """TemporalMISA 变体训练循环（含辅助损失）。"""
    set_seed(SEED)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.25)
    diff_fn, recon_fn, cmd_fn = DiffLoss(), MSE(), CMD()

    best_acc, best_metrics, no_improve = 0.0, (0., 0., 0.), 0

    for epoch in range(epochs):
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

            task_loss             = criterion(logits, labels)
            diff_l, recon_l, _    = get_misa_losses(model, config, diff_fn, recon_fn, cmd_fn)
            cdiv_loss             = conf_diversity_loss(modal_c)

            loss = (task_loss
                    + config.diff_weight  * diff_l
                    + config.recon_weight * recon_l
                    + 0.05 * cdiv_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        acc, f1, auc = evaluate(model, val_loader, is_misa=True)
        print(f"  [E{epoch+1:02d}] acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        if acc > best_acc:
            best_acc     = acc
            best_metrics = (acc, f1, auc)
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    return best_metrics


# ════════════════════════════════════════════════════════════════════
# 8. MAIN EXPERIMENT RUNNER
# ════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)

    # ── 数据路径 ─────────────────────────────────────────────────
    features_path = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'processed_features.pkl')
    labels_path   = os.path.join(PROJECT_ROOT, 'temporal_reddit_data', 'user_labels.json')
    best_model_path = os.path.join(PROJECT_ROOT, 'MISA', 'best_model_seed42.pth')

    assert os.path.exists(features_path), f"Features not found: {features_path}"
    assert os.path.exists(labels_path),   f"Labels not found: {labels_path}"

    with open(labels_path) as f:
        label_map = json.load(f)

    # ── 主数据加载器 ─────────────────────────────────────────────
    print("\n[INFO] Loading data loaders...")
    train_loader, val_loader, beh_stats = make_loaders(features_path, label_map, batch_size=8)

    results_42  = {}   # Table 4.1 baseline
    results_abl = {}   # Table 4.2 ablation
    results_kst = {}   # Table 4.3 k-step

    # ── 加载实验进度缓存（避免重复训练）──────────────────────────
    checkpoint_path = os.path.join(PROJECT_ROOT, 'MISA', 'experiment_checkpoint.json')
    checkpoint = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        print(f"[INFO] Loaded {len(checkpoint)} cached results from checkpoint")

    def maybe_cached(key):
        if key in checkpoint:
            m = tuple(checkpoint[key])
            print(f"  [CACHED] {key}: acc={m[0]:.4f}  f1={m[1]:.4f}  auc={m[2]:.4f}")
            return m
        return None

    def save_ckpt(key, val):
        checkpoint[key] = list(val)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    t_start = time.time()

    # ════════════════════════════════════════════════════════════════
    # TABLE 4.1: BASELINE COMPARISON
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TABLE 4.1  BASELINE COMPARISON")
    print("="*60)

    # 1. Text-only
    print("\n--- Text-only LSTM ---")
    if (r := maybe_cached('Text-only')) is not None:
        results_42['Text-only'] = r
    else:
        m = SingleModalLSTM(input_dim=768)
        r = train_baseline(m, train_loader, val_loader, modality='text', epochs=25, patience=7)
        results_42['Text-only'] = r;  save_ckpt('Text-only', r)

    # 2. Image-only
    print("\n--- Image-only LSTM ---")
    if (r := maybe_cached('Image-only')) is not None:
        results_42['Image-only'] = r
    else:
        m = SingleModalLSTM(input_dim=768)
        r = train_baseline(m, train_loader, val_loader, modality='image', epochs=25, patience=7)
        results_42['Image-only'] = r;  save_ckpt('Image-only', r)

    # 3. Behavior-only
    print("\n--- Behavior-only LSTM ---")
    if (r := maybe_cached('Behavior-only')) is not None:
        results_42['Behavior-only'] = r
    else:
        m = SingleModalLSTM(input_dim=16)
        r = train_baseline(m, train_loader, val_loader, modality='behavior', epochs=25, patience=7)
        results_42['Behavior-only'] = r;  save_ckpt('Behavior-only', r)

    # 4. Concat
    print("\n--- Concat (simple concatenation) ---")
    if (r := maybe_cached('Concat')) is not None:
        results_42['Concat'] = r
    else:
        m = ConcatLSTM()
        r = train_baseline(m, train_loader, val_loader, modality='concat', epochs=25, patience=7)
        results_42['Concat'] = r;  save_ckpt('Concat', r)

    # 5. Weighted (equal weights)
    print("\n--- Weighted (equal-weight fusion) ---")
    if (r := maybe_cached('Weighted')) is not None:
        results_42['Weighted'] = r
    else:
        m = WeightedLSTM()
        r = train_baseline(m, train_loader, val_loader, modality='weighted', epochs=25, patience=7)
        results_42['Weighted'] = r;  save_ckpt('Weighted', r)

    # 6. Standard MISA (no confidence fusion = uniform 1/3 weights)
    print("\n--- Standard MISA (no confidence weighting) ---")
    if (r := maybe_cached('MISA')) is not None:
        results_42['MISA'] = r
    else:
        cfg_misa = Config(no_confidence_fusion=True)
        m = TemporalMISA(cfg_misa)
        r = train_misa_variant(m, cfg_misa, train_loader, val_loader, epochs=35, patience=9)
        results_42['MISA'] = r;  save_ckpt('MISA', r)

    # 7. HAD-M3H — load saved best model (already trained, acc=85.00%)
    print("\n--- HAD-M3H (full model, loading saved best_model_seed42.pth) ---")
    cfg_full = Config()
    m_full   = TemporalMISA(cfg_full).to(DEVICE)
    if os.path.exists(best_model_path):
        # strict=False: ignore new no_lstm_predictor keys added for ablation
        missing, unexpected = m_full.load_state_dict(
            torch.load(best_model_path, map_location=DEVICE), strict=False)
        if missing:
            print(f"  [INFO] New keys (ablation layers, OK to miss): {missing}")
        print(f"  Loaded: {best_model_path}")
        r = evaluate(m_full, val_loader, is_misa=True)
        results_42['HAD-M3H'] = r;  save_ckpt('HAD-M3H', r)
    else:
        print(f"  [WARN] best_model not found, retraining HAD-M3H...")
        m_full = TemporalMISA(cfg_full)
        r = train_misa_variant(m_full, cfg_full, train_loader, val_loader, epochs=35, patience=9)
        results_42['HAD-M3H'] = r;  save_ckpt('HAD-M3H', r)

    # ── Print Table 4.1 ─────────────────────────────────────────
    print("\n\n" + "="*70)
    print(f"{'表4.1  基准对比实验结果':^70}")
    print("="*70)
    print(f"{'模型':<22} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
    print("-"*70)
    for name, (acc, f1, auc) in results_42.items():
        print(f"{name:<22} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
    print("="*70)

    elapsed = (time.time() - t_start) / 60
    print(f"\n[INFO] Baseline experiments done in {elapsed:.1f} min")

    # ════════════════════════════════════════════════════════════════
    # TABLE 4.2: ABLATION STUDY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TABLE 4.2  ABLATION STUDY")
    print("="*60)

    # 0. Full HAD-M3H (reference)
    results_abl['HAD-M3H (原模型)'] = results_42['HAD-M3H']

    # 1. wo-CF: remove confidence fusion (equal weights)
    print("\n--- HAD-M3H-wo-CF (no confidence fusion) ---")
    if (r := maybe_cached('wo-CF')) is not None:
        results_abl['wo-CF'] = r
    else:
        cfg = Config(no_confidence_fusion=True)
        m = TemporalMISA(cfg)
        r = train_misa_variant(m, cfg, train_loader, val_loader, epochs=35, patience=9)
        results_abl['wo-CF'] = r;  save_ckpt('wo-CF', r)

    # 2. wo-MC: remove MC-Dropout uncertainty estimation (base score only)
    print("\n--- HAD-M3H-wo-MC (no MC-Dropout uncertainty) ---")
    if (r := maybe_cached('wo-MC')) is not None:
        results_abl['wo-MC'] = r
    else:
        cfg = Config(no_mc_uncertainty=True)
        m = TemporalMISA(cfg)
        r = train_misa_variant(m, cfg, train_loader, val_loader, epochs=35, patience=9)
        results_abl['wo-MC'] = r;  save_ckpt('wo-MC', r)

    # 3. wo-IM: remove image zero-vector masking
    print("\n--- HAD-M3H-wo-IM (no image masking) ---")
    if (r := maybe_cached('wo-IM')) is not None:
        results_abl['wo-IM'] = r
    else:
        cfg = Config(no_image_masking=True)
        m = TemporalMISA(cfg)
        r = train_misa_variant(m, cfg, train_loader, val_loader, epochs=35, patience=9)
        results_abl['wo-IM'] = r;  save_ckpt('wo-IM', r)

    # 4. wo-LSTM: remove temporal LSTM (last-week features only)
    print("\n--- HAD-M3H-wo-LSTM (no temporal LSTM) ---")
    if (r := maybe_cached('wo-LSTM')) is not None:
        results_abl['wo-LSTM'] = r
    else:
        cfg = Config(no_temporal_lstm=True)
        m = TemporalMISA(cfg)
        r = train_misa_variant(m, cfg, train_loader, val_loader, epochs=35, patience=9)
        results_abl['wo-LSTM'] = r;  save_ckpt('wo-LSTM', r)

    # 5. wo-BF: 4-dim behavior features instead of 16-dim
    print("\n--- HAD-M3H-wo-BF (4-dim behavior features) ---")
    if (r := maybe_cached('wo-BF')) is not None:
        results_abl['wo-BF'] = r
    else:
        train_4d, val_4d, _ = make_loaders(features_path, label_map, batch_size=8, use_4dim=True)
        cfg = Config(acoustic_size=4)
        m = TemporalMISA(cfg)
        r = train_misa_variant(m, cfg, train_4d, val_4d, epochs=35, patience=9)
        results_abl['wo-BF'] = r;  save_ckpt('wo-BF', r)

    # ── Print Table 4.2 ─────────────────────────────────────────
    base_acc = results_abl['HAD-M3H (原模型)'][0]
    print("\n\n" + "="*80)
    print(f"{'表4.2  消融实验结果':^80}")
    print("="*80)
    print(f"{'模型':<28} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Acc 下降':>10}")
    print("-"*80)
    for name, (acc, f1, auc) in results_abl.items():
        drop = base_acc - acc
        drop_str = f"-{drop*100:.2f}%" if name != 'HAD-M3H (原模型)' else "—"
        print(f"{name:<28} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f} {drop_str:>10}")
    print("="*80)

    elapsed = (time.time() - t_start) / 60
    print(f"\n[INFO] Ablation experiments done in {elapsed:.1f} min total")

    # ════════════════════════════════════════════════════════════════
    # TABLE 4.3: K-STEP PREDICTION
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("TABLE 4.3  K-STEP PREDICTION")
    print("="*60)
    print("  (使用最优模型 best_model_seed42.pth，截断最后 k 周评估)")

    if os.path.exists(best_model_path):
        cfg_full = Config()
        m_full   = TemporalMISA(cfg_full).to(DEVICE)
        m_full.load_state_dict(torch.load(best_model_path, map_location=DEVICE), strict=False)

        # k=0 (current baseline)
        acc0, f10, auc0 = evaluate(m_full, val_loader, is_misa=True, k_step=0)
        results_kst['k=0 (当前, 对照)'] = (acc0, f10, auc0)

        for k in [1, 2, 3]:
            print(f"\n--- k={k} (提前 {k} 周预测) ---")
            acc, f1, auc = evaluate(m_full, val_loader, is_misa=True, k_step=k)
            results_kst[f'k={k} (提前{k}周)'] = (acc, f1, auc)
            print(f"  acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")
    else:
        print("[WARN] best_model_seed42.pth not found. Skipping k-step evaluation.")

    # ── Print Table 4.3 ─────────────────────────────────────────
    if results_kst:
        print("\n\n" + "="*70)
        print(f"{'表4.3  k步预测实验结果':^70}")
        print("="*70)
        print(f"{'预测步长':^20} {'Accuracy':>10} {'F1':>10} {'AUC':>10}")
        print("-"*70)
        for name, (acc, f1, auc) in results_kst.items():
            print(f"{name:<20} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f}")
        print("="*70)

    # ════════════════════════════════════════════════════════════════
    # EXTRA: Modal weight statistics for high-risk users (Section 4.5.3)
    # ════════════════════════════════════════════════════════════════
    if os.path.exists(best_model_path):
        print("\n" + "="*60)
        print("EXTRA:  高风险用户模态权重统计（4.5.3节）")
        print("="*60)
        cfg_full = Config()
        m_full   = TemporalMISA(cfg_full).to(DEVICE)
        m_full.load_state_dict(torch.load(best_model_path, map_location=DEVICE), strict=False)
        m_full.eval()

        risk_weights   = []   # modal weights for correctly predicted risk users
        ctrl_weights   = []   # modal weights for correctly predicted control users

        with torch.no_grad():
            for text, image, behavior, labels, lengths in val_loader:
                text, image, behavior = text.to(DEVICE), image.to(DEVICE), behavior.to(DEVICE)
                outputs = m_full(text, image, behavior, lengths=lengths)
                logits  = outputs['future_risk']
                conf    = outputs['modal_confidences']  # [B, T, 3]

                preds = (torch.softmax(logits, dim=1)[:, 1] > 0.5).long()
                w_avg = conf.mean(dim=1)  # [B, 3]  average over weeks

                for i in range(labels.size(0)):
                    if preds[i] == labels[i]:
                        w = w_avg[i].cpu().numpy()
                        if labels[i] == 1:
                            risk_weights.append(w)
                        else:
                            ctrl_weights.append(w)

        if risk_weights:
            rw = np.array(risk_weights).mean(axis=0)
            print(f"\n  高风险用户 (n={len(risk_weights)}) 平均模态权重:")
            print(f"    文本(Text)    = {rw[0]:.4f}")
            print(f"    图像(Image)   = {rw[1]:.4f}")
            print(f"    行为(Behavior)= {rw[2]:.4f}")
        if ctrl_weights:
            cw = np.array(ctrl_weights).mean(axis=0)
            print(f"\n  对照用户 (n={len(ctrl_weights)}) 平均模态权重:")
            print(f"    文本(Text)    = {cw[0]:.4f}")
            print(f"    图像(Image)   = {cw[1]:.4f}")
            print(f"    行为(Behavior)= {cw[2]:.4f}")

    # ── 最终汇总 ─────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    print(f"\n\n{'='*60}")
    print(f"实验全部完成，总耗时 {total_min:.1f} 分钟")
    print("="*60)


if __name__ == '__main__':
    main()
