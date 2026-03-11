import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


import torch.nn.functional as F

class ModalityConfidenceScorer(nn.Module):
    """
    Hierarchical Confidence Scorer per modality.
    Combines a learned MLP score with MC-Dropout uncertainty estimation.
    High variance across MC samples = high uncertainty = low confidence.
    """
    def __init__(self, input_dim, hidden_dim=64, dropout_p=0.3, n_mc_samples=8):
        super(ModalityConfidenceScorer, self).__init__()
        self.n_mc_samples = n_mc_samples
        self.dropout_p = dropout_p
        # Learned scorer: produces a base confidence in [0, 1]
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, use_mc=True):
        """
        Args:
            x: [N, input_dim]
            use_mc: if False, skip MC-Dropout; return base_score only (ablation: no_mc_uncertainty)
        Returns:
            confidence: [N]  scalar confidence per sample
        """
        # 1. Base learned score
        base_score = self.scorer(x).squeeze(-1)   # [N]

        if not use_mc:
            return base_score

        # 2. MC-Dropout uncertainty (dropout always active regardless of train/eval)
        mc_scores = []
        for _ in range(self.n_mc_samples):
            x_drop = F.dropout(x, p=self.dropout_p, training=True)
            mc_scores.append(self.scorer(x_drop).squeeze(-1))
        mc_scores = torch.stack(mc_scores, dim=0)          # [n_mc, N]
        uncertainty = mc_scores.var(dim=0)                 # [N]

        # Normalize uncertainty to [0, 1] and down-weight high-uncertainty inputs
        uncertainty_norm = uncertainty / (uncertainty.detach().max() + 1e-8)
        confidence = base_score * (1.0 - uncertainty_norm.detach())

        return confidence   # [N]


# let's define a simple model that can deal with multimodal variable length sequence
class BehaviorSequenceEncoder(nn.Module):
    """
    Encodes the within-week post timestamp sequence via a small LSTM.
    Models: late-night posting rhythm, inter-post intervals, circadian patterns.

    Input  : per-post 7-dim feature sequence  [N, max_posts, 7]
    Output : 32-dim behavior embedding         [N, 32]

    Feature layout (POST_FEAT_DIM = 7):
      0-1: hour_sin / hour_cos  (circadian cycle,  range [-1,1])
      2-3: day_sin  / day_cos   (weekly cycle,     range [-1,1])
      4  : is_late_night        (22h-5h flag,      0/1)
      5  : log1p_score          (engagement level, unbounded -> normalised)
      6  : log1p_interval_h     (posting rhythm,   unbounded -> normalised)

    Notes:
      - LayerNorm before LSTM aligns feature scales (score/interval can be >>1).
      - Mean-pool over LSTM outputs (vs. last hidden) is robust when sequences
        are length-1 (single post per week, 68% of weeks in this dataset).
    """
    POST_FEAT_DIM = 7
    HIDDEN_DIM    = 32

    def __init__(self):
        super(BehaviorSequenceEncoder, self).__init__()
        self.input_norm = nn.LayerNorm(self.POST_FEAT_DIM)
        self.lstm = nn.LSTM(
            input_size=self.POST_FEAT_DIM,
            hidden_size=self.HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, post_seq, post_lengths):
        """
        Args:
            post_seq    : [N, max_posts, POST_FEAT_DIM]  (padded)
            post_lengths: [N]  actual number of posts per sample (>= 1)
        Returns:
            [N, HIDDEN_DIM]
        """
        lengths_clamped = post_lengths.clamp(min=1)

        # Normalise raw features before LSTM (handles score/interval scale mismatch)
        post_seq = self.input_norm(post_seq)

        packed = pack_padded_sequence(
            post_seq, lengths_clamped.cpu(),
            batch_first=True, enforce_sorted=False
        )
        out_packed, (h_n, _) = self.lstm(packed)

        # Mean-pool over valid timesteps (more stable than last-hidden for len-1 seqs)
        out_padded, _ = pad_packed_sequence(out_packed, batch_first=True)   # [N, T, H]
        N, T, H = out_padded.shape
        mask   = torch.arange(T, device=post_seq.device).unsqueeze(0) < lengths_clamped.unsqueeze(1)
        mask_f = mask.unsqueeze(2).float()                                   # [N, T, 1]
        mean_out = (out_padded * mask_f).sum(dim=1) / mask_f.sum(dim=1)     # [N, H]
        return mean_out


class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        
        # Removed RNNs and BERT as we are using pre-extracted features

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=self.text_size, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=self.visual_size, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        # LayerNorm on raw 16-dim input to handle scale mismatch
        # (trig values in [-1,1] vs log-scores in [0,4+])
        self.project_a.add_module('project_a_input_norm', nn.LayerNorm(self.acoustic_size))
        self.project_a.add_module('project_a', nn.Linear(in_features=self.acoustic_size, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


        ##########################################
        # private encoders  (Dropout added for regularization)
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())
        self.private_t.add_module('private_t_dropout', nn.Dropout(dropout_rate))

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        self.private_v.add_module('private_v_dropout', nn.Dropout(dropout_rate))

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())
        self.private_a.add_module('private_a_dropout', nn.Dropout(dropout_rate))


        ##########################################
        # shared encoder  (Dropout added for regularization)
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        self.shared.add_module('shared_dropout', nn.Dropout(dropout_rate))


        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))


        # CHANGED: Split fusion into feature extraction and classification
        self.fusion_layer = nn.Sequential()
        self.fusion_layer.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion_layer.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_layer.add_module('fusion_layer_1_activation', self.activation)
        
        self.output_layer = nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, dropout=0.3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        ##########################################
        # Hierarchical Confidence Scorers
        ##########################################
        self.conf_scorer_t = ModalityConfidenceScorer(config.hidden_size)
        self.conf_scorer_v = ModalityConfidenceScorer(config.hidden_size)
        self.conf_scorer_a = ModalityConfidenceScorer(config.hidden_size)

        # Placeholder: filled after each alignment() forward pass
        self.modal_confidences = None

        

    def alignment(self, text_vec, image_vec, behavior_vec):
        
        # Shared-private encoders
        self.shared_private(text_vec, image_vec, behavior_vec)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()

        # ── HIERARCHICAL CONFIDENCE FUSION ──────────────────────────────────
        # Ablation flags (getattr returns False by default → original behavior)
        _no_cf  = getattr(self.config, 'no_confidence_fusion', False)
        _no_mc  = getattr(self.config, 'no_mc_uncertainty',    False)
        _no_im  = getattr(self.config, 'no_image_masking',     False)

        if _no_cf:
            # Ablation: equal 1/3 weights, skip confidence scoring entirely
            N = text_vec.size(0)
            conf_weights = torch.ones(N, 3, device=text_vec.device) / 3.0
        else:
            # 1. Score each modality's private representation
            use_mc = not _no_mc
            conf_t = self.conf_scorer_t(self.utt_private_t, use_mc=use_mc)   # [N]
            conf_v = self.conf_scorer_v(self.utt_private_v, use_mc=use_mc)   # [N]
            conf_a = self.conf_scorer_a(self.utt_private_a, use_mc=use_mc)   # [N]

            # 2. Zero-out image confidence when raw image_vec is all-zeros
            #    (user posted no image → ViT returns zero vector)
            if not _no_im:
                image_is_zero = (image_vec.abs().sum(dim=-1) < 1e-4).float()   # [N]
                conf_v = conf_v * (1.0 - image_is_zero)

            # 3. Softmax over [text, image, behavior] to get normalised weights
            conf_stack = torch.stack([conf_t, conf_v, conf_a], dim=-1)   # [N, 3]
            conf_weights = torch.softmax(conf_stack, dim=-1)              # [N, 3]

        # 4. Store for monitoring / interpretability
        self.modal_confidences = conf_weights.detach()   # [N, 3]

        # 5. Apply weights to private and shared representations
        wt = conf_weights[:, 0:1]   # [N, 1]
        wv = conf_weights[:, 1:2]
        wa = conf_weights[:, 2:3]

        private_t_w = self.utt_private_t * wt
        private_v_w = self.utt_private_v * wv
        private_a_w = self.utt_private_a * wa
        shared_t_w  = self.utt_shared_t  * wt
        shared_v_w  = self.utt_shared_v  * wv
        shared_a_w  = self.utt_shared_a  * wa

        # 6. Transformer fusion on confidence-weighted tokens (batch_first=True: [N, 6, H])
        h = torch.stack((private_t_w, private_v_w, private_a_w,
                         shared_t_w, shared_v_w, shared_a_w), dim=1)   # [N, 6, H]
        h = self.transformer_encoder(h)                                # [N, 6, H]
        h = h.reshape(h.size(0), -1)                                   # [N, 6*H]
        # ─────────────────────────────────────────────────────────────────────

        features = self.fusion_layer(h)
        logits = self.output_layer(features)
        return logits, features
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    def shared_private(self, utterance_t, utterance_v, utterance_a):
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)


        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)


    def forward(self, text_vec, image_vec, behavior_vec):
        logits, features = self.alignment(text_vec, image_vec, behavior_vec)
        return logits

class TemporalMISA(nn.Module):
    def __init__(self, config):
        super(TemporalMISA, self).__init__()
        self.config = config
        
        # 1. Base MISA model for single-week feature fusion
        # behavior input: 16-dim rich temporal features (from update_behavior_features.py)
        # (BehaviorSequenceEncoder is defined above but not used in this forward path,
        #  as 68% of weeks have only 1 post, making per-post LSTM redundant)
        self.misa = MISA(config)
        
        # 2. Temporal Evolution Module (LSTM over weeks)
        self.temporal_input_size  = config.hidden_size * 3
        self.temporal_hidden_size = 32  # Reduced: 64→32 to cut capacity
        
        self.temporal_rnn = nn.LSTM(
            input_size=self.temporal_input_size, 
            hidden_size=self.temporal_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.temporal_dropout = nn.Dropout(p=config.dropout)  # 8.2: 跟随 Config.dropout（当前 0.3）
        
        # 3. Global text mean residual (shortcut from mean-pooled text → hidden space)
        # This ensures HAD-M3H can learn at least what a mean-pooling LR baseline learns.
        # The gate projects 768-dim global text mean into temporal_hidden_size space and
        # adds it residually to the LSTM output before classification.
        self.global_text_residual = nn.Linear(config.embedding_size, self.temporal_hidden_size)

        # 4. Prediction Head
        self.future_predictor = nn.Linear(self.temporal_hidden_size, config.num_classes)
        
        # Ablation: no_temporal_lstm — direct prediction from last-week features (skip LSTM)
        self.no_lstm_predictor = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.temporal_input_size, config.num_classes)
        )
        
    def forward(self, text_seq, image_seq, behavior_seq, lengths=None):
        """
        Args:
            text_seq     : [B, T, text_dim]
            image_seq    : [B, T, visual_dim]
            behavior_seq : [B, T, 16]  rich per-week temporal behavior features
            lengths      : [B]  actual week-sequence lengths (for temporal LSTM)
        """
        batch_size, seq_len, _ = text_seq.size()
        
        # 1. Flatten batch x seq_len for MISA
        text_flat     = text_seq.reshape(batch_size * seq_len, -1)
        image_flat    = image_seq.reshape(batch_size * seq_len, -1)
        behavior_flat = behavior_seq.reshape(batch_size * seq_len, -1)

        # 2. MISA: private/shared decomposition + confidence-weighted fusion
        _, week_features = self.misa.alignment(text_flat, image_flat, behavior_flat)

        # Confidence weights: [B*T, 3] -> [B, T, 3]
        conf_seq = self.misa.modal_confidences.reshape(batch_size, seq_len, 3)

        # Week feature sequence: [B*T, H*3] -> [B, T, H*3]
        week_features_seq = week_features.view(batch_size, seq_len, -1)

        # --- 全局文本均值（masked mean，排除 padding 周）---
        if lengths is not None:
            len_dev = lengths.to(text_seq.device).float()
            # 构造 mask [B, T, 1]，padding 位置为 0
            arange = torch.arange(seq_len, device=text_seq.device).unsqueeze(0)  # [1, T]
            mask = (arange < lengths.to(text_seq.device).unsqueeze(1)).unsqueeze(-1).float()  # [B, T, 1]
            global_text_mean = (text_seq * mask).sum(dim=1) / len_dev.unsqueeze(-1).clamp(min=1)
        else:
            global_text_mean = text_seq.mean(dim=1)  # [B, 768]
        # 映射到 temporal_hidden_size 空间，用 tanh 做门控
        global_text_res = torch.tanh(self.global_text_residual(global_text_mean))  # [B, 32]

        # Ablation: no_temporal_lstm — use only last valid week (skip LSTM)
        if getattr(self.config, 'no_temporal_lstm', False):
            if lengths is not None:
                last_valid = (lengths - 1).clamp(min=0).to(week_features_seq.device)
                last_feat = week_features_seq[torch.arange(batch_size, device=week_features_seq.device), last_valid]
            else:
                last_feat = week_features_seq[:, -1, :]
            future_risk = self.no_lstm_predictor(last_feat)
            return {
                'week_states':       week_features_seq,
                'future_risk':       future_risk,
                'modal_confidences': conf_seq,
            }
        
        # 3. Temporal LSTM over weeks (with pack_padded for variable-length sequences)
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            week_features_sorted = week_features_seq[sorted_idx]
            
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                week_features_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
            _, (h_n, _) = self.temporal_rnn(packed)
            
            last_hidden_sorted = h_n[-1]
            _, unsort_idx = sorted_idx.sort()
            last_hidden = last_hidden_sorted[unsort_idx]
        else:
            rnn_out, _ = self.temporal_rnn(week_features_seq)
            last_hidden = rnn_out[:, -1, :]
        
        # 4. Classification（LSTM 输出 + 全局文本残差）
        last_hidden = self.temporal_dropout(last_hidden + global_text_res)
        future_risk = self.future_predictor(last_hidden)
        
        return {
            'week_states':       week_features_seq,
            'future_risk':       future_risk,
            'modal_confidences': conf_seq   # [B, T, 3]
        }


# ════════════════════════════════════════════════════════════════════
# MMIM: MultiModal InfoMax (Han et al., EMNLP 2021)
# Adapted for our task: pre-extracted features + temporal LSTM
# Key change: replaces MISA's private/shared disentanglement with
# CPC-based MI maximization between fusion result Z and each modality.
# Reference: https://github.com/declare-lab/Multimodal-Infomax
# ════════════════════════════════════════════════════════════════════

class MMIMFusion(nn.Module):
    """Per-week multimodal fusion with CPC MI maximization.

    Architecture:
      1. Project each modality to unified H-dim space (with LayerNorm)
      2. Fuse via MLP: [h_t, h_v, h_a] → Z  (same output dim H*3 as MISA)
      3. CPC reverse predictors: G_m(Z) ≈ h_m  (InfoNCE contrastive loss)

    No private/shared decomposition, no adversarial discriminator,
    no reconstruction loss — much simpler than MISA for small datasets.
    """
    def __init__(self, config):
        super().__init__()
        H = config.hidden_size   # 64

        # Per-modality encoders → unified H-dim
        self.enc_t = nn.Sequential(
            nn.Linear(config.embedding_size, H), nn.LayerNorm(H), nn.ReLU())
        self.enc_v = nn.Sequential(
            nn.Linear(config.visual_size, H), nn.LayerNorm(H), nn.ReLU())
        self.enc_a = nn.Sequential(
            nn.LayerNorm(config.acoustic_size),
            nn.Linear(config.acoustic_size, H), nn.LayerNorm(H), nn.ReLU())

        # Fusion network: concat(h_t, h_v, h_a) → Z  [N, H*3]
        self.fusion_net = nn.Sequential(
            nn.Linear(H * 3, H * 3), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(H * 3, H * 3))

        # CPC reverse predictors G_m: Z → ĥ_m
        self.cpc_t = nn.Linear(H * 3, H)
        self.cpc_v = nn.Linear(H * 3, H)
        self.cpc_a = nn.Linear(H * 3, H)

    def forward(self, text_vec, image_vec, behavior_vec):
        ht = self.enc_t(text_vec)        # [N, H]
        hv = self.enc_v(image_vec)       # [N, H]
        ha = self.enc_a(behavior_vec)    # [N, H]
        z  = self.fusion_net(torch.cat([ht, hv, ha], dim=-1))  # [N, 3H]
        return z, ht, hv, ha

    @staticmethod
    def _nce(pred, target, temperature=0.1):
        """InfoNCE contrastive loss.  pred[i] should match target[i]."""
        pred_n   = F.normalize(pred,   dim=-1)
        target_n = F.normalize(target, dim=-1)
        logits   = pred_n @ target_n.T / temperature   # [N, N]
        labels   = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def cpc_loss(self, z, ht, hv, ha):
        """Fusion-level CPC: Z should 'predict' each modality representation.
        Three terms: Z→text, Z→image, Z→behavior."""
        return (self._nce(self.cpc_t(z), ht) +
                self._nce(self.cpc_v(z), hv) +
                self._nce(self.cpc_a(z), ha))


class MMIMTemporalModel(nn.Module):
    """Temporal mental health risk model using MMIM-based per-week fusion.

    Architecture mirrors TemporalMISA but replaces MISA's disentanglement
    module with MMIMFusion.  The temporal LSTM and global-text-residual
    shortcut are kept identical for fair comparison.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion = MMIMFusion(config)

        self.temporal_input_size  = config.hidden_size * 3   # 192
        self.temporal_hidden_size = 32

        self.temporal_rnn = nn.LSTM(
            self.temporal_input_size, self.temporal_hidden_size,
            num_layers=1, batch_first=True)
        self.temporal_dropout = nn.Dropout(config.dropout)

        # Global text residual shortcut (same as TemporalMISA)
        self.global_text_residual = nn.Linear(
            config.embedding_size, self.temporal_hidden_size)

        self.future_predictor = nn.Linear(
            self.temporal_hidden_size, config.num_classes)

    def forward(self, text_seq, image_seq, behavior_seq, lengths=None):
        B, T, _ = text_seq.shape

        # 1. Per-week MMIM fusion (flattened)
        z, ht, hv, ha = self.fusion(
            text_seq.reshape(B * T, -1),
            image_seq.reshape(B * T, -1),
            behavior_seq.reshape(B * T, -1))
        week_seq = z.view(B, T, -1)   # [B, T, 3H]

        # 2. Global text residual (masked mean over valid weeks)
        if lengths is not None:
            arange = torch.arange(T, device=text_seq.device).unsqueeze(0)
            mask   = (arange < lengths.to(text_seq.device).unsqueeze(1)).unsqueeze(-1).float()
            global_text = (text_seq * mask).sum(1) / lengths.float().to(text_seq.device).unsqueeze(-1).clamp(min=1)
        else:
            global_text = text_seq.mean(1)
        global_res = torch.tanh(self.global_text_residual(global_text))  # [B, 32]

        # 3. Temporal LSTM
        if lengths is not None:
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            week_sorted = week_seq[sorted_idx]
            packed = pack_padded_sequence(
                week_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True)
            _, (h_n, _) = self.temporal_rnn(packed)
            last_sorted = h_n[-1]
            _, unsort_idx = sorted_idx.sort()
            last_hidden = last_sorted[unsort_idx]
        else:
            _, (h_n, _) = self.temporal_rnn(week_seq)
            last_hidden = h_n[-1]

        # 4. Classification with global text residual
        last_hidden = self.temporal_dropout(last_hidden + global_res)
        future_risk = self.future_predictor(last_hidden)

        return {
            'future_risk': future_risk,
            '_z': z, '_ht': ht, '_hv': hv, '_ha': ha,
        }

    def cpc_loss(self, outputs):
        """Compute CPC loss from the stored per-week representations."""
        return self.fusion.cpc_loss(
            outputs['_z'], outputs['_ht'], outputs['_hv'], outputs['_ha'])
