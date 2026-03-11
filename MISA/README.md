# HAD-M3H / MMIM-HAD: 多模态时序心理健康风险检测

> **H**ierarchical **A**daptive **D**etection — **M**ultimodal, **M**ulti-week, **M**ental **H**ealth (Reddit)

本项目面向 Reddit 社交媒体数据的**抑郁/焦虑风险二分类**任务，融合文本、图像、行为三种模态的多周时序信息，逐步演进出两代核心模型：
- **HAD-M3H**：基于 MISA 私有-共享解耦 + MC-Dropout 置信度融合 + 时序 LSTM
- **MMIM-HAD**（最新，最优）：基于互信息最大化（CPC）的多模态融合 + 时序 LSTM，当前所有模型中精度最高（Acc **90.67%**，AUC **0.9601**）

---

## 目录

1. [项目背景](#项目背景)
2. [数据集与特征](#数据集与特征)
3. [模型架构](#模型架构)
   - [HAD-M3H](#had-m3h-第一代)
   - [MMIM-HAD（推荐）](#mmim-had-第二代推荐)
4. [实验结果](#实验结果)
5. [消融分析](#消融分析)
6. [环境配置与运行](#环境配置与运行)
7. [优化方向与参考文献](#优化方向与参考文献)
8. [引用](#引用)

---

## 项目背景

心理健康问题的早期发现对干预和预防至关重要。Reddit 等社交媒体平台留下了用户随时间变化的丰富多模态行为轨迹——帖子文本、配图、发帖节律——为非侵入式风险筛查提供了可能。

本项目从 Reddit 抓取 **600 名用户**（300 高风险 / 300 对照）的历史时间线，将每位用户的多周数据建模为三模态时序序列，利用预训练表示（RoBERTa 文本、ViT 图像）与手工行为特征，训练二分类模型判断未来心理健康风险。

**主要挑战：**
- 小样本（600 用户）：深度模型易过拟合
- 模态缺失：部分周次无图像（约 60%）
- 时序不等长：不同用户活跃周数差异显著
- 类别不均衡：部分子任务中风险用户比例偏低

---

## 数据集与特征

| 项目 | 说明 |
|------|------|
| **用户数** | 600（风险 300 / 对照 300）|
| **时序粒度** | 以"周"为单位，每用户平均约 8 个时间步 |
| **文本特征** | `RoBERTa-base` CLS 向量，768 维 |
| **图像特征** | `ViT-B/16` CLS 向量，768 维；缺失时置零 |
| **行为特征** | 16 维富特征向量（发帖时段、节律、互动量等，经 log1p 标准化）|
| **标签** | 二分类（1 = 风险用户，0 = 对照用户）|
| **划分** | 5 折分层交叉验证（StratifiedKFold, seed=42），训练 480 / 验证 120 |

---

## 模型架构

### HAD-M3H（第一代）

```
[文本 768] ──┐
[图像 768] ──┤─→ MISA (私有/共享解耦 + 对抗判别器 + 重建) ──→ 融合特征 [3H]
[行为  16] ──┘         ↑ MC-Dropout置信度加权

融合特征序列 [B, T, 3H] ──→ 时序 LSTM ──→ [B, 32]
                                              + 全局文本残差 [B, 32]
                                              ↓
                                        分类头 → {0, 1}
```

**关键组件**：
- **MISA 私有-共享解耦**：每模态分解为私有子空间（模态特异）+ 共享子空间（跨模态不变），通过差异损失、对抗判别器、重建损失约束
- **MC-Dropout 置信度融合**：对三模态特征进行 8 次 Dropout 前向传播，用方差估计不确定性，高不确定性模态降低权重
- **时序 LSTM**：将各周次融合特征建模为序列，捕获心理状态随时间的演化
- **全局文本残差**：将全局文本均值直接旁路连接到 LSTM 输出，防止时序建模丢失文本主导信号

> **问题发现**：HAD-M3H 在 600 用户小样本下 Acc = 86.67%，**低于** RoBERTa+MLP 基线（90.33%）。分析表明 MISA 的复杂正则（diff_loss + recon_loss + 对抗）在小数据下产生过强约束，MC-Dropout 在小批量（8样本）下方差估计噪声大，置信度融合反而引入误差（wo-CF = 89.33% > HAD-M3H）。

---

### MMIM-HAD（第二代，推荐）

灵感来源：Han et al., **"Improving Multimodal Fusion with Hierarchical Mutual Information Maximization"**, EMNLP 2021

```
[文本 768] ──→ enc_t (Linear→LN→ReLU) ──→ h_t [H=64]  ─┐
[图像 768] ──→ enc_v (Linear→LN→ReLU) ──→ h_v [H=64]  ──┤ concat [3H=192]
[行为  16] ──→ enc_a (LN→Linear→LN→ReLU) → h_a [H=64] ─┘
                                                  ↓
                              融合 MLP [3H → 3H] → Z [3H=192]
                                                  ↓
              CPC 损失: G_t(Z)≈h_t, G_v(Z)≈h_v, G_a(Z)≈h_a  (InfoNCE)

per-week Z 序列 [B, T, 192] ──→ 时序 LSTM ──→ [B, 32]
                                                + 全局文本残差 [B, 32]
                                                ↓
                                          分类头 → {0, 1}
```

**总损失**：

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \lambda \cdot \mathcal{L}_{\text{CPC}}$$

$$\mathcal{L}_{\text{CPC}} = \mathcal{L}_{\text{NCE}}(G_t(Z), h_t) + \mathcal{L}_{\text{NCE}}(G_v(Z), h_v) + \mathcal{L}_{\text{NCE}}(G_a(Z), h_a)$$

其中 $\lambda = 0.1$，InfoNCE 温度 $\tau = 0.1$，批内负样本对比。

**相比 HAD-M3H 的改进**：

| 方面 | HAD-M3H | MMIM-HAD |
|------|---------|----------|
| 融合机制 | MISA 私有/共享解耦 | CPC 互信息最大化 |
| 正则损失 | diff + recon + adversarial (3项) | 仅 CPC (1项) |
| 置信度融合 | MC-Dropout (小数据噪声大) | 无，CPC 隐式对齐 |
| 参数量 | ~180K | ~130K |
| 训练稳定性 | 对超参敏感 | 更稳定（无对抗训练）|

---

## 实验结果

**5 折分层交叉验证（600 用户，seed=42）**

| 模型 | Acc (mean ± std) | F1 (weighted) | AUC |
|------|-----------------|---------------|-----|
| **MMIM-HAD** | **0.9067 ± 0.0179** | **0.9065** | **0.9601** |
| RoBERTa-Mean+MLP | 0.9033 ± 0.0272 | 0.9032 | 0.9563 |
| Weighted LSTM | 0.8950 ± 0.0128 | 0.8948 | 0.9544 |
| wo-CF (无置信度融合) | 0.8933 ± 0.0335 | 0.8933 | 0.9309 |
| Concat LSTM | 0.8833 ± 0.0156 | 0.8833 | 0.9435 |
| RoBERTa-Mean+LR | 0.8800 ± 0.0135 | 0.8798 | 0.9464 |
| wo-IM (无图像缺失掩码) | 0.8800 ± 0.0325 | 0.8797 | 0.9259 |
| wo-BF (无富行为特征) | 0.8783 ± 0.0337 | 0.8782 | 0.9359 |
| Text-only LSTM | 0.8767 ± 0.0312 | 0.8761 | 0.9358 |
| HAD-M3H | 0.8667 ± 0.0190 | 0.8664 | 0.9289 |
| wo-MC (无MC不确定性) | 0.8633 ± 0.0174 | 0.8631 | 0.9162 |
| Image-only LSTM | 0.7950 ± 0.0284 | 0.7929 | 0.8882 |
| Standard-MISA | 0.7767 ± 0.0134 | 0.7758 | 0.8251 |
| wo-LSTM (无时序) | 0.7667 ± 0.0154 | 0.7663 | 0.8084 |
| Behavior-only LSTM | 0.6450 ± 0.0210 | 0.6407 | 0.6435 |

**关键发现**：
- MMIM-HAD 是所有 15 个模型中 Acc 和 AUC 最高的模型
- MMIM-HAD vs HAD-M3H：**+4.00%**；vs RoBERTa+MLP（此前最强）：**+0.34%**
- MMIM-HAD 标准差（0.0179）优于 wo-CF（0.0335），稳定性更好
- 时序 LSTM 贡献显著：MMIM-HAD vs wo-LSTM = **+14.00%**（0.9067 vs 0.7667）
- 置信度融合（CF）反而有害：wo-CF（89.33%）> HAD-M3H（86.67%），说明 MC-Dropout 在小样本下引入了噪声

---

## 消融分析

以下消融均基于 HAD-M3H（TemporalMISA）进行：

| 变体 | 说明 | Acc | 相对 HAD-M3H |
|------|------|-----|-------------|
| wo-CF | 去除置信度融合，改为简单均值加权 | 0.8933 | **+2.66%** |
| wo-MC | 保留简单学习置信度，去除 MC-Dropout | 0.8633 | -0.34% |
| wo-IM | 去除图像缺失时置零掩码 | 0.8800 | +1.33% |
| wo-LSTM | 去除时序 LSTM，直接用最后一周特征 | 0.7667 | -10.00% |
| wo-BF | 行为特征降至 4 维（去掉富特征） | 0.8783 | +1.16% |

**结论**：
1. 时序 LSTM 是最重要的组件（去除后下降 10%）
2. MC-Dropout 置信度融合带来负收益，应在后续工作中移除
3. 简单的图像掩码（woIM vs HAD-M3H: +1.33%）和富行为特征（wo-BF vs HAD-M3H: +1.16%）效果有限

---

## 环境配置与运行

```bash
# 创建环境
conda env create -f environment.yml
conda activate hadm3h

# 数据预处理（已完成，pkl 文件已存在时跳过）
python collect_temporal_data.py
python MISA/src/extract_features.py

# 运行5折交叉验证（支持断点续传）
cd ~/zzq/HAD_M3H_project
CUDA_VISIBLE_DEVICES=0 python -u MISA/src/run_cv_experiments_v3.py 2>&1 | tee cv_result_v3.log
```

**结果文件**：`MISA/cv_results_v3.json`（15 模型 × 5 折，每条 `[acc, f1, auc]`）

---

## 优化方向与参考文献

### 方向一：跨模态注意力机制

当前 MMIM-HAD 的 CPC 融合是**无监督对齐**，没有显式的跨模态交互（如文本引导图像关注）。引入**跨模态 Transformer**可通过多头注意力让模态间相互查询，学习更细粒度的交互表示。

**关键挑战**：600 用户的小样本可能导致注意力矩阵学习不稳定，需配合 LoRA 或 Adapter 参数高效微调。

推荐论文：
- Tsai et al., **"Multimodal Transformer for Unaligned Multimodal Language Sequences"**, ACL 2019 [[论文]](https://arxiv.org/abs/1906.00295) — MulT，引入定向跨模态注意力，无需对齐
- Rahman et al., **"Integrating Multimodal Information in Large Pretrained Transformers"**, ACL 2020 [[论文]](https://arxiv.org/abs/1908.05787) — MAG-BERT，在 BERT 层内注入视觉/声学门控偏移

---

### 方向二：完整 MMIM 的 BA 下界（inter-modality MI）

当前 MMIM-HAD 只实现了**fusion-level CPC**（单向 Z→模态），原论文还有**inter-modality BA 下界**（任意两模态之间直接最大化 MI，用 GMM 估计边际熵）。在原论文中 BA+CPC 的联合使用带来约 1% 的额外提升。

**注意**：BA 损失在小批量（≤8）下数值不稳定（方差大），在扩大 batch size 或用梯度累积后再启用。

推荐论文：
- Han et al., **"Improving Multimodal Fusion with Hierarchical Mutual Information Maximization"**, EMNLP 2021 [[论文]](https://arxiv.org/abs/2109.00412) — MMIM 原文，详细推导 BA+CPC 两级 MI 损失

---

### 方向三：时序建模增强（Transformer/Mamba 替换 LSTM）

当前时序编码器是单层 LSTM（hidden=32），对**长程依赖**和**稀疏激活**（某周无帖）的建模能力有限。可考虑：
- **Temporal Fusion Transformer (TFT)**：专为多步时序预测设计，含变量选择网络和门控残差
- **Mamba（SSM）**：线性复杂度状态空间模型，比 Transformer 更适合中长序列（8-52 周）

推荐论文：
- Lim et al., **"Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"**, IJF 2021 [[论文]](https://arxiv.org/abs/1912.09363)
- Gu & Dao, **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"**, COLM 2024 [[论文]](https://arxiv.org/abs/2312.00752)

---

### 方向四：自监督预训练 + 迁移学习

600 用户的标注数据极少。在**大量未标注 Reddit 帖子**（可从 PushShift 获取）上进行自监督预训练，再迁移到下游任务，能显著缓解过拟合。

可行方案：
- **对比预训练**：将同一用户不同周次的帖子视为正样本对，不同用户为负样本（类 SimCLR）
- **掩码多模态建模**：随机掩盖某模态的某周数据，要求模型从其他模态/时间步重建（类 MAE/VideoMAE）

推荐论文：
- Chen et al., **"A Simple Framework for Contrastive Learning of Visual Representations"**, ICML 2020 [[论文]](https://arxiv.org/abs/2002.05709) — SimCLR
- He et al., **"Masked Autoencoders Are Scalable Vision Learners"**, CVPR 2022 [[论文]](https://arxiv.org/abs/2111.06377) — MAE
- Liu et al., **"MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare"**, LREC 2022 [[论文]](https://arxiv.org/abs/2110.01296) — 专为心理健康任务预训练的 BERT

---

### 方向五：图神经网络用户关系建模

Reddit 用户之间存在隐式社交关系（互回复、共同订阅子版块），将此关系建模为**用户图**，通过 GNN 传播社区信号可能提供个体特征之外的群体信息。

推荐论文：
- Yao et al., **"Graph Convolutional Networks for Text Classification"**, AAAI 2019 [[论文]](https://arxiv.org/abs/1809.05679) — TextGCN
- Guo et al., **"Leveraging Graph to Improve Abstractive Multi-Document Summarization"**, ACL 2021
- Ji et al., **"Suicidal Ideation Detection: A Review of Machine Learning Methods and Applications"**, IEEE TNNLS 2021 [[论文]](https://doi.org/10.1109/TNNLS.2021.3098336) — 综述

---

### 方向六：标签平滑 + 焦点损失（针对小样本）

当前使用 **CrossEntropy + label_smoothing=0.1**。对小样本进一步优化：
- **Focal Loss**（Lin et al. 2017）：自动降低已正确分类样本的权重，专注困难样本
- **MixUp / CutMix 数据增强**：在特征空间插值创造伪样本，扩充有效训练集
- **Sharpness-Aware Minimization (SAM)**：优化器层面寻求平坦极小值，泛化性更强

推荐论文：
- Lin et al., **"Focal Loss for Dense Object Detection"**, ICCV 2017 [[论文]](https://arxiv.org/abs/1708.02002) — RetinaNet
- Zhang et al., **"Mixup: Beyond Empirical Risk Minimization"**, ICLR 2018 [[论文]](https://arxiv.org/abs/1710.09412)
- Foret et al., **"Sharpness-Aware Minimization for Efficiently Improving Generalization"**, ICLR 2021 [[论文]](https://arxiv.org/abs/2010.01412) — SAM

---

### 方向七：可解释性与临床应用

当前为黑盒决策，临床场景需要模型解释原因（哪段时间、哪种模态触发了高风险预测）。

推荐论文：
- Ribeiro et al., **"'Why Should I Trust You?' Explaining the Predictions of Any Classifier"**, KDD 2016 [[论文]](https://arxiv.org/abs/1602.04938) — LIME
- Lundberg & Lee, **"A Unified Approach to Interpreting Model Predictions"**, NeurIPS 2017 [[论文]](https://arxiv.org/abs/1705.07874) — SHAP
- Yates et al., **"Depression and Self-Harm Risk Assessment in Online Forums"**, EMNLP 2017 [[论文]](https://arxiv.org/abs/1709.01848)

---

### 方向八：大语言模型集成

GPT-4/LLaMA 等 LLM 对帖子文本有更强的理解能力（情感细节、隐喻、否定表达），可替换或补充 RoBERTa 向量。

推荐论文：
- Yang et al., **"Towards Interpretable Mental Health Analysis with Large Language Models"**, EMNLP 2023 [[论文]](https://arxiv.org/abs/2304.03347)
- Xu et al., **"Mental-LLM: Leveraging Large Language Models for Mental Health Prediction via Online Text Data"**, arXiv 2023 [[论文]](https://arxiv.org/abs/2307.14385)
- Abdullah et al., **"Detection and Prediction of Future Mental Disorder from Social Media Data Using Machine Learning, Ensemble Learning and Large Language Models"**, 2024 — 本项目基线来源

---

## 引用

如果本项目对您的研究有用，请引用：

```bibtex
@inproceedings{han2021mmim,
  title     = {Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis},
  author    = {Han, Wei and Chen, Hui and Poria, Soujanya},
  booktitle = {Proceedings of EMNLP 2021},
  year      = {2021}
}

@inproceedings{hazarika2020misa,
  title     = {MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis},
  author    = {Hazarika, Devamanyu and Zimmermann, Roger and Poria, Soujanya},
  booktitle = {Proceedings of ACM MM 2020},
  year      = {2020}
}
```

---

*基于 MISA 开源代码改进，原始 LICENSE 见 [LICENSE](LICENSE) 文件。*
