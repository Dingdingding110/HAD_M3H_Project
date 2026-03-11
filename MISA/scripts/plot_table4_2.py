# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

models = [
    'HAD-M3H', 'wo-CF', 'wo-MC', 'wo-IM', 'wo-LSTM', 'wo-BF'
]
accuracies = [0.85, 0.825, 0.8333, 0.8, 0.7083, 0.825]
f1s = [0.85, 0.8249, 0.8332, 0.798, 0.7082, 0.8245]
aucs = [0.9141, 0.9002, 0.9127, 0.8712, 0.762, 0.9174]

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
os.makedirs(out_dir, exist_ok=True)

x = np.arange(len(models))
width = 0.6
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x, accuracies, width, color='#55A868')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('Accuracy')
ax.set_title('Table 4.2 — Ablation Study (Accuracy)')
for i,v in enumerate(accuracies):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
output_path = os.path.join(out_dir, 'table4_2_accuracy_bar.png')
fig.savefig(output_path, dpi=200)
print('Saved:', output_path)

# grouped chart
fig2, ax2 = plt.subplots(figsize=(12,6))
bar_w = 0.25
ax2.bar(x - bar_w, accuracies, bar_w, label='Accuracy')
ax2.bar(x, f1s, bar_w, label='F1')
ax2.bar(x + bar_w, aucs, bar_w, label='AUC')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, ha='right')
ax2.set_ylim(0,1)
ax2.set_ylabel('Score')
ax2.set_title('Table 4.2 — Accuracy / F1 / AUC')
ax2.legend()
for i in range(len(models)):
    ax2.text(i - bar_w, accuracies[i] + 0.01, f'{accuracies[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i, f1s[i] + 0.01, f'{f1s[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i + bar_w, aucs[i] + 0.01, f'{aucs[i]:.4f}', ha='center', fontsize=8)
plt.tight_layout()
output_path2 = os.path.join(out_dir, 'table4_2_grouped.png')
fig2.savefig(output_path2, dpi=200)
print('Saved:', output_path2)
