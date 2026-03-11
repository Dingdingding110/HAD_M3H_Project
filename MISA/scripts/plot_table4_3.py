# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

steps = ['k=0', 'k=1', 'k=2', 'k=3']
accuracies = [0.85, 0.775, 0.775, 0.775]
f1s = [0.85, 0.7749, 0.7749, 0.7747]
aucs = [0.9141, 0.8899, 0.8674, 0.8626]

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
os.makedirs(out_dir, exist_ok=True)

# Accuracy bar chart
x = np.arange(len(steps))
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(x, accuracies, color='#C44E52', width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(steps)
ax.set_ylim(0,1)
ax.set_ylabel('Accuracy')
ax.set_title('Table 4.3 — k-step Prediction (Accuracy)')
for i,v in enumerate(accuracies):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.tight_layout()
output_path = os.path.join(out_dir, 'table4_3_accuracy_bar.png')
fig.savefig(output_path, dpi=200)
print('Saved:', output_path)

# Grouped chart Accuracy/F1/AUC
fig2, ax2 = plt.subplots(figsize=(10,5))
bar_w = 0.25
ax2.bar(x - bar_w, accuracies, bar_w, label='Accuracy')
ax2.bar(x, f1s, bar_w, label='F1')
ax2.bar(x + bar_w, aucs, bar_w, label='AUC')
ax2.set_xticks(x)
ax2.set_xticklabels(steps)
ax2.set_ylim(0,1)
ax2.set_ylabel('Score')
ax2.set_title('Table 4.3 — Accuracy / F1 / AUC')
ax2.legend()
for i in range(len(steps)):
    ax2.text(i - bar_w, accuracies[i] + 0.01, f'{accuracies[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i, f1s[i] + 0.01, f'{f1s[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i + bar_w, aucs[i] + 0.01, f'{aucs[i]:.4f}', ha='center', fontsize=8)
plt.tight_layout()
output_path2 = os.path.join(out_dir, 'table4_3_grouped.png')
fig2.savefig(output_path2, dpi=200)
print('Saved:', output_path2)
