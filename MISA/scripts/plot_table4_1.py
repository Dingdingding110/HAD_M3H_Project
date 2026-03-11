# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

models = [
    'Text-only', 'Image-only', 'Behavior-only',
    'Concat', 'Weighted', 'Standard-MISA', 'HAD-M3H'
]
accuracies = [0.8833, 0.8167, 0.6333, 0.8667, 0.9083, 0.8083, 0.85]
f1s = [0.8831, 0.8163, 0.6284, 0.8658, 0.9083, 0.8081, 0.85]
aucs = [0.9549, 0.8999, 0.6224, 0.9132, 0.9725, 0.8865, 0.9141]

out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots')
os.makedirs(out_dir, exist_ok=True)

# Plot accuracy bar chart
x = np.arange(len(models))
width = 0.6
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x, accuracies, width, color='#4C72B0')

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('Accuracy')
ax.set_title('Table 4.1 — Benchmark Comparison (Accuracy)')

# Annotate values
for i,v in enumerate(accuracies):
    ax.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
output_path = os.path.join(out_dir, 'table4_1_accuracy_bar.png')
fig.savefig(output_path, dpi=200)
print('Saved:', output_path)

# Also save a grouped bar chart with Accuracy/F1/AUC
fig2, ax2 = plt.subplots(figsize=(12,6))
bar_width = 0.25
ax2.bar(x - bar_width, accuracies, bar_width, label='Accuracy')
ax2.bar(x, f1s, bar_width, label='F1')
ax2.bar(x + bar_width, aucs, bar_width, label='AUC')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, ha='right')
ax2.set_ylim(0,1)
ax2.set_ylabel('Score')
ax2.set_title('Table 4.1 — Accuracy / F1 / AUC')
ax2.legend()
for i in range(len(models)):
    ax2.text(i - bar_width, accuracies[i] + 0.01, f'{accuracies[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i, f1s[i] + 0.01, f'{f1s[i]:.4f}', ha='center', fontsize=8)
    ax2.text(i + bar_width, aucs[i] + 0.01, f'{aucs[i]:.4f}', ha='center', fontsize=8)
plt.tight_layout()
output_path2 = os.path.join(out_dir, 'table4_1_grouped.png')
fig2.savefig(output_path2, dpi=200)
print('Saved:', output_path2)
