#!/usr/bin/env python3
"""
Generate publication-ready figures for the final report.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERATING PUBLICATION FIGURES")
print("=" * 60)

# ============================================================================
# FIGURE 1: Model Comparison Bar Chart
# ============================================================================
print("\n[1] Model Comparison...")

models = [
    ('ESN_500n', 0.763, 'Classical', '#2ecc71'),
    ('ESN_400n', 0.754, 'Classical', '#2ecc71'),
    ('ESN_300n', 0.733, 'Classical', '#2ecc71'),
    ('LSTM_128h_3L', 0.666, 'Deep Learning', '#3498db'),
    ('LSTM_128h', 0.662, 'Deep Learning', '#3498db'),
    ('LSTM_64h', 0.659, 'Deep Learning', '#3498db'),
    ('Weekly Profile', 0.645, 'Baseline', '#95a5a6'),
    ('Hybrid_8q_100n', 0.637, 'Quantum', '#9b59b6'),
    ('QuadQRC_8q', 0.558, 'Quantum', '#9b59b6'),
    ('MultiBasis_8q', 0.519, 'Quantum', '#9b59b6'),
]

fig, ax = plt.subplots(figsize=(12, 6))

names = [m[0] for m in models]
scores = [m[1] for m in models]
colors = [m[3] for m in models]

bars = ax.barh(range(len(models)), scores, color=colors, edgecolor='white', linewidth=0.5)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=10)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(names)
ax.set_xlabel('R² Score (Test Set)')
ax.set_title('Model Performance Comparison: EV Charging Demand Prediction\n(Palo Alto 2017-2019)')
ax.set_xlim(0, 0.85)
ax.axvline(x=0.645, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# Legend
legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Classical (ESN)'),
    mpatches.Patch(color='#3498db', label='Deep Learning (LSTM)'),
    mpatches.Patch(color='#9b59b6', label='Quantum (QRC)'),
    mpatches.Patch(color='#95a5a6', label='Baseline'),
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(output_dir / 'fig1_model_comparison.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig1_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig1_model_comparison.png/pdf")

# ============================================================================
# FIGURE 2: Training Time vs Accuracy
# ============================================================================
print("\n[2] Training Time vs Accuracy...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data: (name, r2, time, category)
time_data = [
    ('ESN_500n', 0.763, 1, 'Classical'),
    ('ESN_300n', 0.733, 0.5, 'Classical'),
    ('LSTM_128h_3L', 0.666, 31, 'Deep Learning'),
    ('LSTM_64h', 0.659, 38, 'Deep Learning'),
    ('Hybrid_8q_100n', 0.637, 1834, 'Quantum'),
    ('QuadQRC_8q', 0.558, 858, 'Quantum'),
]

colors_map = {'Classical': '#2ecc71', 'Deep Learning': '#3498db', 'Quantum': '#9b59b6'}

for name, r2, t, cat in time_data:
    ax.scatter(t, r2, s=200, c=colors_map[cat], edgecolor='white', linewidth=2, zorder=5)
    ax.annotate(name, (t, r2), textcoords="offset points", xytext=(10, 5), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel('Training Time (seconds, log scale)')
ax.set_ylabel('R² Score')
ax.set_title('Accuracy vs Computational Cost')

# Legend
for cat, color in colors_map.items():
    ax.scatter([], [], c=color, s=100, label=cat)
ax.legend(loc='lower left')

# Add quadrant annotations
ax.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=100, color='gray', linestyle=':', alpha=0.5)
ax.text(2, 0.78, 'IDEAL\n(Fast & Accurate)', fontsize=10, color='green', ha='center')
ax.text(500, 0.53, 'AVOID\n(Slow & Inaccurate)', fontsize=10, color='red', ha='center')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_time_vs_accuracy.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig2_time_vs_accuracy.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig2_time_vs_accuracy.png/pdf")

# ============================================================================
# FIGURE 3: Method Category Comparison
# ============================================================================
print("\n[3] Method Category Summary...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

categories = ['Classical\n(ESN)', 'Deep Learning\n(LSTM)', 'Quantum\n(QRC)']
best_r2 = [0.763, 0.666, 0.637]
best_time = [1, 31, 1834]
best_names = ['ESN_500n', 'LSTM_128h_3L', 'Hybrid_8q_100n']

colors = ['#2ecc71', '#3498db', '#9b59b6']

# R² comparison
axes[0].bar(categories, best_r2, color=colors, edgecolor='white')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Best Accuracy by Category')
axes[0].set_ylim(0, 0.85)
for i, (v, n) in enumerate(zip(best_r2, best_names)):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

# Training time comparison
axes[1].bar(categories, best_time, color=colors, edgecolor='white')
axes[1].set_ylabel('Training Time (s)')
axes[1].set_title('Training Time by Category')
axes[1].set_yscale('log')
for i, v in enumerate(best_time):
    axes[1].text(i, v * 1.5, f'{v}s', ha='center', fontsize=10)

# Efficiency (R² / log(time))
efficiency = [r2 / np.log10(max(t, 1) + 1) for r2, t in zip(best_r2, best_time)]
axes[2].bar(categories, efficiency, color=colors, edgecolor='white')
axes[2].set_ylabel('Efficiency (R² / log₁₀(time+1))')
axes[2].set_title('Cost-Efficiency by Category')
for i, v in enumerate(efficiency):
    axes[2].text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'fig3_category_summary.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig3_category_summary.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig3_category_summary.png/pdf")

# ============================================================================
# FIGURE 4: ESN Scaling
# ============================================================================
print("\n[4] ESN Scaling...")

esn_neurons = [100, 200, 300, 400, 500]
esn_r2 = [0.678, 0.703, 0.733, 0.754, 0.763]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(esn_neurons, esn_r2, 'o-', color='#2ecc71', markersize=10, linewidth=2)
ax.fill_between(esn_neurons, esn_r2, alpha=0.2, color='#2ecc71')

ax.set_xlabel('Number of Reservoir Neurons')
ax.set_ylabel('R² Score')
ax.set_title('ESN Performance vs Reservoir Size')

# Add LSTM reference line
ax.axhline(y=0.666, color='#3498db', linestyle='--', label='Best LSTM (0.666)')
ax.axhline(y=0.637, color='#9b59b6', linestyle='--', label='Best QRC (0.637)')

ax.legend()
ax.set_ylim(0.6, 0.8)

plt.tight_layout()
plt.savefig(output_dir / 'fig4_esn_scaling.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig4_esn_scaling.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig4_esn_scaling.png/pdf")

# ============================================================================
# FIGURE 5: QRC Architecture Comparison
# ============================================================================
print("\n[5] QRC Architecture Comparison...")

qrc_models = [
    ('Hybrid\n8q+100n', 0.637),
    ('Hybrid\n12q+100n', 0.635),
    ('Polynomial\n14q', 0.133),  # From earlier experiments
    ('Quadratic\n8q', 0.558),
    ('Quadratic\n10q', 0.557),
    ('MultiBasis\n8q', 0.519),
    ('ReEncode\n8q', 0.199),
]

fig, ax = plt.subplots(figsize=(10, 5))

names = [m[0] for m in qrc_models]
scores = [m[1] for m in qrc_models]

bars = ax.bar(names, scores, color='#9b59b6', edgecolor='white')

# Reference lines
ax.axhline(y=0.763, color='#2ecc71', linestyle='-', linewidth=2, label='ESN_500n (0.763)')
ax.axhline(y=0.666, color='#3498db', linestyle='--', label='LSTM (0.666)')
ax.axhline(y=0.645, color='gray', linestyle=':', label='Baseline (0.645)')

ax.set_ylabel('R² Score')
ax.set_title('QRC Architecture Comparison\n(All variants underperform classical methods)')
ax.legend(loc='upper right')
ax.set_ylim(0, 0.85)

# Add value labels
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.02, f'{score:.3f}', 
            ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / 'fig5_qrc_architectures.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig5_qrc_architectures.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig5_qrc_architectures.png/pdf")

# ============================================================================
# FIGURE 6: Key Findings Summary
# ============================================================================
print("\n[6] Key Findings Infographic...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'QRC vs Classical Methods: Key Findings', 
        fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
ax.text(0.5, 0.90, 'EV Charging Demand Prediction (Palo Alto 2017-2019)', 
        fontsize=12, ha='center', transform=ax.transAxes, style='italic')

# Three columns
col_x = [0.17, 0.5, 0.83]
col_titles = ['Classical (ESN)', 'Deep Learning (LSTM)', 'Quantum (QRC)']
col_colors = ['#2ecc71', '#3498db', '#9b59b6']
col_r2 = ['0.763', '0.666', '0.637']
col_time = ['1s', '31s', '1834s']
col_verdict = ['✓ WINNER', '2nd Place', '3rd Place']

for i, (x, title, color, r2, time, verdict) in enumerate(zip(
    col_x, col_titles, col_colors, col_r2, col_time, col_verdict)):
    
    # Box
    rect = plt.Rectangle((x-0.13, 0.25), 0.26, 0.55, 
                         facecolor=color, alpha=0.2, edgecolor=color, linewidth=2,
                         transform=ax.transAxes)
    ax.add_patch(rect)
    
    # Title
    ax.text(x, 0.75, title, fontsize=14, fontweight='bold', ha='center', 
            transform=ax.transAxes, color=color)
    
    # R² Score
    ax.text(x, 0.65, f'R² = {r2}', fontsize=20, fontweight='bold', ha='center',
            transform=ax.transAxes)
    
    # Training time
    ax.text(x, 0.55, f'Training: {time}', fontsize=12, ha='center',
            transform=ax.transAxes)
    
    # Verdict
    ax.text(x, 0.35, verdict, fontsize=14, ha='center', transform=ax.transAxes,
            fontweight='bold' if i == 0 else 'normal',
            color='green' if i == 0 else 'black')

# Bottom conclusions
conclusions = [
    "• ESN outperforms LSTM by 14.6% while training 30x faster",
    "• ESN outperforms best QRC by 19.8% while training 1800x faster", 
    "• All quantum variants perform worse than the weekly baseline",
    "• Quantum features add noise rather than useful signal for this task"
]

for i, conclusion in enumerate(conclusions):
    ax.text(0.1, 0.18 - i*0.05, conclusion, fontsize=11, transform=ax.transAxes)

ax.text(0.5, 0.02, 'Conclusion: For periodic time-series like EV charging, classical ESN is optimal.',
        fontsize=12, ha='center', transform=ax.transAxes, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'fig6_key_findings.png', bbox_inches='tight')
plt.savefig(output_dir / 'fig6_key_findings.pdf', bbox_inches='tight')
plt.close()
print("  ✓ fig6_key_findings.png/pdf")

print("\n" + "=" * 60)
print("✓ All figures generated in docs/figures/")
print("=" * 60)
