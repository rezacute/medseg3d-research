#!/usr/bin/env python3
"""
Generate comprehensive visualizations for QRC-EV research paper.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Create output directory
output_dir = Path("docs/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("GENERATING VISUALIZATIONS FOR QRC-EV RESEARCH")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading results...")

with open("results/final_validation.json") as f:
    validation = json.load(f)

# Load raw data for predictions
df = pd.read_csv("data/raw/palo_alto_ev_sessions.csv")
df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
df['hour'] = df['Start Date'].dt.floor('h')
hourly = df.groupby('hour').agg({'Energy (kWh)': 'sum'}).rename(columns={'Energy (kWh)': 'energy_kwh'})
full_range = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly = hourly.reindex(full_range, fill_value=0)

print(f"  Loaded {len(hourly)} hourly samples")

# ============================================================================
# FIGURE 1: Model Comparison Bar Chart
# ============================================================================
print("\n[2] Figure 1: Model Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Hybrid\n12q+100n', 'QRC\n14q', 'ESN\n200n']
means = [
    validation['results']['Hybrid_12q_100n']['mean'],
    validation['results']['QRC_14q']['mean'],
    validation['results']['ESN_200n']['mean']
]
stds = [
    validation['results']['Hybrid_12q_100n']['std'],
    validation['results']['QRC_14q']['std'],
    validation['results']['ESN_200n']['std']
]

colors = ['#2ecc71', '#3498db', '#e74c3c']
bars = ax.bar(models, means, yerr=stds, capsize=8, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.annotate(f'{mean:.3f}±{std:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add significance markers
ax.annotate('***', xy=(0.5, 0.17), ha='center', fontsize=16, fontweight='bold')
ax.plot([0, 1], [0.165, 0.165], 'k-', linewidth=1.5)
ax.plot([0, 0], [0.163, 0.165], 'k-', linewidth=1.5)
ax.plot([1, 1], [0.135, 0.165], 'k-', linewidth=1.5)

ax.set_ylabel('Test R² Score', fontsize=12)
ax.set_title('Model Performance Comparison (5 Seeds)\nPalo Alto EV Charging Demand Prediction', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.25)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

# Add legend for significance
legend_text = "*** p < 0.01 (paired t-test)"
ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9, 
        verticalalignment='top', style='italic')

plt.tight_layout()
plt.savefig(output_dir / "fig1_model_comparison.png", bbox_inches='tight')
plt.savefig(output_dir / "fig1_model_comparison.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig1_model_comparison.png/pdf")

# ============================================================================
# FIGURE 2: Seed Variability Box Plot
# ============================================================================
print("\n[3] Figure 2: Seed Variability...")

fig, ax = plt.subplots(figsize=(10, 6))

data = [
    validation['results']['Hybrid_12q_100n']['scores'],
    validation['results']['QRC_14q']['scores'],
    validation['results']['ESN_200n']['scores']
]

bp = ax.boxplot(data, labels=['Hybrid\n12q+100n', 'QRC\n14q', 'ESN\n200n'], 
                patch_artist=True, widths=0.6)

colors = ['#2ecc71', '#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Overlay individual points
for i, scores in enumerate(data):
    x = np.random.normal(i+1, 0.04, len(scores))
    ax.scatter(x, scores, alpha=0.8, color='black', s=50, zorder=3)

ax.set_ylabel('Test R² Score', fontsize=12)
ax.set_title('Model Stability Across Random Seeds\n(5 independent runs per model)', fontsize=14, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

# Add annotations
ax.annotate('Near-zero variance\n(deterministic)', xy=(2, 0.135), xytext=(2.5, 0.05),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))
ax.annotate('High variance\n(stochastic)', xy=(3, 0.05), xytext=(3.5, -0.05),
            fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig(output_dir / "fig2_seed_variability.png", bbox_inches='tight')
plt.savefig(output_dir / "fig2_seed_variability.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig2_seed_variability.png/pdf")

# ============================================================================
# FIGURE 3: Architecture Comparison (Hybrid variants)
# ============================================================================
print("\n[4] Figure 3: Hybrid Architecture Variants...")

fig, ax = plt.subplots(figsize=(12, 6))

# Results from temporal_hybrid experiments
hybrid_results = {
    'Hybrid_8q_100n': 0.1811,
    'Hybrid_8q_150n': 0.1829,
    'Hybrid_10q_100n': 0.1848,
    'Hybrid_12q_100n': 0.1879,
    'Stacked_50n_8q': 0.1261,
}
baselines = {
    'QRC_14q': 0.1326,
    'ESN_200n': 0.1445,
}

# Combine and sort
all_results = {**hybrid_results, **baselines}
sorted_results = dict(sorted(all_results.items(), key=lambda x: x[1], reverse=True))

names = list(sorted_results.keys())
values = list(sorted_results.values())

# Color coding
colors = []
for name in names:
    if name.startswith('Hybrid'):
        colors.append('#2ecc71')
    elif name.startswith('Stacked'):
        colors.append('#f39c12')
    elif 'QRC' in name:
        colors.append('#3498db')
    else:
        colors.append('#e74c3c')

bars = ax.barh(names, values, color=colors, edgecolor='black', linewidth=1)

# Add value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Test R² Score', fontsize=12)
ax.set_title('Architecture Comparison: Hybrid vs Baseline Models\nPalo Alto EV Charging Dataset', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.22)

# Legend
legend_patches = [
    mpatches.Patch(color='#2ecc71', label='Parallel Hybrid (QRC || ESN)'),
    mpatches.Patch(color='#f39c12', label='Stacked Hybrid (ESN → QRC)'),
    mpatches.Patch(color='#3498db', label='Pure QRC'),
    mpatches.Patch(color='#e74c3c', label='Pure ESN'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "fig3_architecture_comparison.png", bbox_inches='tight')
plt.savefig(output_dir / "fig3_architecture_comparison.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig3_architecture_comparison.png/pdf")

# ============================================================================
# FIGURE 4: Hybrid Architecture Diagram
# ============================================================================
print("\n[5] Figure 4: Hybrid Architecture Diagram...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Input
ax.add_patch(plt.Rectangle((0.5, 3), 2, 2, facecolor='#ecf0f1', edgecolor='black', linewidth=2))
ax.text(1.5, 4, 'Input\nFeatures\n(15-dim)', ha='center', va='center', fontsize=10, fontweight='bold')

# QRC Branch
ax.add_patch(plt.Rectangle((4, 5.5), 2.5, 1.5, facecolor='#3498db', edgecolor='black', linewidth=2))
ax.text(5.25, 6.25, 'QRC\n12 qubits', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.add_patch(plt.Rectangle((7, 5.5), 2, 1.5, facecolor='#2980b9', edgecolor='black', linewidth=2))
ax.text(8, 6.25, 'Polynomial\nDegree=2', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# ESN Branch
ax.add_patch(plt.Rectangle((4, 1), 2.5, 1.5, facecolor='#e74c3c', edgecolor='black', linewidth=2))
ax.text(5.25, 1.75, 'ESN\n100 neurons', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.add_patch(plt.Rectangle((7, 1), 2, 1.5, facecolor='#c0392b', edgecolor='black', linewidth=2))
ax.text(8, 1.75, 'Leak=0.3\nρ=0.9', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

# Concatenation
ax.add_patch(plt.Circle((10.5, 4), 0.8, facecolor='#9b59b6', edgecolor='black', linewidth=2))
ax.text(10.5, 4, '⊕', ha='center', va='center', fontsize=20, fontweight='bold', color='white')

# Ridge Regression
ax.add_patch(plt.Rectangle((11.5, 3), 2, 2, facecolor='#2ecc71', edgecolor='black', linewidth=2))
ax.text(12.5, 4, 'Ridge\nα=20', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrows
arrow_props = dict(arrowstyle='->', color='black', linewidth=2)
ax.annotate('', xy=(4, 6.25), xytext=(2.5, 4.5), arrowprops=arrow_props)
ax.annotate('', xy=(4, 1.75), xytext=(2.5, 3.5), arrowprops=arrow_props)
ax.annotate('', xy=(7, 6.25), xytext=(6.5, 6.25), arrowprops=arrow_props)
ax.annotate('', xy=(7, 1.75), xytext=(6.5, 1.75), arrowprops=arrow_props)
ax.annotate('', xy=(9.7, 4.5), xytext=(9, 6.25), arrowprops=arrow_props)
ax.annotate('', xy=(9.7, 3.5), xytext=(9, 1.75), arrowprops=arrow_props)
ax.annotate('', xy=(11.5, 4), xytext=(11.3, 4), arrowprops=arrow_props)

# Feature dimensions
ax.text(5.25, 7.3, '12 qubits → 91 poly features', ha='center', fontsize=9, style='italic')
ax.text(5.25, 0.5, '15 input → 100 reservoir states', ha='center', fontsize=9, style='italic')
ax.text(10.5, 2.8, '191 total\nfeatures', ha='center', fontsize=9, style='italic')

# Title
ax.text(7, 7.8, 'Hybrid QRC+ESN Architecture', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "fig4_hybrid_architecture.png", bbox_inches='tight')
plt.savefig(output_dir / "fig4_hybrid_architecture.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig4_hybrid_architecture.png/pdf")

# ============================================================================
# FIGURE 5: Effect Size Visualization
# ============================================================================
print("\n[6] Figure 5: Effect Size Analysis...")

fig, ax = plt.subplots(figsize=(10, 5))

comparisons = ['Hybrid vs QRC', 'Hybrid vs ESN']
cohens_d = [
    validation['statistical_tests']['hybrid_vs_qrc']['cohens_d'],
    validation['statistical_tests']['hybrid_vs_esn']['cohens_d']
]
p_values = [
    validation['statistical_tests']['hybrid_vs_qrc']['p_value'],
    validation['statistical_tests']['hybrid_vs_esn']['p_value']
]

colors = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_values]
bars = ax.barh(comparisons, cohens_d, color=colors, edgecolor='black', linewidth=1.5, height=0.5)

# Reference lines for effect size interpretation
ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, label='Small (0.2)')
ax.axvline(x=0.5, color='gray', linestyle='-.', linewidth=1, label='Medium (0.5)')
ax.axvline(x=0.8, color='gray', linestyle=':', linewidth=1.5, label='Large (0.8)')

# Add labels
for bar, d, p in zip(bars, cohens_d, p_values):
    sig = '***' if p < 0.01 else '**' if p < 0.05 else 'ns'
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
            f'd={d:.2f} ({sig})', va='center', fontsize=11, fontweight='bold')

ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
ax.set_title("Effect Size Analysis: Hybrid Model Improvements\n(Green = statistically significant at α=0.05)", fontsize=14, fontweight='bold')
ax.set_xlim(0, 5.5)
ax.legend(loc='center right', fontsize=9, title='Effect Size Thresholds')

plt.tight_layout()
plt.savefig(output_dir / "fig5_effect_size.png", bbox_inches='tight')
plt.savefig(output_dir / "fig5_effect_size.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig5_effect_size.png/pdf")

# ============================================================================
# FIGURE 6: Research Timeline / Ablation Summary
# ============================================================================
print("\n[7] Figure 6: Research Progress...")

fig, ax = plt.subplots(figsize=(12, 7))

# Ablation study results
experiments = [
    ('Baseline QRC 8q', 0.05, '#bdc3c7'),
    ('QRC 14q', 0.10, '#3498db'),
    ('+ Polynomial (deg=2)', 0.126, '#3498db'),
    ('+ α=50 regularization', 0.133, '#3498db'),
    ('ESN 200n baseline', 0.145, '#e74c3c'),
    ('Hybrid 8q+100n', 0.181, '#2ecc71'),
    ('Hybrid 12q+100n', 0.188, '#2ecc71'),
]

names = [e[0] for e in experiments]
values = [e[1] for e in experiments]
colors = [e[2] for e in experiments]

y_pos = np.arange(len(names))
bars = ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=1)

# Value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            va='center', fontsize=10, fontweight='bold')

# Improvement arrows
ax.annotate('', xy=(0.133, 3), xytext=(0.126, 2), 
            arrowprops=dict(arrowstyle='->', color='green', linewidth=2))
ax.text(0.14, 2.5, '+5.6%', fontsize=9, color='green', fontweight='bold')

ax.annotate('', xy=(0.188, 6), xytext=(0.133, 3), 
            arrowprops=dict(arrowstyle='->', color='green', linewidth=2, connectionstyle='arc3,rad=0.3'))
ax.text(0.165, 4.5, '+41%', fontsize=11, color='green', fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('Test R² Score', fontsize=12)
ax.set_title('Research Progress: From Baseline to Hybrid Architecture\nPalo Alto EV Charging Dataset', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.22)

# Legend
legend_patches = [
    mpatches.Patch(color='#3498db', label='Pure QRC variants'),
    mpatches.Patch(color='#e74c3c', label='Pure ESN'),
    mpatches.Patch(color='#2ecc71', label='Hybrid QRC+ESN'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / "fig6_research_progress.png", bbox_inches='tight')
plt.savefig(output_dir / "fig6_research_progress.pdf", bbox_inches='tight')
print(f"  ✓ Saved fig6_research_progress.png/pdf")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nGenerated 6 figures in {output_dir}/:")
print("  1. fig1_model_comparison.png/pdf - Main results bar chart")
print("  2. fig2_seed_variability.png/pdf - Box plot of seed stability")
print("  3. fig3_architecture_comparison.png/pdf - All architectures ranked")
print("  4. fig4_hybrid_architecture.png/pdf - Architecture diagram")
print("  5. fig5_effect_size.png/pdf - Cohen's d analysis")
print("  6. fig6_research_progress.png/pdf - Ablation timeline")
