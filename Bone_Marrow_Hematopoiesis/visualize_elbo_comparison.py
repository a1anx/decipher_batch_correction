"""
Visualize ELBO component comparison across experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('elbo_comparison_results.csv')

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# Define colors for each experiment
colors = {
    'Full_Training (4 heads, β=0.1)': '#1f77b4',
    'AttentionHeads_2 (β=0.1)': '#ff7f0e',
    'Beta_1.0 (4 heads)': '#2ca02c',
    'AttentionHeads_1 (β=0.1)': '#d62728',
}

# Short names for plotting
short_names = {
    'Full_Training (4 heads, β=0.1)': '4 heads\nβ=0.1',
    'AttentionHeads_2 (β=0.1)': '2 heads\nβ=0.1',
    'Beta_1.0 (4 heads)': '4 heads\nβ=1.0',
    'AttentionHeads_1 (β=0.1)': '1 head\nβ=0.1',
}

df['Short Name'] = df['Experiment'].map(short_names)
df['Color'] = df['Experiment'].map(colors)

# 1. ELBO Component Stacked Bar Chart
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(df))
width = 0.6

# Stacked bars
reconstruction = df['Recon. Est.']
kl_z = df['β·KL_z']
kl_v = df['β·KL_v']

# Plot from bottom to top
bars1 = ax1.bar(x, reconstruction, width, label='Reconstruction', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x, kl_z, width, bottom=reconstruction, label='β·KL_z', color='#e74c3c', alpha=0.8)
bars3 = ax1.bar(x, kl_v, width, bottom=reconstruction + kl_z, label='β·KL_v', color='#f39c12', alpha=0.8)

ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax1.set_title('ELBO Components (Stacked)', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Short Name'], fontsize=9)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add total loss text on top
for i, (idx, row) in enumerate(df.iterrows()):
    total = row['Total Loss']
    ax1.text(i, total + 50, f"{total:,.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

# 2. KL Contributions (separated)
ax2 = plt.subplot(2, 3, 2)
x_offset = np.arange(len(df))
bar_width = 0.35

bars_z = ax2.bar(x_offset - bar_width/2, df['β·KL_z'], bar_width, label='β·KL_z', color='#e74c3c', alpha=0.8)
bars_v = ax2.bar(x_offset + bar_width/2, df['β·KL_v'], bar_width, label='β·KL_v', color='#f39c12', alpha=0.8)

ax2.set_ylabel('KL Contribution', fontsize=11, fontweight='bold')
ax2.set_title('KL Divergence Contributions', fontsize=12, fontweight='bold')
ax2.set_xticks(x_offset)
ax2.set_xticklabels(df['Short Name'], fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (z_val, v_val) in enumerate(zip(df['β·KL_z'], df['β·KL_v'])):
    ax2.text(i - bar_width/2, z_val + 0.5, f"{z_val:.0f}", ha='center', va='bottom', fontsize=8)
    ax2.text(i + bar_width/2, v_val + 0.05, f"{v_val:.1f}", ha='center', va='bottom', fontsize=8)

# 3. V-space Silhouette Score
ax3 = plt.subplot(2, 3, 3)
bars = ax3.barh(df['Short Name'], df['V Silhouette'], color=[colors[exp] for exp in df['Experiment']], alpha=0.8)

# Add reference line for regular Decipher
regular_decipher_silh = -0.2840
ax3.axvline(regular_decipher_silh, color='green', linestyle='--', linewidth=2, label='Regular Decipher\n(-0.2840)', alpha=0.7)

ax3.set_xlabel('V-space Silhouette Score', fontsize=11, fontweight='bold')
ax3.set_title('V-space Structure Quality', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='lower right')
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(df.iterrows()):
    silh = row['V Silhouette']
    ax3.text(silh - 0.003, i, f"{silh:.4f}", ha='right', va='center', fontsize=9, fontweight='bold')

# Highlight best (closest to -0.2840)
best_idx = (df['V Silhouette'] - regular_decipher_silh).abs().idxmin()
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)

# 4. Embedding Norms
ax4 = plt.subplot(2, 3, 4)
x_offset = np.arange(len(df))
bar_width = 0.35

bars_z_norm = ax4.bar(x_offset - bar_width/2, df['z_norm_mean'], bar_width, label='z norm', color='#9b59b6', alpha=0.8)
bars_v_norm = ax4.bar(x_offset + bar_width/2, df['v_norm_mean'], bar_width, label='v norm', color='#1abc9c', alpha=0.8)

ax4.set_ylabel('Mean Embedding Norm', fontsize=11, fontweight='bold')
ax4.set_title('Embedding Magnitudes', fontsize=12, fontweight='bold')
ax4.set_xticks(x_offset)
ax4.set_xticklabels(df['Short Name'], fontsize=9)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (z_norm, v_norm) in enumerate(zip(df['z_norm_mean'], df['v_norm_mean'])):
    ax4.text(i - bar_width/2, z_norm + 0.15, f"{z_norm:.2f}", ha='center', va='bottom', fontsize=8)
    ax4.text(i + bar_width/2, v_norm + 0.04, f"{v_norm:.2f}", ha='center', va='bottom', fontsize=8)

# 5. Total Loss comparison
ax5 = plt.subplot(2, 3, 5)
bars = ax5.bar(df['Short Name'], df['Total Loss'], color=[colors[exp] for exp in df['Experiment']], alpha=0.8)

ax5.set_ylabel('Total Loss', fontsize=11, fontweight='bold')
ax5.set_title('Total Validation Loss', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.set_ylim([160500, 161200])

# Add value labels
for i, (idx, row) in enumerate(df.iterrows()):
    loss = row['Total Loss']
    ax5.text(i, loss + 15, f"{loss:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Find min loss
min_loss_idx = df['Total Loss'].idxmin()
bars[min_loss_idx].set_edgecolor('blue')
bars[min_loss_idx].set_linewidth(3)

# 6. Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary text
summary_text = """
KEY FINDINGS:

Beta = 1.0 (4 heads):
  ✓ BEST V-space structure (-0.2722)
  ✓ Closest to regular Decipher (-0.2840)
  ✓ Smallest z/v embedding norms (6.36, 1.40)
  ✓ Higher KL contributions (22 + 1)
  • Total loss: 160,962 (middle)

AttentionHeads_1 (β=0.1):
  ✓ LOWEST total loss (160,711)
  ✗ WORST V-space structure (-0.3324)
  ✗ Most blob-like trajectory

Effect of β:
  β=0.1 → β=1.0:
  • β·KL_z: 3 → 22 (7× increase)
  • β·KL_v: 0.2 → 1.1 (5× increase)
  • V-space: -0.3168 → -0.2722 (14% better!)
  • Loss: +38 (only +0.02%)

Effect of Attention Heads (β=0.1):
  4 → 2 heads: V-space improves slightly
  2 → 1 head:  V-space worsens dramatically

CONCLUSION: Use Beta=1.0
  Strong regularization preserves biology!
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=9, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('ELBO Component Comparison Across Batch-Corrected Decipher Models',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('elbo_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: elbo_comparison_visualization.png")
plt.close()

# Create a second figure focusing on the loss breakdown
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Loss breakdown with actual values
ax_left = axes[0]
components = ['Reconstruction', 'β·KL_z', 'β·KL_v']
x = np.arange(len(df))
width = 0.2

for i, component in enumerate(components):
    if component == 'Reconstruction':
        values = df['Recon. Est.']
    elif component == 'β·KL_z':
        values = df['β·KL_z']
    else:
        values = df['β·KL_v']

    offset = (i - 1) * width
    bars = ax_left.bar(x + offset, values, width, label=component, alpha=0.8)

ax_left.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax_left.set_title('Loss Components (Separated)', fontsize=13, fontweight='bold')
ax_left.set_xticks(x)
ax_left.set_xticklabels(df['Short Name'], fontsize=10)
ax_left.legend(fontsize=10)
ax_left.grid(axis='y', alpha=0.3)

# Right: Percentage contribution
ax_right = axes[1]
percentages = []
for idx, row in df.iterrows():
    total = row['Total Loss']
    recon_pct = (row['Recon. Est.'] / total) * 100
    kl_z_pct = (row['β·KL_z'] / total) * 100
    kl_v_pct = (row['β·KL_v'] / total) * 100
    percentages.append([recon_pct, kl_z_pct, kl_v_pct])

percentages = np.array(percentages)

# Stacked percentage bar
bottom = np.zeros(len(df))
labels = ['Reconstruction', 'β·KL_z', 'β·KL_v']
colors_comp = ['#3498db', '#e74c3c', '#f39c12']

for i, (label, color) in enumerate(zip(labels, colors_comp)):
    ax_right.bar(df['Short Name'], percentages[:, i], bottom=bottom, label=label, color=color, alpha=0.8)

    # Add percentage labels
    for j, pct in enumerate(percentages[:, i]):
        if pct > 0.05:  # Only show if >0.05%
            ax_right.text(j, bottom[j] + pct/2, f'{pct:.2f}%', ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')

    bottom += percentages[:, i]

ax_right.set_ylabel('Percentage of Total Loss (%)', fontsize=12, fontweight='bold')
ax_right.set_title('Loss Components (Percentage)', fontsize=13, fontweight='bold')
ax_right.legend(fontsize=10)
ax_right.grid(axis='y', alpha=0.3)

plt.suptitle('ELBO Loss Breakdown Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('elbo_breakdown_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved detailed breakdown to: elbo_breakdown_detailed.png")
plt.close()

print("\n✓ All visualizations created successfully!")
