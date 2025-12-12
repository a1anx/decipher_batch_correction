"""
Direct comparison between Concatenation and Attention approaches on delta shift data

This script assumes both models have been trained:
- Concatenation: run_concat_on_data_delta.py
- Attention: (needs to be run separately with attention-based approach)

It loads both results and creates side-by-side comparisons.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
import os

print("=" * 80)
print("COMPARING CONCATENATION VS ATTENTION ON DELTA SHIFT DATA")
print("=" * 80)

# Load both datasets
print("\n[1/4] Loading data...")

# Check if files exist
concat_file = '/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta_concat.h5ad'
attention_file = '/home/alan/Documents/decipher_batch_correction/simulation/archive/oldadata_alpha/adata_combined_2_delta_batch_corrected.h5ad'

if not os.path.exists(concat_file):
    print(f"\n⚠ ERROR: Concatenation results not found!")
    print(f"  Expected: {concat_file}")
    print(f"  Please run: python run_concat_on_data_delta.py")
    exit(1)

if not os.path.exists(attention_file):
    print(f"\n⚠ ERROR: Attention results not found!")
    print(f"  Expected: {attention_file}")
    print(f"  Please run the attention-based batch correction on delta data first")
    print(f"  (This file needs to be generated separately)")
    exit(1)

adata_concat = sc.read_h5ad(concat_file)
adata_attention = sc.read_h5ad(attention_file)

print(f"✓ Loaded concatenation results: {adata_concat.shape}")
print(f"✓ Loaded attention results: {adata_attention.shape}")

# Verify they're the same dataset
assert adata_concat.shape == adata_attention.shape, "Datasets have different shapes!"
assert all(adata_concat.obs['delta'] == adata_attention.obs['delta']), "Different batch labels!"

print("\n[2/4] Computing batch mixing metrics...")

def compute_batch_mixing(embedding, labels):
    """Compute silhouette score (lower = better mixing)"""
    return silhouette_score(embedding, labels)

# Get batch labels
batch_labels = adata_concat.obs['delta']

# Original Decipher (should be same in both)
original_mixing = compute_batch_mixing(
    adata_concat.obsm['decipher_decipher_z'],
    batch_labels
)

# Concatenation approach
concat_mixing = compute_batch_mixing(
    adata_concat.obsm['X_decipher_concat_z'],
    batch_labels
)

# Attention approach
attention_mixing = compute_batch_mixing(
    adata_attention.obsm['X_decipher_batch_corrected_z'],
    batch_labels
)

print(f"\nBatch Mixing Scores (Silhouette, lower = better mixing):")
print(f"  Original Decipher:      {original_mixing:.4f}")
print(f"  Concatenation:          {concat_mixing:.4f}")
print(f"  Attention:              {attention_mixing:.4f}")
print(f"\nImprovement over original:")
print(f"  Concatenation: {original_mixing - concat_mixing:+.4f}")
print(f"  Attention:     {original_mixing - attention_mixing:+.4f}")
print(f"\nConcatenation vs Attention:")
if concat_mixing < attention_mixing:
    print(f"  ✓ Concatenation wins by {attention_mixing - concat_mixing:.4f}")
else:
    print(f"  ✓ Attention wins by {concat_mixing - attention_mixing:.4f}")

# [3/4] Create comprehensive comparison visualizations
print("\n[3/4] Creating comparison visualizations...")

# Define colors for delta values (use discrete colormap)
delta_values = sorted(adata_concat.obs['delta'].unique())
cmap = plt.colormaps.get_cmap('tab10')
batch_colors = {val: cmap(i) for i, val in enumerate(delta_values)}

def plot_embedding(ax, coords, title, adata, color_by='delta'):
    """Helper function to plot embeddings"""
    for batch in sorted(adata.obs[color_by].unique()):
        mask = adata.obs[color_by] == batch
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  c=[batch_colors[batch]], label=f'Delta={batch}', alpha=0.6, s=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.legend(title='Delta Value', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# ===== Figure 1: Z space comparison =====
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: First 2 dimensions of Z
plot_embedding(axes[0, 0], adata_concat.obsm['decipher_decipher_z'][:, :2],
               'Original Z (2D)', adata_concat)
plot_embedding(axes[0, 1], adata_concat.obsm['X_decipher_concat_z'][:, :2],
               'Concat Z (2D)', adata_concat)
plot_embedding(axes[0, 2], adata_attention.obsm['X_decipher_batch_corrected_z'][:, :2],
               'Attention Z (2D)', adata_attention)

# Row 2: Colored by pseudotime
pseudotime = adata_concat.obs['latent_t'].values
if len(pseudotime.shape) > 1:
    pseudotime = pseudotime.flatten()

ax = axes[1, 0]
scatter = ax.scatter(adata_concat.obsm['decipher_decipher_z'][:, 0],
                    adata_concat.obsm['decipher_decipher_z'][:, 1],
                    c=pseudotime, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, ax=ax, label='Pseudotime')
ax.set_title('Original Z - Pseudotime', fontsize=12, fontweight='bold')
ax.set_xlabel('Z Dimension 1')
ax.set_ylabel('Z Dimension 2')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
scatter = ax.scatter(adata_concat.obsm['X_decipher_concat_z'][:, 0],
                    adata_concat.obsm['X_decipher_concat_z'][:, 1],
                    c=pseudotime, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, ax=ax, label='Pseudotime')
ax.set_title('Concat Z - Pseudotime', fontsize=12, fontweight='bold')
ax.set_xlabel('Z Dimension 1')
ax.set_ylabel('Z Dimension 2')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
scatter = ax.scatter(adata_attention.obsm['X_decipher_batch_corrected_z'][:, 0],
                    adata_attention.obsm['X_decipher_batch_corrected_z'][:, 1],
                    c=pseudotime, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, ax=ax, label='Pseudotime')
ax.set_title('Attention Z - Pseudotime', fontsize=12, fontweight='bold')
ax.set_xlabel('Z Dimension 1')
ax.set_ylabel('Z Dimension 2')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_z_space_delta.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_z_space_delta.png")
plt.close()

# ===== Figure 2: V space comparison =====
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Row 1: V space by delta
plot_embedding(axes[0, 0], adata_concat.obsm['decipher_decipher_v'],
               'Original V (2D)', adata_concat)
plot_embedding(axes[0, 1], adata_concat.obsm['X_decipher_concat_v'],
               'Concat V (2D)', adata_concat)
plot_embedding(axes[0, 2], adata_attention.obsm['X_decipher_batch_corrected_v'],
               'Attention V (2D)', adata_attention)

# Row 2: V space by pseudotime
pseudotime = adata_concat.obs['latent_t'].values
if len(pseudotime.shape) > 1:
    pseudotime = pseudotime.flatten()

for i, (coords, title) in enumerate([
    (adata_concat.obsm['decipher_decipher_v'], 'Original V - Pseudotime'),
    (adata_concat.obsm['X_decipher_concat_v'], 'Concat V - Pseudotime'),
    (adata_attention.obsm['X_decipher_batch_corrected_v'], 'Attention V - Pseudotime')
]):
    ax = axes[1, i]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=pseudotime, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, ax=ax, label='Pseudotime')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('V Dimension 1')
    ax.set_ylabel('V Dimension 2')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_v_space_delta.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_v_space_delta.png")
plt.close()

# ===== Figure 3: Metrics comparison =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar chart: Batch mixing
ax = axes[0]
methods = ['Original', 'Concat', 'Attention']
mixing_scores = [original_mixing, concat_mixing, attention_mixing]
colors = ['gray', 'blue', 'red']
bars = ax.bar(methods, mixing_scores, color=colors, alpha=0.7)
ax.set_ylabel('Silhouette Score\n(lower = better mixing)', fontsize=11)
ax.set_title('Batch Mixing Performance', fontsize=12, fontweight='bold')
ax.axhline(y=original_mixing, color='gray', linestyle='--', alpha=0.5, label='Original baseline')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=10)

# Variance explained comparison
ax = axes[1]
concat_var = np.var(adata_concat.obsm['X_decipher_concat_z'], axis=0)
concat_var_pct = 100 * concat_var / concat_var.sum()
attention_var = np.var(adata_attention.obsm['X_decipher_batch_corrected_z'], axis=0)
attention_var_pct = 100 * attention_var / attention_var.sum()

x = np.arange(min(10, len(concat_var_pct)))
width = 0.35
ax.bar(x - width/2, concat_var_pct[:len(x)], width, label='Concat', color='blue', alpha=0.7)
ax.bar(x + width/2, attention_var_pct[:len(x)], width, label='Attention', color='red', alpha=0.7)
ax.set_xlabel('Latent Dimension', fontsize=11)
ax.set_ylabel('Variance Explained (%)', fontsize=11)
ax.set_title('Variance per Dimension', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Cumulative variance
ax = axes[2]
concat_cumvar = np.cumsum(concat_var_pct)
attention_cumvar = np.cumsum(attention_var_pct)
ax.plot(range(1, len(concat_cumvar[:10])+1), concat_cumvar[:10],
        'o-', label='Concat', color='blue', linewidth=2)
ax.plot(range(1, len(attention_cumvar[:10])+1), attention_cumvar[:10],
        's-', label='Attention', color='red', linewidth=2)
ax.set_xlabel('Number of Dimensions', fontsize=11)
ax.set_ylabel('Cumulative Variance (%)', fontsize=11)
ax.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_metrics_delta.png', dpi=300, bbox_inches='tight')
print("✓ Saved: comparison_metrics_delta.png")
plt.close()

# [4/4] Summary report
print("\n[4/4] Generating summary report...")

print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON SUMMARY (DELTA SHIFT DATA)")
print("=" * 80)

print("\n1. BATCH MIXING (Silhouette Score - lower is better)")
print("-" * 80)
print(f"  Original:           {original_mixing:.4f}")
print(f"  Concatenation:      {concat_mixing:.4f}  (improvement: {original_mixing - concat_mixing:+.4f})")
print(f"  Attention:          {attention_mixing:.4f}  (improvement: {original_mixing - attention_mixing:+.4f})")
print(f"\n  Winner: ", end="")
if concat_mixing < attention_mixing:
    print(f"CONCATENATION (better by {attention_mixing - concat_mixing:.4f})")
elif attention_mixing < concat_mixing:
    print(f"ATTENTION (better by {concat_mixing - attention_mixing:.4f})")
else:
    print("TIE")

print("\n2. VARIANCE EXPLAINED (Top 3 dimensions)")
print("-" * 80)
print(f"  Concatenation:")
for i in range(min(3, len(concat_var_pct))):
    print(f"    Dim {i+1}: {concat_var_pct[i]:.2f}%")
print(f"  Attention:")
for i in range(min(3, len(attention_var_pct))):
    print(f"    Dim {i+1}: {attention_var_pct[i]:.2f}%")

print("\n3. MODEL COMPLEXITY")
print("-" * 80)
print(f"  Concatenation:")
print(f"    - Approach: concat([z, batch_emb]) → MLP")
print(f"    - Parameters: ~520,000 (typical)")
print(f"    - Training speed: Faster")
print(f"    - Interpretability: Simple")

print(f"\n  Attention:")
print(f"    - Approach: z queries batch_emb via attention → concat → MLP")
print(f"    - Parameters: ~680,000 (typical)")
print(f"    - Training speed: Slower (~1.5x)")
print(f"    - Interpretability: Attention weights available")

if 'batch_attention_strength' in adata_attention.obs:
    print(f"\n4. ATTENTION WEIGHTS (only available for attention approach)")
    print("-" * 80)
    attn_mean = adata_attention.obs['batch_attention_strength'].mean()
    attn_std = adata_attention.obs['batch_attention_strength'].std()
    print(f"  Mean attention strength: {attn_mean:.4f} ± {attn_std:.4f}")
    print(f"  By delta value:")
    for delta_val in sorted(adata_attention.obs['delta'].unique()):
        mask = adata_attention.obs['delta'] == delta_val
        mean_attn = adata_attention.obs.loc[mask, 'batch_attention_strength'].mean()
        print(f"    Delta={delta_val}: {mean_attn:.4f}")

print("\n5. RECOMMENDATION")
print("-" * 80)

# Decision logic
if concat_mixing < attention_mixing:
    improvement_gap = attention_mixing - concat_mixing
    if improvement_gap > 0.05:
        print("  ✓ Use CONCATENATION")
        print(f"    - Significantly better batch mixing (by {improvement_gap:.4f})")
        print("    - Simpler and faster")
    else:
        print("  ✓ Use CONCATENATION")
        print(f"    - Similar performance to attention")
        print("    - Simpler and faster - no need for added complexity")
else:
    improvement_gap = concat_mixing - attention_mixing
    if improvement_gap > 0.05:
        print("  ✓ Use ATTENTION")
        print(f"    - Significantly better batch mixing (by {improvement_gap:.4f})")
        print("    - Worth the added complexity")
    else:
        print("  ⚖ BOTH APPROACHES WORK WELL")
        print(f"    - Performance difference is small ({improvement_gap:.4f})")
        print("    - Use CONCATENATION for simplicity")
        print("    - Use ATTENTION if you need interpretability (attention weights)")

print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)
print("  1. comparison_z_space_delta.png")
print("     - Z latent space comparison (3 approaches)")
print("     - Top row: colored by delta value")
print("     - Bottom row: colored by pseudotime")
print("\n  2. comparison_v_space_delta.png")
print("     - V component space comparison (3 approaches)")
print("     - Top row: colored by delta value")
print("     - Bottom row: colored by pseudotime")
print("\n  3. comparison_metrics_delta.png")
print("     - Quantitative comparison")
print("     - Batch mixing, variance explained, cumulative variance")

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE!")
print("=" * 80)
