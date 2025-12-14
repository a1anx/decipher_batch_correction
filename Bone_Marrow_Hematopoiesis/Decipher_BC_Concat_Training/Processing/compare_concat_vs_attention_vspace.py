"""
Compare Concatenation vs Attention-Based Batch Correction in V Space

This script compares:
1. Concatenation-based batch correction (beta=1.0)
2. Attention-based batch correction (beta=1.0, 2 attention heads)

Focuses on:
- Donor mixing in v space (batch correction quality)
- Biological structure preservation (cell type separation)
- Pseudotime trajectories in v space
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr

print("=" * 80)
print("COMPARISON: Concatenation vs Attention-Based Batch Correction")
print("=" * 80)

# Load data
print("\n[1/5] Loading data...")

print("  Loading concatenation results (beta=1.0)...")
adata_concat = sc.read_h5ad('BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad')

print("  Loading attention-based results (beta=1.0, 2 heads)...")
adata_attention = sc.read_h5ad('../Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad')

print("  Loading native Decipher for reference...")
adata_native = sc.read_h5ad('../bonemarrowmap_small_regular_decipher.h5ad')

print(f"✓ Loaded all datasets: {adata_concat.shape[0]:,} cells")

# Check available embeddings
print("\nAvailable embeddings:")
print(f"  Concat: {[k for k in adata_concat.obsm.keys() if 'decipher' in k.lower()]}")
print(f"  Attention: {[k for k in adata_attention.obsm.keys() if 'decipher' in k.lower()]}")
print(f"  Native: {[k for k in adata_native.obsm.keys() if 'decipher' in k.lower()]}")

# Extract v embeddings
v_concat = adata_concat.obsm['X_decipher_concat_v']
v_attention = adata_attention.obsm['X_decipher_batch_corrected_v']
v_native = adata_native.obsm['X_decipher_regular_v']

# Get metadata
donors = adata_concat.obs['Donor']
cell_type_cols = [col for col in adata_concat.obs.columns
                  if 'celltype' in col.lower() or 'cell_type' in col.lower()]
if cell_type_cols:
    cell_type_key = cell_type_cols[0]
    cell_types = adata_concat.obs[cell_type_key]
    print(f"\n  Cell type column: '{cell_type_key}' ({cell_types.nunique()} types)")
else:
    cell_type_key = None
    print("\n  No cell type column found")

# Compute pseudotime for attention-based if not present
print("\n[2/5] Computing pseudotime on v space...")

def compute_dpt_on_v(adata, v_key, suffix='method'):
    """Compute diffusion pseudotime on v space"""
    print(f"  Computing DPT for {suffix}...")

    # Create temporary adata with v as .X
    adata_v = sc.AnnData(X=adata.obsm[v_key].copy(), obs=adata.obs.copy())

    # Compute neighbors on v space
    sc.pp.neighbors(adata_v, n_neighbors=30, use_rep='X')

    # Compute diffusion map
    sc.tl.diffmap(adata_v)

    # Find HSC cells for root
    if cell_type_key:
        progenitor_keywords = ['HSC', 'HSPC', 'Stem', 'Progenitor', 'prog', 'stem']
        cell_types_str = adata.obs[cell_type_key].astype(str)

        root_idx = None
        for keyword in progenitor_keywords:
            matches = cell_types_str.str.contains(keyword, case=False, na=False)
            if matches.any():
                candidate_cells = np.where(matches)[0]
                dc1_vals = adata_v.obsm['X_diffmap'][candidate_cells, 0]
                median_idx = candidate_cells[np.argsort(dc1_vals)[len(dc1_vals)//2]]
                root_idx = median_idx
                print(f"    Root: {cell_types_str.iloc[root_idx]} (cell {root_idx})")
                break

        if root_idx is None:
            root_idx = np.argmin(adata_v.obsm['X_diffmap'][:, 0])
            print(f"    Root: diffusion map minimum (cell {root_idx})")
    else:
        root_idx = np.argmin(adata_v.obsm['X_diffmap'][:, 0])
        print(f"    Root: diffusion map minimum (cell {root_idx})")

    adata_v.uns['iroot'] = root_idx
    sc.tl.dpt(adata_v)

    adata.obs[f'dpt_{suffix}'] = adata_v.obs['dpt_pseudotime'].values
    print(f"    Range: [{adata.obs[f'dpt_{suffix}'].min():.3f}, {adata.obs[f'dpt_{suffix}'].max():.3f}]")

    return adata

# Compute DPT for all methods
if 'dpt_concat' not in adata_concat.obs.columns:
    adata_concat = compute_dpt_on_v(adata_concat, 'X_decipher_concat_v', 'concat')
else:
    print("  Using existing DPT for concat")

if 'dpt_attention' not in adata_attention.obs.columns:
    adata_attention = compute_dpt_on_v(adata_attention, 'X_decipher_batch_corrected_v', 'attention')
else:
    print("  Using existing DPT for attention")

if 'dpt_native' not in adata_native.obs.columns:
    adata_native = compute_dpt_on_v(adata_native, 'X_decipher_regular_v', 'native')
else:
    print("  Using existing DPT for native")

dpt_concat = adata_concat.obs['dpt_concat']
dpt_attention = adata_attention.obs['dpt_attention']
dpt_native = adata_native.obs['dpt_native']

print("✓ Pseudotime computed for all methods")

# Compute batch mixing metrics
print("\n[3/5] Computing batch mixing metrics...")

def compute_batch_mixing_score(embedding, batch_labels, n_neighbors=30):
    """
    Compute batch mixing score using k-NN graph.
    Higher score = better mixing
    """
    from sklearn.neighbors import NearestNeighbors

    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    # For each cell, compute fraction of neighbors from different batches
    batch_array = np.array(batch_labels)
    mixing_scores = []

    for i, neighbor_indices in enumerate(indices):
        cell_batch = batch_array[i]
        neighbor_batches = batch_array[neighbor_indices[1:]]  # Exclude self
        different_batch = (neighbor_batches != cell_batch).sum()
        mixing_scores.append(different_batch / n_neighbors)

    return np.mean(mixing_scores)

# Encode donors as integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
donor_encoded = le.fit_transform(donors)

mixing_native = compute_batch_mixing_score(v_native, donor_encoded)
mixing_concat = compute_batch_mixing_score(v_concat, donor_encoded)
mixing_attention = compute_batch_mixing_score(v_attention, donor_encoded)

print(f"\nBatch mixing scores (higher = better mixing):")
print(f"  Native Decipher:     {mixing_native:.3f}")
print(f"  Concatenation:       {mixing_concat:.3f}")
print(f"  Attention (2 heads): {mixing_attention:.3f}")

# Compute biological structure preservation (silhouette score on cell types)
if cell_type_key:
    print("\n[4/5] Computing biological structure preservation...")

    # Encode cell types
    celltype_encoded = le.fit_transform(cell_types)

    # Compute silhouette scores (higher = better separation)
    sil_native = silhouette_score(v_native, celltype_encoded, sample_size=5000)
    sil_concat = silhouette_score(v_concat, celltype_encoded, sample_size=5000)
    sil_attention = silhouette_score(v_attention, celltype_encoded, sample_size=5000)

    print(f"\nCell type silhouette scores (higher = better separation):")
    print(f"  Native Decipher:     {sil_native:.3f}")
    print(f"  Concatenation:       {sil_concat:.3f}")
    print(f"  Attention (2 heads): {sil_attention:.3f}")

# Create comprehensive comparison
print("\n[5/5] Creating comparison visualizations...")

fig = plt.figure(figsize=(24, 20))
gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.35)

# Row 1: V space colored by pseudotime
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Row 2: V space colored by cell type
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Row 3: V space colored by donor (batch mixing)
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

# Row 4: Metrics comparison
ax10 = fig.add_subplot(gs[3, 0])
ax11 = fig.add_subplot(gs[3, 1])
ax12 = fig.add_subplot(gs[3, 2])

# Row 5: Pseudotime analysis
ax13 = fig.add_subplot(gs[4, 0])
ax14 = fig.add_subplot(gs[4, 1])
ax15 = fig.add_subplot(gs[4, 2])

# Plot Row 1: Pseudotime
print("  Plotting v space with pseudotime...")
scatter1 = ax1.scatter(v_native[:, 0], v_native[:, 1], c=dpt_native,
                       cmap='viridis', s=3, alpha=0.6, rasterized=True)
ax1.set_xlabel('V1', fontsize=11)
ax1.set_ylabel('V2', fontsize=11)
ax1.set_title('Native Decipher\n(No Batch Correction)', fontsize=12, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Pseudotime')

scatter2 = ax2.scatter(v_concat[:, 0], v_concat[:, 1], c=dpt_concat,
                       cmap='viridis', s=3, alpha=0.6, rasterized=True)
ax2.set_xlabel('V1', fontsize=11)
ax2.set_ylabel('V2', fontsize=11)
ax2.set_title('Concatenation\n(beta=1.0)', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Pseudotime')

scatter3 = ax3.scatter(v_attention[:, 0], v_attention[:, 1], c=dpt_attention,
                       cmap='viridis', s=3, alpha=0.6, rasterized=True)
ax3.set_xlabel('V1', fontsize=11)
ax3.set_ylabel('V2', fontsize=11)
ax3.set_title('Attention (2 heads)\n(beta=1.0)', fontsize=12, fontweight='bold')
plt.colorbar(scatter3, ax=ax3, label='Pseudotime')

# Plot Row 2: Cell types
if cell_type_key:
    print("  Plotting v space with cell types...")

    # Get top 15 cell types by frequency for cleaner visualization
    top_types = cell_types.value_counts().head(15).index
    mask_top = cell_types.isin(top_types)

    unique_types = top_types
    n_types = len(unique_types)
    colors = plt.cm.tab20(np.linspace(0, 1, n_types))
    type_to_color = dict(zip(unique_types, colors))

    # Plot native
    for cell_type in unique_types:
        mask = (cell_types == cell_type) & mask_top
        ax4.scatter(v_native[mask, 0], v_native[mask, 1],
                   c=[type_to_color[cell_type]], label=cell_type[:20],
                   s=3, alpha=0.6, rasterized=True)
    ax4.scatter(v_native[~mask_top, 0], v_native[~mask_top, 1],
               c='lightgray', label='Other types', s=2, alpha=0.3, rasterized=True)
    ax4.set_xlabel('V1', fontsize=11)
    ax4.set_ylabel('V2', fontsize=11)
    ax4.set_title('Native (Cell Types)', fontsize=12, fontweight='bold')

    # Plot concat
    for cell_type in unique_types:
        mask = (cell_types == cell_type) & mask_top
        ax5.scatter(v_concat[mask, 0], v_concat[mask, 1],
                   c=[type_to_color[cell_type]], label=cell_type[:20],
                   s=3, alpha=0.6, rasterized=True)
    ax5.scatter(v_concat[~mask_top, 0], v_concat[~mask_top, 1],
               c='lightgray', label='Other types', s=2, alpha=0.3, rasterized=True)
    ax5.set_xlabel('V1', fontsize=11)
    ax5.set_ylabel('V2', fontsize=11)
    ax5.set_title('Concatenation (Cell Types)', fontsize=12, fontweight='bold')

    # Plot attention
    for cell_type in unique_types:
        mask = (cell_types == cell_type) & mask_top
        ax6.scatter(v_attention[mask, 0], v_attention[mask, 1],
                   c=[type_to_color[cell_type]], label=cell_type[:20],
                   s=3, alpha=0.6, rasterized=True)
    ax6.scatter(v_attention[~mask_top, 0], v_attention[~mask_top, 1],
               c='lightgray', label='Other types', s=2, alpha=0.3, rasterized=True)
    ax6.set_xlabel('V1', fontsize=11)
    ax6.set_ylabel('V2', fontsize=11)
    ax6.set_title('Attention (Cell Types)', fontsize=12, fontweight='bold')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=2)

# Plot Row 3: Donors (batch mixing)
print("  Plotting v space with donors...")

# Show top 10 donors
top_donors = donors.value_counts().head(10).index
donor_colors = plt.cm.tab20(np.linspace(0, 1, 10))

for ax, v_data, title in [(ax7, v_native, 'Native (Donors)'),
                           (ax8, v_concat, 'Concat (Donors)'),
                           (ax9, v_attention, 'Attention (Donors)')]:
    for i, donor in enumerate(top_donors):
        mask = donors == donor
        ax.scatter(v_data[mask, 0], v_data[mask, 1],
                  c=[donor_colors[i]], label=donor[:15],
                  s=3, alpha=0.6, rasterized=True)

    mask_other = ~donors.isin(top_donors)
    ax.scatter(v_data[mask_other, 0], v_data[mask_other, 1],
              c='lightgray', label='Other donors',
              s=2, alpha=0.3, rasterized=True)

    ax.set_xlabel('V1', fontsize=11)
    ax.set_ylabel('V2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=2)

# Plot Row 4: Metrics comparison
print("  Plotting metrics comparison...")

# Bar plot: Batch mixing
methods = ['Native', 'Concat', 'Attention']
mixing_scores = [mixing_native, mixing_concat, mixing_attention]
bars1 = ax10.bar(methods, mixing_scores, color=['gray', 'orange', 'blue'], alpha=0.7, edgecolor='black')
ax10.set_ylabel('Batch Mixing Score', fontsize=11)
ax10.set_title('Batch Mixing Quality\n(Higher = Better)', fontsize=12, fontweight='bold')
ax10.set_ylim([0, max(mixing_scores) * 1.2])
ax10.grid(True, alpha=0.3, axis='y')
# Add values on bars
for bar, score in zip(bars1, mixing_scores):
    height = bar.get_height()
    ax10.text(bar.get_x() + bar.get_width()/2., height,
             f'{score:.3f}', ha='center', va='bottom', fontsize=10)

# Bar plot: Biological structure preservation
if cell_type_key:
    sil_scores = [sil_native, sil_concat, sil_attention]
    bars2 = ax11.bar(methods, sil_scores, color=['gray', 'orange', 'blue'], alpha=0.7, edgecolor='black')
    ax11.set_ylabel('Silhouette Score', fontsize=11)
    ax11.set_title('Cell Type Separation\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax11.set_ylim([min(sil_scores) * 0.9, max(sil_scores) * 1.1])
    ax11.grid(True, alpha=0.3, axis='y')
    for bar, score in zip(bars2, sil_scores):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=10)

# Variance explained
v_vars = [np.var(v_native, axis=0).sum(),
          np.var(v_concat, axis=0).sum(),
          np.var(v_attention, axis=0).sum()]
bars3 = ax12.bar(methods, v_vars, color=['gray', 'orange', 'blue'], alpha=0.7, edgecolor='black')
ax12.set_ylabel('Total Variance', fontsize=11)
ax12.set_title('V Space Variance\n(Information Content)', fontsize=12, fontweight='bold')
ax12.grid(True, alpha=0.3, axis='y')
for bar, var in zip(bars3, v_vars):
    height = bar.get_height()
    ax12.text(bar.get_x() + bar.get_width()/2., height,
             f'{var:.2f}', ha='center', va='bottom', fontsize=10)

# Plot Row 5: Pseudotime analysis
print("  Plotting pseudotime analysis...")

# Histogram comparison
ax13.hist([dpt_native, dpt_concat, dpt_attention], bins=40, alpha=0.6,
          label=['Native', 'Concat', 'Attention'],
          color=['gray', 'orange', 'blue'], edgecolor='black')
ax13.set_xlabel('Pseudotime (DPT)', fontsize=11)
ax13.set_ylabel('Cell Count', fontsize=11)
ax13.set_title('Pseudotime Distribution', fontsize=12, fontweight='bold')
ax13.legend(fontsize=10)
ax13.grid(True, alpha=0.3)

# Correlation: Concat vs Native
corr_concat_native, _ = pearsonr(dpt_concat, dpt_native)
ax14.scatter(dpt_native, dpt_concat, s=2, alpha=0.3, c='orange', rasterized=True)
ax14.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
ax14.set_xlabel('Native Pseudotime', fontsize=11)
ax14.set_ylabel('Concat Pseudotime', fontsize=11)
ax14.set_title(f'Concat vs Native\nCorr: {corr_concat_native:.3f}', fontsize=12, fontweight='bold')
ax14.legend(fontsize=9)
ax14.grid(True, alpha=0.3)

# Correlation: Attention vs Native
corr_attention_native, _ = pearsonr(dpt_attention, dpt_native)
ax15.scatter(dpt_native, dpt_attention, s=2, alpha=0.3, c='blue', rasterized=True)
ax15.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
ax15.set_xlabel('Native Pseudotime', fontsize=11)
ax15.set_ylabel('Attention Pseudotime', fontsize=11)
ax15.set_title(f'Attention vs Native\nCorr: {corr_attention_native:.3f}', fontsize=12, fontweight='bold')
ax15.legend(fontsize=9)
ax15.grid(True, alpha=0.3)

plt.suptitle('V Space Comparison: Concatenation vs Attention-Based Batch Correction (beta=1.0)',
             fontsize=16, fontweight='bold', y=0.998)

# Save
output_file = 'concat_vs_attention_comprehensive_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comprehensive comparison to: {output_file}")

output_pdf = 'concat_vs_attention_comprehensive_comparison.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"✓ Saved PDF to: {output_pdf}")

plt.close()

# Create a focused comparison plot
print("\n  Creating focused comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Pseudotime
scatter1 = axes[0, 0].scatter(v_native[:, 0], v_native[:, 1], c=dpt_native,
                              cmap='viridis', s=5, alpha=0.7, rasterized=True)
axes[0, 0].set_xlabel('V1', fontsize=12)
axes[0, 0].set_ylabel('V2', fontsize=12)
axes[0, 0].set_title('Native Decipher', fontsize=13, fontweight='bold')
plt.colorbar(scatter1, ax=axes[0, 0], label='Pseudotime')

scatter2 = axes[0, 1].scatter(v_concat[:, 0], v_concat[:, 1], c=dpt_concat,
                              cmap='viridis', s=5, alpha=0.7, rasterized=True)
axes[0, 1].set_xlabel('V1', fontsize=12)
axes[0, 1].set_ylabel('V2', fontsize=12)
axes[0, 1].set_title('Concatenation (beta=1.0)', fontsize=13, fontweight='bold')
plt.colorbar(scatter2, ax=axes[0, 1], label='Pseudotime')

scatter3 = axes[0, 2].scatter(v_attention[:, 0], v_attention[:, 1], c=dpt_attention,
                              cmap='viridis', s=5, alpha=0.7, rasterized=True)
axes[0, 2].set_xlabel('V1', fontsize=12)
axes[0, 2].set_ylabel('V2', fontsize=12)
axes[0, 2].set_title('Attention 2-heads (beta=1.0)', fontsize=13, fontweight='bold')
plt.colorbar(scatter3, ax=axes[0, 2], label='Pseudotime')

# Row 2: Donor mixing
for ax, v_data, title in [(axes[1, 0], v_native, 'Native'),
                           (axes[1, 1], v_concat, 'Concatenation'),
                           (axes[1, 2], v_attention, 'Attention')]:
    for i, donor in enumerate(top_donors):
        mask = donors == donor
        ax.scatter(v_data[mask, 0], v_data[mask, 1],
                  c=[donor_colors[i]], s=4, alpha=0.6, rasterized=True)
    mask_other = ~donors.isin(top_donors)
    ax.scatter(v_data[mask_other, 0], v_data[mask_other, 1],
              c='lightgray', s=2, alpha=0.3, rasterized=True)
    ax.set_xlabel('V1', fontsize=12)
    ax.set_ylabel('V2', fontsize=12)
    ax.set_title(f'{title}\n(Colored by Donor)', fontsize=13, fontweight='bold')

plt.suptitle('V Space: Pseudotime & Donor Mixing Comparison', fontsize=15, fontweight='bold')
plt.tight_layout()

output_focused = 'concat_vs_attention_focused.png'
plt.savefig(output_focused, dpi=300, bbox_inches='tight')
print(f"✓ Saved focused comparison to: {output_focused}")

output_focused_pdf = 'concat_vs_attention_focused.pdf'
plt.savefig(output_focused_pdf, format='pdf', bbox_inches='tight')
print(f"✓ Saved focused PDF to: {output_focused_pdf}")

plt.close()

# Save updated data with pseudotime
print("\n  Saving pseudotime to AnnData objects...")
adata_concat.write('BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad')
adata_attention.write('../Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad')
print("✓ Updated AnnData objects saved")

# Print summary
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print("\n1. BATCH MIXING (Higher = Better Mixing):")
print(f"   Native:              {mixing_native:.3f}")
print(f"   Concatenation:       {mixing_concat:.3f}  [Δ = +{mixing_concat - mixing_native:.3f}]")
print(f"   Attention (2 heads): {mixing_attention:.3f}  [Δ = +{mixing_attention - mixing_native:.3f}]")
print(f"   Winner: {'Attention' if mixing_attention > mixing_concat else 'Concatenation'}")

if cell_type_key:
    print("\n2. BIOLOGICAL STRUCTURE (Higher = Better Cell Type Separation):")
    print(f"   Native:              {sil_native:.3f}")
    print(f"   Concatenation:       {sil_concat:.3f}  [Δ = {sil_concat - sil_native:+.3f}]")
    print(f"   Attention (2 heads): {sil_attention:.3f}  [Δ = {sil_attention - sil_native:+.3f}]")
    print(f"   Winner: {'Attention' if sil_attention > sil_concat else 'Concatenation'}")

print("\n3. PSEUDOTIME PRESERVATION (Correlation with Native):")
print(f"   Concatenation vs Native:   {corr_concat_native:.3f}")
print(f"   Attention vs Native:       {corr_attention_native:.3f}")
print(f"   Winner: {'Attention' if corr_attention_native > corr_concat_native else 'Concatenation'}")

print("\n4. V SPACE VARIANCE:")
print(f"   Native:              {v_vars[0]:.3f}")
print(f"   Concatenation:       {v_vars[1]:.3f}")
print(f"   Attention (2 heads): {v_vars[2]:.3f}")

print("\n" + "=" * 80)
print("OUTPUTS SAVED")
print("=" * 80)
print(f"\n1. {output_file}")
print("   - Comprehensive 5x3 comparison (pseudotime, cell types, donors, metrics)")

print(f"\n2. {output_focused}")
print("   - Focused 2x3 comparison (pseudotime + donor mixing)")

print(f"\n3. PDF versions")
print(f"   - {output_pdf}")
print(f"   - {output_focused_pdf}")

print("\n" + "=" * 80)
print("✓ COMPARISON COMPLETE!")
print("=" * 80)
