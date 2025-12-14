"""
Visualize and compare v space: Native Decipher vs Concatenation Batch-Corrected Decipher

This script:
1. Loads both native and batch-corrected (concatenation) results
2. Computes pseudotime using diffusion pseudotime (DPT) on v space
3. Creates side-by-side comparison plots colored by pseudotime
4. Also shows plots colored by cell type and donor to assess batch correction
"""

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

print("=" * 80)
print("V SPACE COMPARISON: Native Decipher vs Concatenation Batch-Corrected")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
print("  Loading native Decipher results...")
adata_native = sc.read_h5ad('bonemarrowmap_small_regular_decipher.h5ad')

print("  Loading concatenation batch-corrected results...")
adata_concat = sc.read_h5ad('BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad')

print(f"✓ Loaded both datasets: {adata_native.shape[0]:,} cells")

# Verify embeddings exist
if 'X_decipher_regular_v' not in adata_native.obsm:
    print("ERROR: Native Decipher v embeddings not found!")
    print(f"Available obsm keys: {list(adata_native.obsm.keys())}")
    exit(1)

if 'X_decipher_concat_v' not in adata_concat.obsm:
    print("ERROR: Concatenation v embeddings not found!")
    print(f"Available obsm keys: {list(adata_concat.obsm.keys())}")
    exit(1)

print("✓ Found v embeddings in both datasets")

# Get cell type column
cell_type_cols = [col for col in adata_native.obs.columns
                  if 'celltype' in col.lower() or 'cell_type' in col.lower()]
if cell_type_cols:
    cell_type_key = cell_type_cols[0]
    print(f"  Cell type column: '{cell_type_key}' ({adata_native.obs[cell_type_key].nunique()} types)")
else:
    cell_type_key = None
    print("  No cell type column found")

# Compute pseudotime using diffusion pseudotime (DPT)
print("\n[2/4] Computing pseudotime on v space...")

def compute_dpt_on_v(adata, v_key='X_decipher_regular_v', suffix='native'):
    """Compute diffusion pseudotime on v space"""
    print(f"\n  Computing DPT for {suffix}...")

    # Create temporary adata with v as .X for DPT computation
    adata_v = sc.AnnData(X=adata.obsm[v_key].copy(), obs=adata.obs.copy())

    # Compute neighbors on v space
    sc.pp.neighbors(adata_v, n_neighbors=30, use_rep='X')

    # Compute diffusion map
    sc.tl.diffmap(adata_v)

    # For DPT, we need to select a root cell
    # Choose a cell from an early progenitor type if available
    # Otherwise, use the cell with minimum first diffusion component
    if cell_type_key:
        # Try to find HSC/progenitor cells
        progenitor_keywords = ['HSC', 'HSPC', 'Stem', 'Progenitor', 'prog', 'stem']
        cell_types = adata.obs[cell_type_key].astype(str)

        root_idx = None
        for keyword in progenitor_keywords:
            matches = cell_types.str.contains(keyword, case=False, na=False)
            if matches.any():
                # Find cell in this type with median diffusion coordinate
                candidate_cells = np.where(matches)[0]
                dc1_vals = adata_v.obsm['X_diffmap'][candidate_cells, 0]
                median_idx = candidate_cells[np.argsort(dc1_vals)[len(dc1_vals)//2]]
                root_idx = median_idx
                print(f"    Selected root cell from {cell_types[root_idx]} (cell {root_idx})")
                break

        if root_idx is None:
            # Fallback: use minimum of first diffusion component
            root_idx = np.argmin(adata_v.obsm['X_diffmap'][:, 0])
            print(f"    Selected root cell based on diffusion map (cell {root_idx})")
    else:
        root_idx = np.argmin(adata_v.obsm['X_diffmap'][:, 0])
        print(f"    Selected root cell based on diffusion map (cell {root_idx})")

    # Set root and compute DPT
    adata_v.uns['iroot'] = root_idx
    sc.tl.dpt(adata_v)

    # Copy DPT back to original adata
    adata.obs[f'dpt_{suffix}'] = adata_v.obs['dpt_pseudotime'].values

    print(f"    ✓ DPT computed for {suffix}")
    print(f"      Range: [{adata.obs[f'dpt_{suffix}'].min():.3f}, {adata.obs[f'dpt_{suffix}'].max():.3f}]")

    return adata

# Compute DPT for both
adata_native = compute_dpt_on_v(adata_native, 'X_decipher_regular_v', 'native')
adata_concat = compute_dpt_on_v(adata_concat, 'X_decipher_concat_v', 'concat')

print("\n✓ Pseudotime computed for both datasets")

# Create comprehensive comparison plots
print("\n[3/4] Creating comparison plots...")

# Set up the figure with multiple panels
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4)

# Row 1: V space colored by pseudotime
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])

# Row 2: V space colored by cell type
ax3 = fig.add_subplot(gs[1, 0:2])
ax4 = fig.add_subplot(gs[1, 2:4])

# Row 3: V space colored by donor (batch)
ax5 = fig.add_subplot(gs[2, 0:2])
ax6 = fig.add_subplot(gs[2, 2:4])

# Row 4: Pseudotime distributions
ax7 = fig.add_subplot(gs[3, 0:2])
ax8 = fig.add_subplot(gs[3, 2:4])

# Plot 1 & 2: V space colored by pseudotime
print("  Plotting v space with pseudotime...")

v_native = adata_native.obsm['X_decipher_regular_v']
v_concat = adata_concat.obsm['X_decipher_concat_v']
dpt_native = adata_native.obs['dpt_native']
dpt_concat = adata_concat.obs['dpt_concat']

scatter1 = ax1.scatter(v_native[:, 0], v_native[:, 1], c=dpt_native,
                       cmap='viridis', s=5, alpha=0.6, rasterized=True)
ax1.set_xlabel('V1', fontsize=12)
ax1.set_ylabel('V2', fontsize=12)
ax1.set_title('Native Decipher V Space\n(colored by pseudotime)', fontsize=13, fontweight='bold')
plt.colorbar(scatter1, ax=ax1, label='Pseudotime')

scatter2 = ax2.scatter(v_concat[:, 0], v_concat[:, 1], c=dpt_concat,
                       cmap='viridis', s=5, alpha=0.6, rasterized=True)
ax2.set_xlabel('V1', fontsize=12)
ax2.set_ylabel('V2', fontsize=12)
ax2.set_title('Concatenation Batch-Corrected V Space\n(colored by pseudotime, beta=1)', fontsize=13, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Pseudotime')

# Plot 3 & 4: V space colored by cell type
if cell_type_key:
    print("  Plotting v space with cell types...")

    # Get unique cell types and create color map
    cell_types = adata_native.obs[cell_type_key]
    unique_types = cell_types.unique()
    n_types = len(unique_types)
    colors = plt.cm.tab20(np.linspace(0, 1, n_types))
    type_to_color = dict(zip(unique_types, colors))

    # Plot native
    for i, cell_type in enumerate(unique_types):
        mask = cell_types == cell_type
        ax3.scatter(v_native[mask, 0], v_native[mask, 1],
                   c=[type_to_color[cell_type]], label=cell_type,
                   s=5, alpha=0.6, rasterized=True)

    ax3.set_xlabel('V1', fontsize=12)
    ax3.set_ylabel('V2', fontsize=12)
    ax3.set_title('Native Decipher V Space\n(colored by cell type)', fontsize=13, fontweight='bold')
    if n_types <= 20:
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)

    # Plot concat
    for i, cell_type in enumerate(unique_types):
        mask = cell_types == cell_type
        ax4.scatter(v_concat[mask, 0], v_concat[mask, 1],
                   c=[type_to_color[cell_type]], label=cell_type,
                   s=5, alpha=0.6, rasterized=True)

    ax4.set_xlabel('V1', fontsize=12)
    ax4.set_ylabel('V2', fontsize=12)
    ax4.set_title('Concatenation Batch-Corrected V Space\n(colored by cell type, beta=1)', fontsize=13, fontweight='bold')
    if n_types <= 20:
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)

# Plot 5 & 6: V space colored by donor (batch effect visualization)
print("  Plotting v space with donors...")

donors = adata_native.obs['Donor']
unique_donors = donors.unique()
n_donors = len(unique_donors)

# Sample donors for visualization if too many
if n_donors > 20:
    # Show top 10 donors by cell count
    top_donors = donors.value_counts().head(10).index
    donor_colors = plt.cm.tab20(np.linspace(0, 1, 10))

    for i, donor in enumerate(top_donors):
        mask = donors == donor
        ax5.scatter(v_native[mask, 0], v_native[mask, 1],
                   c=[donor_colors[i]], label=donor[:20],  # Truncate long names
                   s=5, alpha=0.6, rasterized=True)
        ax6.scatter(v_concat[mask, 0], v_concat[mask, 1],
                   c=[donor_colors[i]], label=donor[:20],
                   s=5, alpha=0.6, rasterized=True)

    # Plot remaining donors in gray
    mask_other = ~donors.isin(top_donors)
    ax5.scatter(v_native[mask_other, 0], v_native[mask_other, 1],
               c='lightgray', label='Other donors',
               s=3, alpha=0.3, rasterized=True)
    ax6.scatter(v_concat[mask_other, 0], v_concat[mask_other, 1],
               c='lightgray', label='Other donors',
               s=3, alpha=0.3, rasterized=True)
else:
    donor_colors = plt.cm.tab20(np.linspace(0, 1, n_donors))
    for i, donor in enumerate(unique_donors):
        mask = donors == donor
        ax5.scatter(v_native[mask, 0], v_native[mask, 1],
                   c=[donor_colors[i]], label=donor[:20],
                   s=5, alpha=0.6, rasterized=True)
        ax6.scatter(v_concat[mask, 0], v_concat[mask, 1],
                   c=[donor_colors[i]], label=donor[:20],
                   s=5, alpha=0.6, rasterized=True)

ax5.set_xlabel('V1', fontsize=12)
ax5.set_ylabel('V2', fontsize=12)
ax5.set_title('Native Decipher V Space\n(colored by donor - batch effect)', fontsize=13, fontweight='bold')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=2)

ax6.set_xlabel('V1', fontsize=12)
ax6.set_ylabel('V2', fontsize=12)
ax6.set_title('Concatenation Batch-Corrected V Space\n(colored by donor - batch correction)', fontsize=13, fontweight='bold')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, markerscale=2)

# Plot 7 & 8: Pseudotime distributions
print("  Plotting pseudotime distributions...")

# Histogram comparison
ax7.hist(dpt_native, bins=50, alpha=0.7, label='Native Decipher', color='blue', edgecolor='black')
ax7.hist(dpt_concat, bins=50, alpha=0.7, label='Concat Batch-Corrected', color='orange', edgecolor='black')
ax7.set_xlabel('Pseudotime (DPT)', fontsize=12)
ax7.set_ylabel('Cell count', fontsize=12)
ax7.set_title('Pseudotime Distribution Comparison', fontsize=13, fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)

# If cell types available, show pseudotime progression by cell type
if cell_type_key:
    # Create box plot
    df = pd.DataFrame({
        'Native': dpt_native,
        'Concat': dpt_concat,
        'CellType': adata_native.obs[cell_type_key].values
    })

    # Melt for seaborn
    df_melt = df.melt(id_vars=['CellType'], var_name='Method', value_name='Pseudotime')

    # Sort cell types by median pseudotime (native)
    type_order = df.groupby('CellType')['Native'].median().sort_values().index

    if len(type_order) <= 15:  # Only plot if reasonable number of types
        sns.boxplot(data=df_melt, x='CellType', y='Pseudotime', hue='Method', ax=ax8)
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax8.set_xlabel('Cell Type', fontsize=12)
        ax8.set_ylabel('Pseudotime (DPT)', fontsize=12)
        ax8.set_title('Pseudotime by Cell Type', fontsize=13, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3, axis='y')
    else:
        # Just show correlation
        ax8.scatter(dpt_native, dpt_concat, s=3, alpha=0.3, rasterized=True)
        ax8.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
        ax8.set_xlabel('Native Decipher Pseudotime', fontsize=12)
        ax8.set_ylabel('Concat Batch-Corrected Pseudotime', fontsize=12)
        ax8.set_title('Pseudotime Correlation', fontsize=13, fontweight='bold')
        corr = np.corrcoef(dpt_native, dpt_concat)[0, 1]
        ax8.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax8.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
else:
    # Just show correlation
    ax8.scatter(dpt_native, dpt_concat, s=3, alpha=0.3, rasterized=True)
    ax8.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
    ax8.set_xlabel('Native Decipher Pseudotime', fontsize=12)
    ax8.set_ylabel('Concat Batch-Corrected Pseudotime', fontsize=12)
    ax8.set_title('Pseudotime Correlation', fontsize=13, fontweight='bold')
    corr = np.corrcoef(dpt_native, dpt_concat)[0, 1]
    ax8.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax8.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

plt.suptitle('V Space Comparison: Native Decipher vs Concatenation Batch-Corrected (beta=1.0)',
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
print("\n[4/4] Saving plots...")
output_file = 'vspace_comparison_native_vs_concat_pseudotime.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive comparison to: {output_file}")

# Also save high-res PDF
output_pdf = 'vspace_comparison_native_vs_concat_pseudotime.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
print(f"✓ Saved high-resolution PDF to: {output_pdf}")

plt.close()

# Create a simpler focused plot just for pseudotime
print("\n  Creating focused pseudotime plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(v_native[:, 0], v_native[:, 1], c=dpt_native,
                           cmap='viridis', s=10, alpha=0.7, rasterized=True)
axes[0].set_xlabel('V1', fontsize=14)
axes[0].set_ylabel('V2', fontsize=14)
axes[0].set_title('Native Decipher V Space', fontsize=15, fontweight='bold')
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Pseudotime (DPT)', fontsize=12)

scatter2 = axes[1].scatter(v_concat[:, 0], v_concat[:, 1], c=dpt_concat,
                           cmap='viridis', s=10, alpha=0.7, rasterized=True)
axes[1].set_xlabel('V1', fontsize=14)
axes[1].set_ylabel('V2', fontsize=14)
axes[1].set_title('Concatenation Batch-Corrected (beta=1.0)', fontsize=15, fontweight='bold')
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Pseudotime (DPT)', fontsize=12)

plt.suptitle('V Space Colored by Pseudotime', fontsize=16, fontweight='bold')
plt.tight_layout()

output_simple = 'vspace_pseudotime_comparison_simple.png'
plt.savefig(output_simple, dpi=300, bbox_inches='tight')
print(f"✓ Saved focused pseudotime plot to: {output_simple}")

output_simple_pdf = 'vspace_pseudotime_comparison_simple.pdf'
plt.savefig(output_simple_pdf, format='pdf', bbox_inches='tight')
print(f"✓ Saved focused PDF to: {output_simple_pdf}")

plt.close()

# Save pseudotime back to original files
print("\n  Saving pseudotime to AnnData objects...")
adata_native.write('bonemarrowmap_small_regular_decipher.h5ad')
adata_concat.write('BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad')
print("✓ Pseudotime saved to both AnnData objects")

# Print summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("\nPseudotime Statistics:")
print(f"  Native Decipher:")
print(f"    Mean: {dpt_native.mean():.3f}")
print(f"    Std:  {dpt_native.std():.3f}")
print(f"    Range: [{dpt_native.min():.3f}, {dpt_native.max():.3f}]")

print(f"\n  Concatenation Batch-Corrected:")
print(f"    Mean: {dpt_concat.mean():.3f}")
print(f"    Std:  {dpt_concat.std():.3f}")
print(f"    Range: [{dpt_concat.min():.3f}, {dpt_concat.max():.3f}]")

print(f"\n  Correlation between methods: {np.corrcoef(dpt_native, dpt_concat)[0, 1]:.3f}")

print("\nV Space Variance:")
v_native_var = np.var(v_native, axis=0)
v_concat_var = np.var(v_concat, axis=0)
print(f"  Native Decipher:")
print(f"    V1 variance: {v_native_var[0]:.3f}")
print(f"    V2 variance: {v_native_var[1]:.3f}")
print(f"    Total variance: {v_native_var.sum():.3f}")

print(f"\n  Concatenation Batch-Corrected:")
print(f"    V1 variance: {v_concat_var[0]:.3f}")
print(f"    V2 variance: {v_concat_var[1]:.3f}")
print(f"    Total variance: {v_concat_var.sum():.3f}")

print("\n" + "=" * 80)
print("OUTPUTS SAVED")
print("=" * 80)
print(f"\n1. {output_file}")
print("   - Comprehensive 4x2 panel comparison")
print("   - Shows pseudotime, cell types, donors, distributions")

print(f"\n2. {output_simple}")
print("   - Focused 1x2 comparison of v space with pseudotime")

print(f"\n3. PDF versions (high resolution)")
print(f"   - {output_pdf}")
print(f"   - {output_simple_pdf}")

print(f"\n4. Updated AnnData objects with pseudotime")
print("   - bonemarrowmap_small_regular_decipher.h5ad")
print("   - BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad")

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 80)
