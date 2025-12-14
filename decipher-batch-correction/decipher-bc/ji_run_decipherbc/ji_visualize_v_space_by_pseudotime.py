"""
Visualization script to plot Decipher V spaces colored by pseudotime
This reproduces plots from batch_correction_comparison_2.png but colors by
latent_t (pseudotime) instead of alpha-shift:
- "Original Decipher V (2D)"
- "Batch-Corrected Decipher V (2D)" (if available)
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("=" * 70)
print("VISUALIZING V SPACES BY PSEUDOTIME")
print("=" * 70)

# Load the results
print("\n[1/2] Loading data...")
adata = sc.read_h5ad('../../simulation/adata/adata_combined_2_batch_corrected.h5ad')
print(f"✓ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")
print(f"  Available representations:")
for key in adata.obsm.keys():
    print(f"    - {key}: {adata.obsm[key].shape}")

# Check for latent_t (pseudotime)
if 'latent_t' in adata.obs.columns:
    print(f"\n✓ Found pseudotime column: 'latent_t'")
    print(f"  Range: [{adata.obs['latent_t'].min():.3f}, {adata.obs['latent_t'].max():.3f}]")
else:
    print("\n⚠ Warning: 'latent_t' column not found in data!")
    print(f"Available columns: {list(adata.obs.columns)}")

# Create visualization
print("\n[2/2] Creating visualization...")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Get V space coordinates
v_coords = adata.obsm['decipher_decipher_v']
pseudotime = adata.obs['latent_t'].values

# Flatten pseudotime if it's 2D
if len(pseudotime.shape) > 1:
    pseudotime = pseudotime.flatten()

# Create scatter plot colored by pseudotime
scatter = ax.scatter(
    v_coords[:, 0],
    v_coords[:, 1],
    c=pseudotime,
    cmap='viridis',
    alpha=0.6,
    s=10
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)

# Set labels and title
ax.set_title('Original Decipher V (2D) - Colored by Pseudotime', fontsize=14, fontweight='bold')
ax.set_xlabel('V Dimension 1', fontsize=12)
ax.set_ylabel('V Dimension 2', fontsize=12)
ax.grid(True, alpha=0.3)

# Save figure
plt.tight_layout()
output_file = 'v_space_by_pseudotime.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# Create side-by-side comparison
print("\nCreating side-by-side comparison...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Define colors for batches (alpha-shift)
batch_colors = {
    'False': '#1f77b4',
    'Alpha 0.01': '#ff7f0e',
    'Alpha 0.05': '#2ca02c',
    'Alpha 0.1': '#d62728'
}

# Left plot: Colored by alpha-shift (original)
ax_left = axes[0]
for batch in adata.obs['shift'].unique():
    mask = adata.obs['shift'] == batch
    ax_left.scatter(
        v_coords[mask, 0],
        v_coords[mask, 1],
        c=batch_colors[batch],
        label=batch,
        alpha=0.6,
        s=10
    )
ax_left.set_title('Original Decipher V (2D) - Colored by Alpha-Shift',
                   fontsize=14, fontweight='bold')
ax_left.set_xlabel('V Dimension 1', fontsize=12)
ax_left.set_ylabel('V Dimension 2', fontsize=12)
ax_left.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
ax_left.grid(True, alpha=0.3)

# Right plot: Colored by pseudotime (new)
ax_right = axes[1]
scatter = ax_right.scatter(
    v_coords[:, 0],
    v_coords[:, 1],
    c=pseudotime,
    cmap='viridis',
    alpha=0.6,
    s=10
)
cbar = plt.colorbar(scatter, ax=ax_right)
cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
ax_right.set_title('Original Decipher V (2D) - Colored by Pseudotime',
                    fontsize=14, fontweight='bold')
ax_right.set_xlabel('V Dimension 1', fontsize=12)
ax_right.set_ylabel('V Dimension 2', fontsize=12)
ax_right.grid(True, alpha=0.3)

plt.tight_layout()
output_file_comparison = 'v_space_comparison_shift_vs_pseudotime.png'
plt.savefig(output_file_comparison, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file_comparison}")
plt.close()

# Create Batch-Corrected V space visualizations (if available)
if 'X_decipher_batch_corrected_v' in adata.obsm.keys():
    print("\nCreating Batch-Corrected V space visualizations...")
    v_bc_coords = adata.obsm['X_decipher_batch_corrected_v']

    # Single batch-corrected V space plot colored by pseudotime
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    scatter = ax.scatter(
        v_bc_coords[:, 0],
        v_bc_coords[:, 1],
        c=pseudotime,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax.set_title('Batch-Corrected Decipher V (2D) - Colored by Pseudotime', fontsize=14, fontweight='bold')
    ax.set_xlabel('V Dimension 1', fontsize=12)
    ax.set_ylabel('V Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file_v_bc = 'v_batch_corrected_by_pseudotime.png'
    plt.savefig(output_file_v_bc, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file_v_bc}")
    plt.close()

    # Batch-corrected V space side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left plot: Batch-corrected V space colored by alpha-shift
    ax_left = axes[0]
    for batch in adata.obs['shift'].unique():
        mask = adata.obs['shift'] == batch
        ax_left.scatter(
            v_bc_coords[mask, 0],
            v_bc_coords[mask, 1],
            c=batch_colors[batch],
            label=batch,
            alpha=0.6,
            s=10
        )
    ax_left.set_title('Batch-Corrected Decipher V (2D) - Colored by Alpha-Shift',
                       fontsize=14, fontweight='bold')
    ax_left.set_xlabel('V Dimension 1', fontsize=12)
    ax_left.set_ylabel('V Dimension 2', fontsize=12)
    ax_left.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_left.grid(True, alpha=0.3)

    # Right plot: Batch-corrected V space colored by pseudotime
    ax_right = axes[1]
    scatter = ax_right.scatter(
        v_bc_coords[:, 0],
        v_bc_coords[:, 1],
        c=pseudotime,
        cmap='viridis',
        alpha=0.6,
        s=10
    )
    cbar = plt.colorbar(scatter, ax=ax_right)
    cbar.set_label('Pseudotime (latent_t)', rotation=270, labelpad=20, fontsize=12)
    ax_right.set_title('Batch-Corrected Decipher V (2D) - Colored by Pseudotime',
                        fontsize=14, fontweight='bold')
    ax_right.set_xlabel('V Dimension 1', fontsize=12)
    ax_right.set_ylabel('V Dimension 2', fontsize=12)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file_v_bc_comparison = 'v_batch_corrected_comparison_shift_vs_pseudotime.png'
    plt.savefig(output_file_v_bc_comparison, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file_v_bc_comparison}")
    plt.close()
else:
    print("\n⚠ Batch-corrected V space (X_decipher_batch_corrected_v) not found in dataset")
    print("  Skipping batch-corrected visualizations.")
    output_file_v_bc = None
    output_file_v_bc_comparison = None

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nFiles created:")
print(f"  1. {output_file}")
print("     - Original Decipher V space (2D) colored by pseudotime")
print(f"\n  2. {output_file_comparison}")
print("     - Side-by-side comparison:")
print("       Left:  Original V space colored by alpha-shift (condition)")
print("       Right: Original V space colored by pseudotime (latent_t)")

if output_file_v_bc is not None:
    print(f"\n  3. {output_file_v_bc}")
    print("     - Batch-Corrected Decipher V space (2D) colored by pseudotime")
    print(f"\n  4. {output_file_v_bc_comparison}")
    print("     - Side-by-side comparison:")
    print("       Left:  Batch-corrected V space colored by alpha-shift (condition)")
    print("       Right: Batch-corrected V space colored by pseudotime (latent_t)")

print("\nInterpretation:")
print("  - Pseudotime (latent_t) represents the temporal progression of cells")
print("  - The color gradient shows how cells evolve over pseudotime")
print("  - V space (2D): Component/trajectory representation")
print("  - Compare original vs batch-corrected to see how batch correction affects")
print("    the biological trajectory structure")
print("  - Compare with alpha-shift plots to see batch effects vs biological time")

print("\n" + "=" * 70)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 70)
