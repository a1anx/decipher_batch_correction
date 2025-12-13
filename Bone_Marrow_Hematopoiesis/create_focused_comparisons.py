"""
Create focused comparison figures showing only V-space plots:
- Component V Space by Batch
- Component V Space by Pseudotime

For three key experiments:
1. Initial Training (Beta=0.1, 4 heads)
2. Beta=1.0 Training (4 heads)
3. Beta=1.0 + 2 Attention Heads Training
"""

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_focused_comparison(regular_path, batch_corrected_path, output_path, title):
    """
    Create a 2x2 comparison figure showing only V-space plots.

    Layout:
    [Regular - By Batch]      [Batch-Corrected - By Batch]
    [Regular - By Pseudotime] [Batch-Corrected - By Pseudotime]
    """

    print(f"\nCreating: {title}")
    print(f"  Regular: {regular_path}")
    print(f"  Batch-corrected: {batch_corrected_path}")

    # Load data
    adata_regular = sc.read_h5ad(regular_path)
    adata_bc = sc.read_h5ad(batch_corrected_path)

    # Get V-space embeddings
    v_regular = adata_regular.obsm['decipher_v']

    # Handle different naming conventions for batch-corrected v-space
    if 'X_decipher_batch_corrected_v' in adata_bc.obsm:
        v_bc = adata_bc.obsm['X_decipher_batch_corrected_v']
    elif 'X_decipher_concat_v' in adata_bc.obsm:
        v_bc = adata_bc.obsm['X_decipher_concat_v']
    else:
        print(f"  ✗ No V-space found! Available keys: {list(adata_bc.obsm.keys())}")
        return

    # Get batch and pseudotime info
    batch_key = None
    for col in adata_regular.obs.columns:
        if 'donor' in col.lower() and 'celltype' not in col.lower():
            batch_key = col
            break

    if batch_key is None:
        print("  ✗ No batch column found!")
        return

    batch_regular = adata_regular.obs[batch_key]
    batch_bc = adata_bc.obs[batch_key]

    # Get pseudotime
    if 'dpt_pseudotime' not in adata_regular.obs or 'dpt_pseudotime' not in adata_bc.obs:
        print("  ✗ Pseudotime not found!")
        return

    pseudotime_regular = adata_regular.obs['dpt_pseudotime'].values
    pseudotime_bc = adata_bc.obs['dpt_pseudotime'].values

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Get unique batches and assign colors
    all_batches = pd.concat([batch_regular, batch_bc]).unique()
    n_batches = len(all_batches)
    colors = sns.color_palette('tab20', n_colors=min(n_batches, 20))
    batch_color_map = {batch: colors[i % len(colors)] for i, batch in enumerate(all_batches)}

    # --- Plot 1: Regular Decipher - V Space by Batch ---
    ax = axes[0, 0]
    for batch in all_batches:
        mask = batch_regular == batch
        if mask.sum() > 0:
            ax.scatter(v_regular[mask, 0], v_regular[mask, 1],
                      c=[batch_color_map[batch]], s=3, alpha=0.6, label=batch)

    ax.set_xlabel('V1', fontsize=11, fontweight='bold')
    ax.set_ylabel('V2', fontsize=11, fontweight='bold')
    ax.set_title('Regular Decipher\nComponent V Space by Batch',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Batch-Corrected Decipher - V Space by Batch ---
    ax = axes[0, 1]
    for batch in all_batches:
        mask = batch_bc == batch
        if mask.sum() > 0:
            ax.scatter(v_bc[mask, 0], v_bc[mask, 1],
                      c=[batch_color_map[batch]], s=3, alpha=0.6, label=batch)

    ax.set_xlabel('V1', fontsize=11, fontweight='bold')
    ax.set_ylabel('V2', fontsize=11, fontweight='bold')
    ax.set_title('Batch-Corrected Decipher\nComponent V Space by Batch',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Regular Decipher - V Space by Pseudotime ---
    ax = axes[1, 0]
    valid_mask = np.isfinite(pseudotime_regular)
    scatter = ax.scatter(v_regular[valid_mask, 0], v_regular[valid_mask, 1],
                        c=pseudotime_regular[valid_mask], cmap='viridis',
                        s=3, alpha=0.6, vmin=0, vmax=1)

    ax.set_xlabel('V1', fontsize=11, fontweight='bold')
    ax.set_ylabel('V2', fontsize=11, fontweight='bold')
    ax.set_title('Regular Decipher\nComponent V Space by Pseudotime',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pseudotime', fontsize=10, fontweight='bold')

    # --- Plot 4: Batch-Corrected Decipher - V Space by Pseudotime ---
    ax = axes[1, 1]
    valid_mask = np.isfinite(pseudotime_bc)
    scatter = ax.scatter(v_bc[valid_mask, 0], v_bc[valid_mask, 1],
                        c=pseudotime_bc[valid_mask], cmap='viridis',
                        s=3, alpha=0.6, vmin=0, vmax=1)

    ax.set_xlabel('V1', fontsize=11, fontweight='bold')
    ax.set_ylabel('V2', fontsize=11, fontweight='bold')
    ax.set_title('Batch-Corrected Decipher\nComponent V Space by Pseudotime',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pseudotime', fontsize=10, fontweight='bold')

    # Overall title
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def main():
    """Create all three focused comparison figures."""

    print("=" * 80)
    print("CREATING FOCUSED V-SPACE COMPARISON FIGURES")
    print("=" * 80)

    regular_decipher = "bonemarrowmap_small_regular_decipher.h5ad"

    # 1. Initial Training (Beta=0.1, 4 heads)
    create_focused_comparison(
        regular_path=regular_decipher,
        batch_corrected_path="Initial_Training/bonemarrowmap_small_batch_corrected.h5ad",
        output_path="Initial_Training/v_space_focused_comparison.png",
        title="Beta = 0.1, Attention Heads = 4"
    )

    # 2. Beta=1.0 Training (4 heads)
    create_focused_comparison(
        regular_path=regular_decipher,
        batch_corrected_path="Beta_1.0_Training/bonemarrowmap_small_150epochs_beta1.h5ad",
        output_path="Beta_1.0_Training/v_space_focused_comparison.png",
        title="Beta = 1.0, Attention Heads = 4"
    )

    # 3. Beta=1.0 + 2 Attention Heads Training
    create_focused_comparison(
        regular_path=regular_decipher,
        batch_corrected_path="Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_small_150epochs_beta1_attnheads2.h5ad",
        output_path="Beta_1.0_AttentionHeads_2_Training/v_space_focused_comparison.png",
        title="Beta = 1.0, Attention Heads = 2"
    )

    # 4. Decipher BC Concat Training
    create_focused_comparison(
        regular_path=regular_decipher,
        batch_corrected_path="Decipher_BC_Concat_Training/BoneMarrowMap_small_20k_5kgenes_concat_beta1.h5ad",
        output_path="Decipher_BC_Concat_Training/v_space_focused_comparison.png",
        title="Decipher BC Concat (Beta = 0.1)"
    )

    print("\n" + "=" * 80)
    print("✓ ALL FOCUSED COMPARISONS CREATED")
    print("=" * 80)
    print()
    print("Output files:")
    print("  1. Initial_Training/v_space_focused_comparison.png")
    print("  2. Beta_1.0_Training/v_space_focused_comparison.png")
    print("  3. Beta_1.0_AttentionHeads_2_Training/v_space_focused_comparison.png")
    print("  4. Decipher_BC_Concat_Training/v_space_focused_comparison.png")
    print()

if __name__ == "__main__":
    main()
