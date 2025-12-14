"""
Explore cell types in BoneMarrowMap dataset to identify good candidates for subsetting.

Following TA feedback:
- Find cell types with good trajectory structure
- Identify cell types with sufficient cell counts across multiple donors
- Look for lineages with known biological markers for validation
"""

import scanpy as sc
import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
adata = sc.read_h5ad("BoneMarrowMap_small_20k_5kgenes.h5ad")

print(f"\nDataset overview:")
print(f"  Total cells: {adata.n_obs:,}")
print(f"  Total genes: {adata.n_vars:,}")
print(f"  Batches (donors): {adata.obs['Donor'].nunique()}")
print(f"  Cell types: {adata.obs['CellType'].nunique()}")

# Get cell type counts
print("\n" + "="*80)
print("CELL TYPE DISTRIBUTION")
print("="*80)
cell_type_counts = adata.obs['CellType'].value_counts()
print(f"\nTotal cell types: {len(cell_type_counts)}")
print("\nTop 20 most abundant cell types:")
print(cell_type_counts.head(20).to_string())

# Get donor distribution per cell type
print("\n" + "="*80)
print("DONOR REPRESENTATION PER CELL TYPE (Top 20)")
print("="*80)
donor_per_celltype = adata.obs.groupby('CellType')['Donor'].nunique().sort_values(ascending=False)
print("\nCell types with most donor diversity:")
print(donor_per_celltype.head(20).to_string())

# Combine information
print("\n" + "="*80)
print("RECOMMENDED CELL TYPES FOR SUBSETTING")
print("="*80)
print("\nCriteria: >500 cells AND >20 donors (good statistical power)\n")

summary = pd.DataFrame({
    'n_cells': cell_type_counts,
    'n_donors': donor_per_celltype
})
summary = summary.sort_values('n_cells', ascending=False)

good_candidates = summary[(summary['n_cells'] >= 500) & (summary['n_donors'] >= 20)]
print(good_candidates.to_string())

# Show specific lineages of interest
print("\n" + "="*80)
print("HEMATOPOIETIC LINEAGES (Known Trajectories)")
print("="*80)

# Common hematopoietic lineages with well-known trajectories
lineages_of_interest = {
    'Erythroid': ['Erythroid', 'erythroid', 'RBC', 'Ery'],
    'Myeloid': ['Myeloid', 'myeloid', 'Mono', 'Neutro', 'Granulo'],
    'Lymphoid': ['T cell', 'B cell', 'NK', 'lymphoid'],
    'HSC/Progenitor': ['HSC', 'MPP', 'progenitor', 'Prog']
}

for lineage_name, keywords in lineages_of_interest.items():
    matching = cell_type_counts[cell_type_counts.index.str.contains('|'.join(keywords), case=False, na=False)]
    if len(matching) > 0:
        print(f"\n{lineage_name} cells:")
        for ct, count in matching.items():
            n_donors = donor_per_celltype.get(ct, 0)
            print(f"  {ct:40s}: {count:5d} cells, {n_donors:2d} donors")

# HSC-specific analysis (if present)
hsc_related = cell_type_counts[cell_type_counts.index.str.contains('HSC|MPP|HSPC', case=False, na=False)]
if len(hsc_related) > 0:
    print("\n" + "="*80)
    print("HSC/PROGENITOR ANALYSIS (Root of trajectories)")
    print("="*80)
    for ct, count in hsc_related.items():
        n_donors = donor_per_celltype.get(ct, 0)
        print(f"  {ct:40s}: {count:5d} cells, {n_donors:2d} donors")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Choose a cell type with:
   - Clear developmental trajectory (e.g., Erythroid: HSC → Proerythroblast → Mature RBC)
   - Good cell count (>500 cells)
   - Multiple donors (>20 donors for batch correction)

2. Known biological markers to validate:
   - Erythroid: HBB, HBA1/2, GYPA, KLF1
   - Myeloid: CD14, CD68, MPO, CEBPA
   - T cells: CD3D, CD3E, CD4, CD8A

3. Subset the data and rerun Decipher to see if "blob" resolves
""")

print("\n✓ Analysis complete!")
