import scanpy as sc

adata = sc.read_h5ad("BoneMarrowMap_small_20k_5kgenes.h5ad")
print("Available columns in adata.obs:")
print(adata.obs.columns.tolist())
print(f"\nShape: {adata.shape}")
