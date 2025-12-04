"""
Quick script to inspect adata_combined_1.h5ad structure
"""
import scanpy as sc

# Load the data
adata = sc.read_h5ad('simulation/adata/adata_combined_1.h5ad')

print("=" * 60)
print("DATA STRUCTURE INSPECTION")
print("=" * 60)

print(f"\n1. Shape: {adata.shape} (cells x genes)")

print("\n2. Available columns in adata.obs:")
for col in adata.obs.columns:
    print(f"   - {col}")
    if adata.obs[col].dtype == 'object' or adata.obs[col].dtype.name == 'category':
        unique_vals = adata.obs[col].unique()
        if len(unique_vals) <= 10:
            print(f"     Unique values: {list(unique_vals)}")
        else:
            print(f"     Number of unique values: {len(unique_vals)}")

print("\n3. Available matrices in adata.obsm:")
for key in adata.obsm.keys():
    print(f"   - {key}: shape {adata.obsm[key].shape}")

print("\n4. Available keys in adata.uns:")
for key in adata.uns.keys():
    print(f"   - {key}")

# Check for batch information
print("\n5. Batch information:")
batch_columns = [col for col in adata.obs.columns if 'batch' in col.lower()]
if batch_columns:
    print(f"   Found batch columns: {batch_columns}")
    for col in batch_columns:
        n_batches = len(adata.obs[col].unique())
        print(f"   - {col}: {n_batches} unique batches")
        print(f"     Values: {adata.obs[col].value_counts().to_dict()}")
else:
    print("   ⚠️ No 'batch' column found!")
    print("   Available columns:", list(adata.obs.columns))

print("\n" + "=" * 60)
