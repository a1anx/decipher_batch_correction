# BoneMarrowMap Batch Correction Analysis

This directory contains scripts for training and visualizing Decipher batch correction on the BoneMarrowMap dataset (263k cells, 45 donors).

## Files

### Analysis Scripts
1. **`preprocess_bonemarrowmap.py`** - Preprocessing and gene filtering
   - Loads the full dataset
   - Explores metadata structure
   - Filters to top 5k highly variable genes
   - Saves: `BoneMarrowMap_5k_genes.h5ad`

2. **`run_bonemarrowmap_analysis.py`** - Training pipeline
   - Trains batch-corrected Decipher model
   - Auto-detects batch/donor and cell type columns
   - Computes embeddings and attention weights
   - Saves: `bonemarrowmap_batch_corrected.h5ad`

3. **`visualize_bonemarrowmap.py`** - Comprehensive visualization
   - Batch correction quality (UMAP, mixing scores)
   - Biological signal preservation (cell types, trajectories)
   - Attention analysis (heatmaps, distributions)
   - Saves: Multiple PNG files

### Data Files (gitignored)
- `BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad` - Original dataset
- `BoneMarrowMap_5k_genes.h5ad` - Filtered dataset
- `bonemarrowmap_batch_corrected.h5ad` - Results with embeddings
- `*.png` - Visualization outputs

## Quick Start

### Running Locally

```bash
cd "Bone_Marrow_Hematopoiesis"

# 1. Preprocess
python preprocess_bonemarrowmap.py

# 2. Train (may take several hours for 263k cells)
python run_bonemarrowmap_analysis.py

# 3. Visualize
python visualize_bonemarrowmap.py
```

### Running on GCP (Recommended)

The full 263k cell dataset is best processed on a cloud VM with more resources.

#### 1. Set up GCP VM

```bash
# Create VM instance (recommended specs)
gcloud compute instances create decipher-analysis \
  --machine-type=n1-highmem-8 \
  --boot-disk-size=100GB \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --zone=us-central1-a

# Or with GPU (5-10x faster):
gcloud compute instances create decipher-analysis-gpu \
  --machine-type=n1-highmem-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --boot-disk-size=100GB \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --zone=us-central1-a \
  --maintenance-policy=TERMINATE

# SSH into VM
gcloud compute ssh decipher-analysis --zone=us-central1-a
```

#### 2. Install dependencies on VM

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip git

# Install Python packages
pip3 install scanpy torch pyro-ppl matplotlib seaborn scikit-learn

# If using GPU, install CUDA-enabled PyTorch
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Upload data and code

```bash
# From your local machine:

# Clone repository
git clone <your-repo-url>

# Upload the h5ad file (from local machine)
gcloud compute scp \
  "BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad" \
  decipher-analysis:~/decipher_batch_correction/Bone_Marrow_Hematopoiesis/ \
  --zone=us-central1-a

# Or use Google Cloud Storage
# gsutil cp BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad gs://your-bucket/
# Then on VM: gsutil cp gs://your-bucket/BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad .
```

#### 4. Run analysis on VM

```bash
cd ~/decipher_batch_correction/Bone_Marrow_Hematopoiesis

# 1. Preprocess (~10-30 minutes)
python3 preprocess_bonemarrowmap.py

# 2. Train (~3-6 hours on CPU, ~1-2 hours on GPU)
python3 run_bonemarrowmap_analysis.py

# 3. Visualize (~10-20 minutes)
python3 visualize_bonemarrowmap.py
```

#### 5. Download results

```bash
# From your local machine:
gcloud compute scp \
  decipher-analysis:~/decipher_batch_correction/Bone_Marrow_Hematopoiesis/*.png \
  . \
  --zone=us-central1-a

gcloud compute scp \
  decipher-analysis:~/decipher_batch_correction/Bone_Marrow_Hematopoiesis/bonemarrowmap_batch_corrected.h5ad \
  . \
  --zone=us-central1-a
```

#### 6. Clean up

```bash
# Stop VM (to avoid charges)
gcloud compute instances stop decipher-analysis --zone=us-central1-a

# Delete VM when done
gcloud compute instances delete decipher-analysis --zone=us-central1-a
```

## Expected Outputs

### From preprocessing:
- `BoneMarrowMap_5k_genes.h5ad` (~50-200 MB)
- `BoneMarrowMap_preprocessing_summary.txt`

### From training:
- `bonemarrowmap_batch_corrected.h5ad` (with embeddings)
- `bonemarrowmap_loss_curves.png`

### From visualization:
- `bonemarrowmap_batch_correction_comparison.png` (3×3 grid)
- `bonemarrowmap_attention_analysis.png`
- `bonemarrowmap_biological_preservation.png`

## GCP Cost Estimates

**n1-highmem-8 (32GB RAM):**
- Cost: ~$0.48/hour
- Expected runtime: 4-7 hours
- Total cost: ~$2-3.50

**With NVIDIA T4 GPU:**
- Cost: ~$0.83/hour
- Expected runtime: 2-3 hours
- Total cost: ~$1.66-2.50

**Cost-saving tips:**
- Use preemptible instances (70% discount, may be interrupted)
- Stop VM when not in use
- Use `n1-standard-8` if highmem is too expensive

## Troubleshooting

### Out of Memory (OOM)
- Increase VM RAM: Use `n1-highmem-16` or `n1-standard-16`
- Reduce batch size in `run_bonemarrowmap_analysis.py`
- Subsample to fewer cells using `subsample_dataset.py`

### CUDA Errors
- Verify CUDA installation: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch: `pip3 install torch --index-url https://download.pytorch.org/whl/cu118`
- Fall back to CPU: Will still work, just slower

### Missing Columns
- Check available columns: Run preprocessing script first
- Manually specify in scripts if auto-detection fails

## Notes

- Gene filtering to 5k HVGs **preserves batch effects** while reducing memory/compute
- HVG selection uses Seurat v3 with batch correction to maintain batch-specific variation
- Training time scales with cell count: ~1 sec per 100 cells on CPU, ~0.2 sec on GPU
- Visualization computes UMAP and neighbor graphs, which may take time for 263k cells

## References

- BoneMarrowMap: [GitHub](https://github.com/andygxzeng/BoneMarrowMap)
- Publication: Blood Cancer Discovery (2025)
- Decipher: Original trajectory inference method
- This implementation: Adds attention-based batch correction to Decipher
