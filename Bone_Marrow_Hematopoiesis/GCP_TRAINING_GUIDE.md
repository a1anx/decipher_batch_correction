# GCP Training Guide - BoneMarrowMap Full Dataset

This guide shows you how to run the full BoneMarrowMap dataset (263k cells) training on Google Cloud Platform.

## Quick Start (Automated)

```bash
# 1. Edit the script to set your PROJECT_ID
nano gcp_setup_and_run.sh  # Change "your-gcp-project-id" to your actual project

# 2. Run the automated script (does everything for you)
bash gcp_setup_and_run.sh
```

**That's it!** The script will:
- Create a GCP VM with GPU
- Upload your data (3.5 GB)
- Install dependencies
- Run training (~1-1.5 hours)
- Download results
- Delete the VM

**Expected cost**: $2-3 total

---

## Manual Setup (Step-by-Step)

If you prefer to do it manually or the automated script has issues:

### Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed: https://cloud.google.com/sdk/docs/install
3. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

### Step 1: Create VM Instance

```bash
# Create instance with GPU
gcloud compute instances create bonemarrow-training \
    --zone=us-central1-a \
    --machine-type=n1-highmem-16 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --boot-disk-size=100GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform
```

**Instance specs**:
- **Machine**: n1-highmem-16 (16 vCPUs, 104 GB RAM)
- **GPU**: NVIDIA T4 (16 GB VRAM)
- **Cost**: ~$1.50/hour with GPU, ~$0.90/hour without
- **Why high-mem?**: Preprocessing the full 263k × 34k dataset requires ~50-60 GB RAM

Wait 1-2 minutes for the instance to fully boot.

### Step 2: Upload Data and Code

```bash
# SSH into instance to create directory
gcloud compute ssh bonemarrow-training --zone=us-central1-a
mkdir -p ~/bonemarrow_training
exit

# Upload dataset (3.5 GB - takes ~5-15 minutes depending on your connection)
gcloud compute scp \
    BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad \
    bonemarrow-training:~/bonemarrow_training/ \
    --zone=us-central1-a

# Upload training script
gcloud compute scp \
    run_bonemarrowmap_full_gcp.py \
    bonemarrow-training:~/bonemarrow_training/ \
    --zone=us-central1-a

# Upload decipher-batch-correction code
gcloud compute scp \
    --recurse \
    ../decipher-batch-correction/decipher-bc \
    bonemarrow-training:~/bonemarrow_training/ \
    --zone=us-central1-a
```

### Step 3: Install Dependencies

```bash
# SSH into the instance
gcloud compute ssh bonemarrow-training --zone=us-central1-a

# Navigate to working directory
cd ~/bonemarrow_training

# Install Python packages
pip install scanpy torch pyro-ppl matplotlib seaborn scikit-learn

# Verify GPU is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Should output:
# CUDA available: True
# GPU: Tesla T4
```

### Step 4: Run Training

```bash
# Still in SSH session on the VM
cd ~/bonemarrow_training

# Run training (takes ~1-1.5 hours)
python3 run_bonemarrowmap_full_gcp.py --epochs 150 2>&1 | tee training.log

# You can monitor progress in real-time:
# - Preprocessing: ~10-15 minutes
# - Training: ~45-60 minutes (with GPU)
# - Post-processing (UMAP, saving): ~5-10 minutes
```

**What happens**:
1. **Preprocessing** (10-15 min):
   - Loads full 263k × 34k dataset into memory
   - Normalizes and computes HVGs (Seurat v3 with batch correction)
   - Filters to top 5k genes
   - Saves `BoneMarrowMap_full_5k_genes.h5ad`

2. **Training** (45-60 min with GPU):
   - Trains Decipher-BC model (beta=1.0, 2 attention heads)
   - 150 epochs with early stopping
   - Batch size 512

3. **Post-processing** (5-10 min):
   - Extracts embeddings (z, v spaces)
   - Computes attention weights
   - Computes UMAP
   - Saves results

**To monitor from another terminal**:
```bash
# In a new terminal on your local machine
gcloud compute ssh bonemarrow-training --zone=us-central1-a
tail -f ~/bonemarrow_training/training.log
```

### Step 5: Download Results

```bash
# Exit SSH session (Ctrl+D or type 'exit')

# Download results to local machine
mkdir -p Full_Beta_1.0_AttentionHeads_2_Training_GCP

gcloud compute scp \
    --recurse \
    bonemarrow-training:~/bonemarrow_training/Full_Beta_1.0_AttentionHeads_2_Training/* \
    ./Full_Beta_1.0_AttentionHeads_2_Training_GCP/ \
    --zone=us-central1-a
```

**Files you'll get**:
- `bonemarrowmap_full_150epochs_beta1_attnheads2.h5ad` (~600-800 MB)
- `bonemarrowmap_full_150epochs_loss_curves.png`

### Step 6: Clean Up (IMPORTANT!)

```bash
# Delete the instance to stop charges
gcloud compute instances delete bonemarrow-training \
    --zone=us-central1-a \
    --quiet
```

**⚠️ Important**: Make sure to delete the instance when done, or you'll continue to be charged!

---

## Alternative: Skip Preprocessing

If you've already preprocessed the data on GCP once, you can skip it on subsequent runs:

```bash
# On the VM
python3 run_bonemarrowmap_full_gcp.py --epochs 150 --skip-preprocessing
```

This assumes `BoneMarrowMap_full_5k_genes.h5ad` already exists.

---

## Cost Breakdown

| Component | Time | Cost |
|-----------|------|------|
| **Instance (n1-highmem-16)** | 1.5 hours | ~$1.35 |
| **GPU (T4)** | 1.5 hours | ~$0.60 |
| **Storage (100 GB)** | 1.5 hours | ~$0.01 |
| **Network egress** | ~1 GB | ~$0.12 |
| **Total** | | **~$2-3** |

**Tips to save money**:
- Delete instance immediately after downloading results
- Use preemptible instances (50% cheaper but can be interrupted)
- Run during off-peak hours if using preemptible

---

## Troubleshooting

### Out of Memory During Preprocessing

**Error**: `numpy._core._exceptions._ArrayMemoryError: Unable to allocate X GiB`

**Solution**: You're not using the high-memory instance. Make sure you created the VM with `--machine-type=n1-highmem-16` (104 GB RAM).

### GPU Not Available

**Error**: Training is very slow or `CUDA available: False`

**Solution**:
1. Check GPU is attached: `nvidia-smi`
2. If not, the GPU driver may not be installed
3. Try recreating the instance with `--metadata="install-nvidia-driver=True"`

### Upload is Too Slow

**Solution**:
- Use `gcloud compute scp` with `--compress` flag
- Upload from a location with better internet
- Alternatively, upload to Google Cloud Storage first, then download from VM (faster):
  ```bash
  # Upload to GCS (faster)
  gsutil cp BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad gs://your-bucket/

  # On VM, download from GCS (much faster)
  gsutil cp gs://your-bucket/BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad .
  ```

### Training Fails with "Killed"

**Error**: Process killed during training

**Possible causes**:
1. **OOM during training**: Reduce batch size in the script (512 → 256)
2. **Preemptible instance preempted**: Use regular instance instead
3. **Out of disk space**: Increase boot disk size to 200GB

---

## Monitoring Costs

Check your current usage:
```bash
# View running instances
gcloud compute instances list

# Estimate cost
# n1-highmem-16: ~$0.90/hour
# T4 GPU: ~$0.40/hour
# Total: ~$1.30/hour
```

---

## Next Steps After Training

Once you've downloaded the results:

1. **Load and visualize**:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad('Full_Beta_1.0_AttentionHeads_2_Training_GCP/bonemarrowmap_full_150epochs_beta1_attnheads2.h5ad')

   # Plot UMAP colored by batch
   sc.pl.umap(adata, color='Donor', use_raw=False, legend_loc='on data')

   # Plot attention strength
   sc.pl.umap(adata, color='batch_attention_strength', use_raw=False)
   ```

2. **Compare with small/medium results**:
   - Load all three datasets
   - Compare batch mixing metrics
   - Compare attention patterns
   - Validate that scaling up improves results

3. **Downstream analysis**:
   - Differential expression
   - Trajectory inference in V space
   - Cell type refinement

---

## Questions?

If you encounter issues:
1. Check the training.log file for error messages
2. Verify you're using the correct machine type (n1-highmem-16)
3. Ensure GPU is available with `nvidia-smi`
4. Check GCP quotas (you may need to request GPU quota increase)
