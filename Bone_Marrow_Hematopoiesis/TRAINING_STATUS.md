# BoneMarrowMap Training Status

## Current Status

### ✅ Running Now: Medium Dataset Training (Local)

**Started**: Just now
**Expected completion**: ~1.5-2.5 hours from start
**Progress**: Epoch 10/150 (training started successfully)

**Configuration**:
- Dataset: 85,792 cells × 5,000 genes
- Beta: 1.0
- Attention heads: 2
- Batch size: 256
- Device: CPU

**Output location**: `Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_medium_150epochs_beta1_attnheads2.h5ad`

**Monitoring**:
```bash
# Check progress
tail -f Bone_Marrow_Hematopoiesis/medium_training_log.txt

# Or use the task output command
# Task ID: b7df0b6
```

---

## Next: Full Dataset Training (GCP)

### Ready to Run: GCP Scripts Created ✅

I've created three files to help you run the full dataset (263k cells) on GCP:

1. **[run_bonemarrowmap_full_gcp.py](run_bonemarrowmap_full_gcp.py)** - Training script optimized for GCP
2. **[gcp_setup_and_run.sh](gcp_setup_and_run.sh)** - Automated setup script
3. **[GCP_TRAINING_GUIDE.md](GCP_TRAINING_GUIDE.md)** - Complete step-by-step guide

### Quick Start (GCP)

```bash
# 1. Edit the automated script to set your GCP project ID
nano gcp_setup_and_run.sh
# Change: PROJECT_ID="your-gcp-project-id"

# 2. Run the automated script
bash gcp_setup_and_run.sh
```

**What it does**:
- Creates VM with GPU (n1-highmem-16 + T4 GPU)
- Uploads data (3.5 GB) and code
- Installs dependencies
- Preprocesses full dataset (263k cells → 5k genes)
- Trains model (~1-1.5 hours total)
- Downloads results
- Deletes VM

**Expected cost**: $2-3 total

### Manual Setup

See [GCP_TRAINING_GUIDE.md](GCP_TRAINING_GUIDE.md) for detailed step-by-step instructions.

---

## Training Comparison

| Dataset | Cells | Genes | Status | Time (CPU) | Time (GPU) | Location |
|---------|-------|-------|--------|------------|------------|----------|
| **Small** | 20k | 5k | ✅ Complete | ~10 min | ~3 min | Local (done) |
| **Medium** | 86k | 5k | 🏃 Running | ~2 hours | ~30 min | Local (in progress) |
| **Full** | 263k | 5k | 📋 Ready | ~3-4 hours | ~1 hour | GCP (scripts ready) |

---

## Why GCP for Full Dataset?

**Memory requirements**:
- Local machine: 7.2 GB RAM ❌
- Full dataset preprocessing: ~50-60 GB RAM needed ✅ (GCP n1-highmem-16 has 104 GB)

**Speed benefits**:
- Local CPU: ~3-4 hours
- GCP with T4 GPU: ~1 hour (preprocessing + training)

**Cost**:
- ~$2-3 for complete run (preprocessing + training + download)
- Auto-cleanup prevents ongoing charges

---

## Files Created

### Local Training
- ✅ `run_bonemarrowmap_beta1_attnheads2.py` - Small/medium dataset training
- ✅ `run_bonemarrowmap_full_beta1_attnheads2.py` - Full dataset (local, requires preprocessing)

### GCP Training
- ✅ `run_bonemarrowmap_full_gcp.py` - Full dataset with integrated preprocessing
- ✅ `gcp_setup_and_run.sh` - Automated GCP setup and execution
- ✅ `GCP_TRAINING_GUIDE.md` - Complete manual instructions

---

## Next Steps

### While Medium Dataset Trains (Local)

1. **Set up GCP** (if you haven't already):
   ```bash
   # Install gcloud CLI
   # https://cloud.google.com/sdk/docs/install

   # Authenticate
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Review the GCP guide**:
   - Read `GCP_TRAINING_GUIDE.md`
   - Decide: automated script vs manual setup
   - Check your GCP quota for GPUs (you may need to request T4 access)

3. **Prepare to run on GCP**:
   - Edit `gcp_setup_and_run.sh` with your project ID
   - When ready, just run: `bash gcp_setup_and_run.sh`

### After Medium Training Completes

1. **Analyze medium results**:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad('Beta_1.0_AttentionHeads_2_Training/bonemarrowmap_medium_150epochs_beta1_attnheads2.h5ad')

   # Plot results
   sc.pl.umap(adata, color='Donor')  # Check batch mixing
   sc.pl.umap(adata, color='batch_attention_strength')  # Attention patterns
   ```

2. **Compare with small dataset** to verify scaling is working

3. **Launch GCP training** if medium results look good

---

## Monitoring

### Medium Dataset (Running Now)

```bash
# Check log file
tail -f Bone_Marrow_Hematopoiesis/medium_training_log.txt

# Expected output:
# Epoch 1/150 - Train Loss: ~300000, Val Loss: ~275000
# Epoch 10/150 - Train Loss: ~273000, Val Loss: ~270000
# ... (loss should decrease)
# Final epochs - Train Loss: ~220000-240000, Val Loss: ~215000-235000
```

### GCP Training (When Running)

```bash
# SSH into instance
gcloud compute ssh bonemarrow-training --zone=us-central1-a

# Monitor log
tail -f ~/bonemarrow_training/training.log
```

---

## Questions?

- **Medium training taking too long?** Check CPU usage with `top` - should be near 100%
- **GCP access issues?** Make sure billing is enabled and you have GPU quota
- **Upload too slow?** Consider using Google Cloud Storage as intermediate step

---

**Last Updated**: $(date)
