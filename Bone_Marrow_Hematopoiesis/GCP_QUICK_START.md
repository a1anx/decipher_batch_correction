# GCP Quick Start - 5 Minutes to Launch

## Prerequisites
- GCP account with billing enabled
- gcloud CLI installed

## Step 1: Set Your Project ID (30 seconds)

```bash
cd Bone_Marrow_Hematopoiesis

# Edit the script
nano gcp_setup_and_run.sh

# Change this line:
PROJECT_ID="your-gcp-project-id"  # CHANGE THIS
# To:
PROJECT_ID="your-actual-project-id"

# Save and exit (Ctrl+X, Y, Enter)
```

## Step 2: Run (1 command!)

```bash
bash gcp_setup_and_run.sh
```

That's it! The script will:
- ✅ Create VM with GPU
- ✅ Upload data (3.5 GB, ~10 min)
- ✅ Install dependencies
- ✅ Preprocess data (~15 min)
- ✅ Train model (~45 min)
- ✅ Download results
- ✅ Delete VM

**Total time**: ~1.5 hours
**Cost**: ~$2-3
**Your involvement**: Edit 1 line, run 1 command, wait

## Monitoring (Optional)

While it's running, you can watch progress:

```bash
# In another terminal
gcloud compute ssh bonemarrow-training --zone=us-central1-a
tail -f ~/bonemarrow_training/training.log
```

## Manual Alternative

Don't trust automation? See `GCP_TRAINING_GUIDE.md` for step-by-step manual instructions.

## Troubleshooting

**"Project not found"**: Make sure you set the correct PROJECT_ID in the script

**"Quota exceeded"**: Request T4 GPU quota increase in GCP Console

**"gcloud not found"**: Install from https://cloud.google.com/sdk/docs/install

**Upload slow**: Normal for 3.5 GB file, takes ~10-15 min on average connection
