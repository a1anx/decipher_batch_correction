#!/bin/bash
#
# GCP Setup and Training Script for BoneMarrowMap Full Dataset
#
# This script:
# 1. Creates a GCP VM instance with GPU
# 2. Uploads data and code
# 3. Installs dependencies
# 4. Runs training
# 5. Downloads results
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - GCP project with billing enabled
# - Compute Engine API enabled
#
# Usage:
#   bash gcp_setup_and_run.sh
#

set -e  # Exit on error

# Configuration
PROJECT_ID="your-gcp-project-id"  # CHANGE THIS
ZONE="us-central1-a"  # Good for GPU availability
INSTANCE_NAME="bonemarrow-training"
MACHINE_TYPE="n1-highmem-16"  # 16 vCPUs, 104 GB RAM (for preprocessing)
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

echo "=========================================="
echo "GCP BoneMarrowMap Training Setup"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ ERROR: gcloud CLI not found"
    echo "   Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "❌ ERROR: Not authenticated with gcloud"
    echo "   Run: gcloud auth login"
    exit 1
fi

# Verify project ID is set
if [ "$PROJECT_ID" = "your-gcp-project-id" ]; then
    echo "❌ ERROR: Please set PROJECT_ID in this script"
    echo "   Edit gcp_setup_and_run.sh and set your GCP project ID"
    exit 1
fi

gcloud config set project $PROJECT_ID

echo ""
echo "Step 1: Creating GCP VM instance..."
echo "  Instance: $INSTANCE_NAME"
echo "  Machine: $MACHINE_TYPE (16 vCPUs, 104 GB RAM)"
echo "  GPU: $GPU_TYPE"
echo "  Zone: $ZONE"
echo ""

# Create instance
gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform

echo "✓ Instance created!"
echo "  Waiting 60 seconds for instance to fully boot..."
sleep 60

echo ""
echo "Step 2: Uploading data and code..."

# Create remote directory
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="mkdir -p ~/bonemarrow_training"

# Upload dataset (3.5 GB - will take a few minutes)
echo "  Uploading dataset (3.5 GB, this will take 5-15 minutes)..."
gcloud compute scp \
    BoneMarrowMap_Annotated_Dataset_expandedFeatures.h5ad \
    $INSTANCE_NAME:~/bonemarrow_training/ \
    --zone=$ZONE

# Upload code
echo "  Uploading training scripts..."
gcloud compute scp \
    run_bonemarrowmap_full_gcp.py \
    $INSTANCE_NAME:~/bonemarrow_training/ \
    --zone=$ZONE

# Upload decipher-batch-correction code
echo "  Uploading decipher-batch-correction code..."
gcloud compute scp \
    --recurse \
    ../decipher-batch-correction/decipher-bc \
    $INSTANCE_NAME:~/bonemarrow_training/ \
    --zone=$ZONE

echo "✓ Upload complete!"

echo ""
echo "Step 3: Installing dependencies..."

gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
set -e
cd ~/bonemarrow_training

# Install Python packages
pip install --quiet scanpy torch pyro-ppl matplotlib seaborn scikit-learn

# Verify CUDA is available
python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}\")'

echo '✓ Dependencies installed!'
"

echo ""
echo "Step 4: Running training..."
echo "  This will take ~1-1.5 hours (preprocessing + training)"
echo "  You can monitor progress by SSHing into the instance:"
echo "    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "    tail -f ~/bonemarrow_training/training.log"
echo ""

# Run training (this blocks until complete)
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
set -e
cd ~/bonemarrow_training

# Run training with logging
python3 run_bonemarrowmap_full_gcp.py --epochs 150 2>&1 | tee training.log

echo ''
echo '✓ Training complete!'
echo 'Outputs saved in Full_Beta_1.0_AttentionHeads_2_Training/'
ls -lh Full_Beta_1.0_AttentionHeads_2_Training/
"

echo ""
echo "Step 5: Downloading results..."

# Create local output directory
mkdir -p ./Full_Beta_1.0_AttentionHeads_2_Training_GCP

# Download results
gcloud compute scp \
    --recurse \
    $INSTANCE_NAME:~/bonemarrow_training/Full_Beta_1.0_AttentionHeads_2_Training/* \
    ./Full_Beta_1.0_AttentionHeads_2_Training_GCP/ \
    --zone=$ZONE

echo "✓ Results downloaded to: ./Full_Beta_1.0_AttentionHeads_2_Training_GCP/"

echo ""
echo "Step 6: Cleaning up..."
echo "  Deleting instance to stop charges..."

gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet

echo "✓ Instance deleted!"

echo ""
echo "=========================================="
echo "✓ COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  ./Full_Beta_1.0_AttentionHeads_2_Training_GCP/"
echo ""
echo "Estimated cost: \$2-3"
echo ""
