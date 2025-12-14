#!/bin/bash
#
# Full Erythroid Lineage Analysis Pipeline
#
# This script runs the complete workflow:
# 1. Subset to erythroid lineage
# 2. Train Regular Decipher
# 3. Train Batch-Corrected Decipher (Beta=1.0, 2 heads)
# 4. Compare results and validate with marker genes
#
# Usage:
#   bash run_full_analysis.sh
#

set -e  # Exit on error

echo "=============================================================================="
echo "ERYTHROID LINEAGE ANALYSIS - FULL PIPELINE"
echo "=============================================================================="

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source ../../decipher_env/bin/activate

# Step 1: Subset data
echo ""
echo "=============================================================================="
echo "STEP 1: Subsetting to Erythroid Lineage"
echo "=============================================================================="
python 01_subset_erythroid_lineage.py

# Step 2: Train Regular Decipher
echo ""
echo "=============================================================================="
echo "STEP 2: Training Regular Decipher (Baseline)"
echo "=============================================================================="
python 02_train_regular_decipher_erythroid.py --epochs 150

# Step 3: Train Batch-Corrected Decipher
echo ""
echo "=============================================================================="
echo "STEP 3: Training Batch-Corrected Decipher (Beta=1.0, 2 heads)"
echo "=============================================================================="
python 03_train_batch_corrected_erythroid.py --beta 1.0 --heads 2 --epochs 150

# Step 4: Compare and validate
echo ""
echo "=============================================================================="
echo "STEP 4: Comparing Models and Validating with Markers"
echo "=============================================================================="
python 04_compare_and_validate_markers.py \
    --regular erythroid_regular_decipher.h5ad \
    --batch-corrected erythroid_batch_corrected_beta1.0_heads2.h5ad

echo ""
echo "=============================================================================="
echo "✓ PIPELINE COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results saved:"
echo "  1. BoneMarrowMap_Erythroid_5kgenes.h5ad - Subset data"
echo "  2. erythroid_regular_decipher.h5ad - Regular Decipher results"
echo "  3. erythroid_batch_corrected_beta1.0_heads2.h5ad - Batch-corrected results"
echo "  4. erythroid_comparison_with_markers.png - Visual comparison"
echo "  5. erythroid_comparison_metrics.csv - Quantitative metrics"
echo ""
echo "Next steps:"
echo "  - Review erythroid_comparison_with_markers.png"
echo "  - Check if marker genes (HBB, GYPA) show biological gradients"
echo "  - Verify developmental trajectory is preserved"
echo "  - Compare batch mixing improvements"
echo ""
