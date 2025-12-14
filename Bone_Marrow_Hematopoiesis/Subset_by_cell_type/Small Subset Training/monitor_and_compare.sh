#!/bin/bash
#
# Monitor training jobs and run comparison automatically
#

cd /home/alan/Documents/decipher_batch_correction/Bone_Marrow_Hematopoiesis/Subset_by_cell_type
source ../../decipher_env/bin/activate

echo "=================================="
echo "MONITORING TRAINING PROGRESS"
echo "=================================="
echo ""
echo "Waiting for all three models to complete training..."
echo ""

# Files to check for
REGULAR="erythroid_regular_decipher.h5ad"
BETA01="erythroid_batch_corrected_beta0.1_heads4_initial.h5ad"
BETA10="erythroid_batch_corrected_beta1.0_heads2.h5ad"

# Monitor until all files exist
while true; do
    ALL_COMPLETE=true
    
    if [ ! -f "$REGULAR" ]; then
        ALL_COMPLETE=false
        echo "[$(date +%H:%M:%S)] Waiting for Regular Decipher..."
    fi
    
    if [ ! -f "$BETA01" ]; then
        ALL_COMPLETE=false
        echo "[$(date +%H:%M:%S)] Waiting for Beta=0.1, Heads=4..."
    fi
    
    if [ ! -f "$BETA10" ]; then
        ALL_COMPLETE=false
        echo "[$(date +%H:%M:%S)] Waiting for Beta=1.0, Heads=2..."
    fi
    
    if [ "$ALL_COMPLETE" = true ]; then
        echo ""
        echo "✓ All three models completed!"
        break
    fi
    
    sleep 60  # Check every minute
done

echo ""
echo "=================================="
echo "RUNNING COMPARISON ANALYSIS"
echo "=================================="
echo ""

# Run comparison for Beta=0.1, Heads=4
echo "Comparing Regular vs Beta=0.1 (4 heads)..."
python 04_compare_and_validate_markers.py \
    --regular "$REGULAR" \
    --batch-corrected "$BETA01"
mv erythroid_comparison_with_markers.png erythroid_comparison_beta0.1_heads4.png
mv erythroid_comparison_metrics.csv erythroid_comparison_beta0.1_heads4_metrics.csv

echo ""
echo "Comparing Regular vs Beta=1.0 (2 heads)..."
python 04_compare_and_validate_markers.py \
    --regular "$REGULAR" \
    --batch-corrected "$BETA10"
mv erythroid_comparison_with_markers.png erythroid_comparison_beta1.0_heads2.png
mv erythroid_comparison_metrics.csv erythroid_comparison_beta1.0_heads2_metrics.csv

echo ""
echo "=================================="
echo "✓ ALL ANALYSIS COMPLETE!"
echo "=================================="
echo ""
echo "Results:"
echo "  1. $REGULAR"
echo "  2. $BETA01"
echo "  3. $BETA10"
echo "  4. erythroid_comparison_beta0.1_heads4.png"
echo "  5. erythroid_comparison_beta1.0_heads2.png"
echo "  6. erythroid_comparison_beta0.1_heads4_metrics.csv"
echo "  7. erythroid_comparison_beta1.0_heads2_metrics.csv"
echo ""
echo "You can now review the comparison plots to see:"
echo "  - V-space structure preservation"
echo "  - Batch mixing improvements"
echo "  - Marker gene expression patterns (HBB, GYPA, etc.)"
echo ""
