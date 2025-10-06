#!/bin/bash
#
# Batch training script for all tooth segmentation models with different modes
# Simple shell script version for easy execution
#
# Usage: ./train_all_models.sh [mode]
#   mode: training mode - "classification", "segmentation", or "multi-task" (default: "segmentation")

# Default training mode
DEFAULT_MODE="segmentation"
# DEFAULT_MODE="classification"
# DEFAULT_MODE="multi-task"

# Parse command line arguments
if [ $# -eq 0 ]; then
    MODE="$DEFAULT_MODE"
    echo "🤖 Starting batch training for all models in $MODE mode"
elif [ $# -eq 1 ]; then
    MODE="$1"
    # Validate mode parameter
    case "$MODE" in
        "classification"|"segmentation"|"multi-task")
            echo "🤖 Starting batch training for all models in $MODE mode"
            ;;
        *)
            echo "❌ Error: Invalid mode '$MODE'"
            echo "   Available modes: classification, segmentation, multi-task"
            exit 1
            ;;
    esac
else
    echo "❌ Error: Too many arguments"
    echo "   Usage: $0 [mode]"
    echo "   mode: classification, segmentation, or multi-task"
    exit 1
fi

echo "====================================================="

# List of all available models
models=(
    # "deeplabv3_resnet50"
    # "deeplabv3_resnet101"
    # "fcn_resnet50"
    # "fcn_resnet101"
    # "lraspp_mobilenet_v3_large"
)

echo "📋 Models to train: ${#models[@]}"
for i in "${!models[@]}"; do
    echo "   $((i+1)). ${models[i]}"
done

echo "🎯 Training mode: $MODE"
echo ""

# Track start time
start_time=$(date +%s)

# Train each model
for model in "${models[@]}"; do
    echo "🚀 Training model: $model (mode: $MODE)"
    echo "----------------------------------------"
    
    # Run training command with specified mode and error handling
    if python multi_task_main.py \
        --model "$model" \
        --mode "$MODE"; then
        echo "✅ $model - Training completed successfully!"
    else
        echo "❌ $model - Training failed! Skipping to next model..."
        # Continue to next model even if this one fails
        continue
    fi
    
    echo ""
    
    # Add delay between models (except for the last one)
    if [ "$model" != "${models[-1]}" ]; then
        echo "⏳ Waiting 10 seconds before next model..."
        sleep 10
        echo ""
    fi
done

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))

hours=$((total_time / 3600))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$((total_time % 60))

echo "====================================================="
echo "🎉 Batch training completed!"
echo "⏱️  Total time: ${hours}h ${minutes}m ${seconds}s"
echo "====================================================="