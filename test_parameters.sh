#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable section ----
INPUT_FOLDER="./input_images"
RECT="0 0 8000 8000"
PARAM_NAME="--gauss-kernel"         # Parameter you want to vary
PARAM_VALUES=(3 5 7)                # Different values to try
# All other fixed parameters for your run
FIXED_PARAMS="--gauss-sigma 1.0 --canny-th1 60 --canny-th2 120 --dilate-kernel 5 --dilate-iters 2 --dilate-shape ellipse"
# -----------------------------------

# Create timestamped output root folder
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ROOT="./results_${TIMESTAMP}"
mkdir -p "$OUTPUT_ROOT"

echo "Starting runs..."
for VALUE in "${PARAM_VALUES[@]}"; do
    RUN_FOLDER="${OUTPUT_ROOT}/${PARAM_NAME#--}_${VALUE}"
    mkdir -p "$RUN_FOLDER"

    echo "Running with ${PARAM_NAME}=${VALUE}"
    python run_pipeline.py \
        --input "$INPUT_FOLDER" \
        --output "$RUN_FOLDER" \
        --rect $RECT \
        ${PARAM_NAME} "$VALUE" \
        $FIXED_PARAMS

    # Save parameters to a file inside run folder
    {
        echo "Run timestamp: $(date)"
        echo "Input folder: $INPUT_FOLDER"
        echo "Output folder: $RUN_FOLDER"
        echo "Rectangle: $RECT"
        echo "${PARAM_NAME} = $VALUE"
        echo "Fixed params: $FIXED_PARAMS"
    } > "${RUN_FOLDER}/params_used.txt"
done

echo "All runs completed. Results in $OUTPUT_ROOT"
