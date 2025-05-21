#!/bin/bash

# Directory containing subfolders
OUTPUTS_DIR="outputs"
RESULTS_DIR="results"
INFERENCE_SCRIPT="src/open_r1/inference.py"
DATASET_PATH="data/all_test_dataset"

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Loop over sub-subfolders, e.g. outputs/SomeModel/checkpoint-xxx
for MODEL_DIR in "${OUTPUTS_DIR}"/*/*; do

  # Make sure it's a directory
  if [ -d "${MODEL_DIR}" ]; then
    # Check if directory is non-empty
    if [ "$(ls -A "${MODEL_DIR}")" ]; then
      
      # Create an output filename based on the second-level and third-level folder names.
      # e.g. open-r1/outputs/Qwen2.5-1.5B-Instruct_qwenCOT_512/checkpoint-730
      #      => output_csv = open-r1/results/Qwen2.5-1.5B-Instruct_qwenCOT_512_checkpoint-730.csv
      PARENT_NAME=$(basename "$(dirname "${MODEL_DIR}")")
      SUB_NAME=$(basename "${MODEL_DIR}")
      OUTPUT_CSV="${RESULTS_DIR}/${PARENT_NAME}_${SUB_NAME}.csv"

      # If the result file already exists, skip
      if [ -f "${OUTPUT_CSV}" ]; then
        echo "Result file ${OUTPUT_CSV} already exists. Skipping..."
        continue
      fi
      
      # Otherwise, run the inference script
      echo "Running inference for model directory: ${MODEL_DIR}"
      python "${INFERENCE_SCRIPT}" \
        --dataset_path "${DATASET_PATH}" \
        --model_name "${MODEL_DIR}" \
        --output_csv "${OUTPUT_CSV}"
      
    fi
  fi
done

echo "Done."