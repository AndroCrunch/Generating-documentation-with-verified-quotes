#!/usr/bin/env bash
# env_setup.sh — set up all the env vars for Quote-Tuning on the HPC

# 1) Where your code lives
export PROJ_DIR="$HOME/path/to/verifiable-by-design"
export PYTHONPATH="$PROJ_DIR"

# 2) Where all your LLM weights live
export MODEL_DIR="/scratch/akranjcevic/models"

# 3) Which sub-folder under MODEL_DIR you want
export model="llama3.1-8b-instruct"

# 4) Scratch-dir for model path lookup (auto-finds config.json parent dir)
export model_path=$(find "$MODEL_DIR/$model" -name config.json -printf '%h\n' -quit)

# 5) Where your quote-tuning HF dataset lives
export DATA_DIR="$PROJ_DIR/data/hf_nq_for_quote_tuning"

# 6) Where you want outputs/checkpoints
export OUTPUT_DIR="/scratch/akranjcevic/outputs/verifiable-by-design"
mkdir -p "$OUTPUT_DIR"

# Sanity check
echo "PROJ_DIR      = $PROJ_DIR"
echo "PYTHONPATH    = $PYTHONPATH"
echo "MODEL_DIR     = $MODEL_DIR"
echo "model         = $model"
echo "model_path    = $model_path"
echo "DATA_DIR      = $DATA_DIR"
echo "OUTPUT_DIR    = $OUTPUT_DIR"
