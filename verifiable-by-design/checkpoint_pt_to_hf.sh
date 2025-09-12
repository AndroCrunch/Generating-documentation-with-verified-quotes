#!/bin/bash
conda activate vllm

# CKPT_PATH should be provided as an argument when running the script
CKPT_PATH=$1  # e.g., /scratch/akranjcevic/verifiable-by-design/outputs/phi-3-mini-4k-dpo-gpu/akranjcevic/phi-3-mini-4k-instruct-DPO-qt_phi-3-mini-4k-instruct_bo32_dq0.10_dl0.10-concise_nosys-GPU_2025-06-16_19-34-37_687137/LATEST/policy.pt

# Specify the model name corresponding to Phi-3
MODEL_NAME="phi-3-mini-4k-instruct"

# Your Hugging Face token (replace with your actual token)
HF_TOKEN="hf_abc123"  # Replace with your actual Hugging Face token

# Run the conversion from .pt to Hugging Face format using the token for authentication
python /scratch/akranjcevic/verifiable-by-design/checkpoint_pt_to_hf.py \
  $MODEL_NAME \
  $CKPT_PATH \
  --token $HF_TOKEN


