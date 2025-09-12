#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate vllm
export OUTLINES_CACHE_DIR=/scratch/akranjcevic/output/.outlines
export PROJ_DIR=/scratch/akranjcevic/controllable-safety-alignment
export MODEL_DIR=/scratch/akranjcevic/models
export DATA_DIR=$PWD/data
export OUTPUT_DIR=$PWD/dpo/runs
export PYTHONPATH=/scratch/akranjcevic/controllable-safety-alignment

# BUG FIX: If "too many open files": add --disable-frontend-multiprocessing (https://github.com/vllm-project/vllm/issues/7290)

model="llama3.1-8b-instruct"
#model="Wiki-LLAMA20_90-10DELTAtest35"
#model="phi-3-mini-4k-instruct-ORIG"
#model="F_Wiki-Phi20_90-10delta35r2"

size=2
port=8000

model_path=$(find "$MODEL_DIR/$model" -name config.json -type l -printf '%h\n' -quit)
echo "Starting vLLM API server with model $MODEL_DIR/$model on port $port with tensor parallel size $size"
python -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --port $port \
    --dtype auto \
    --trust-remote-code \
    --api-key token-abc123 \
    --tensor-parallel-size $size \
    ${@:3}
