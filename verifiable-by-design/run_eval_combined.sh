#!/bin/bash
eval "$(conda shell.bash hook)"
#conda activate vllm
export CUDA_VISIBLE_DEVICES=1
export PROJ_DIR=/scratch/akranjcevic/controllable-safety-alignment
export PYTHONPATH=/scratch/akranjcevic/controllable-safety-alignment
export MODEL_DIR=/scratch/akranjcevic/models

##################################################################

DATAPATH="data/nq/dev"
#DATAPATH="data/nq_NNE/dev1"
##################################################################

PROMPT_TEMPLATE_NAME="concise_gr"
MODEL="/scratch/akranjcevic/models/F_Wiki-Phi20_90-10delta35r2/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85"
MODEL_DISPLAY_NAME="F_Wiki-Phi20_90-10delta35r2"
##################################################################

# STEP 1: Generate on eval data
# NOTE: need to start vllm server before running this

ts=$(date +%F_%T)
echo "Starting inference at $ts"
# for DATAPATH in "${DATAPATHS[@]}"
# do
echo "DATAPATH: $DATAPATH"
conda run -n vllm python $PROJ_DIR/src/oai_inference.py \
    ${@} \
    --model $MODEL --model_display_name $MODEL_DISPLAY_NAME \
    --input_dataset $DATAPATH \
    -p prompt_templates/$PROMPT_TEMPLATE_NAME.json \
    | tee -a "logs/oai_inference_$ts.log" # --kwargs $PROJ_DIR/src/kwargs/p095.json \
# done
echo "log file: logs/oai_inference_$ts.log"

##################################################################

# STEP 2: Evaluate generated data
# NOTE: REQUIRES GPU; make sure to STOP VLLM SERVER to clear vllm GPU memory before running this part.

OUTPATH=${DATAPATH}_model-${MODEL_DISPLAY_NAME}_${PROMPT_TEMPLATE_NAME}

export PYTHONPATH=$PWD
conda activate vllm

# # conda run -n py310 
python run_metric_on_gen.py \
    $OUTPATH \
    --modes quip 

# conda run -n py310 
python evaluation.py \
    "${OUTPATH}_quip_bartscore.json" \
    --tokenizer $MODEL \
    --nq_short_answer_acc \
    --ppl $MODEL_DIR/F_Wiki-Phi20_90-10delta35r2/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85