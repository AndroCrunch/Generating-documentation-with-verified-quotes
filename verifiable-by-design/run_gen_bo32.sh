#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate vllm
export PROJ_DIR=/scratch/akranjcevic/controllable-safety-alignment
export PYTHONPATH=/scratch/akranjcevic/controllable-safety-alignment
export MODEL_DIR=/scratch/akranjcevic/models

##################################################################

#DATAPATHS=("data/nq/train_20pct")
DATAPATHS=("data/nq/dev")
#DATAPATHS=("data/nq_NNE/train")

##################################################################

model="llama3.1-8b-instruct"
#model="phi-3-mini-4k-instruct"
#model="phi-3-mini-4k-instruct-ORIG"
model_path=$(find "$MODEL_DIR/$model" -name config.json -type l -printf '%h\n' -quit)

##################################################################

ts=$(date +%F_%T)
echo "Starting inference at $ts"
for DATAPATH in "${DATAPATHS[@]}"
do
    echo "DATAPATH: $DATAPATH"
    python $PROJ_DIR/src/oai_inference.py \
        ${@} \
        --model $model_path --model_display_name $model \
        --input_dataset $DATAPATH \
        -p prompt_templates/concise_gr_nosys.json \
        --kwargs $PROJ_DIR/src/kwargs/bestof32.json \
        | tee -a "/scratch/akranjcevic/output/logs/oai_inference_$ts.log"
done

echo "log file: /scratch/akranjcevic/output/logs/oai_inference_$ts.log"