Code for MSc Thesis collaboration with Novo Nordsik Engineering MIT department "Generating documentation with verified quotes using fine-tuned large language models"

## Dependencies:
- https://github.com/microsoft/controllable-safety-alignment
- https://github.com/JHU-CLSP/verifiable-by-design
- https://github.com/ruyimarone/data-portraits 

Make sure to correctly set up environment variables before running all scripts:
- `PROJ_DIR` and `PYTHONPATH` should be the project directory of the `controllable-safety-alignment` repo
- `MODEL_DIR`, `DATA_DIR`: dedicated directory to store *downloaded* models and data
- `OUTPUT_DIR`: dedicated directory to store *trained* checkpoints

## Steps for running the Quote-Tuning pipeline
1. Format data into HuggingFace dataset

Example: `data/nq_NNE/dev`, `data/nq_NNE/train`
```
Dataset({
    features: ['prompt', 'reference'],
    num_rows: 1200
})
```

2. Start the vLLM server by using `start_vllm.sh` to generate responses on training and dev (test) data.

Example: `run_gen_bo32.sh`

3. Set up the QUIP score server (or use the existing one accessible on the internet, 'https://acc2-private-wiki.dataportraits.org/quip'). 

Example: `run_gen_bo32.sh`

4. Use `run_metric_on_gen.py` to score responses with the QUIP score. During this step, the QUIP server should be up and running.


5. Use `best_of_n_to_paired_gen.py` to produce paired data for DPO. Then, convert the produced .json into a HuggingFace dataset via `data_processing/convert_paired_gens_json_to_dataset.py`. 


6. Add path of the paired data (converted to HF dataset) as a dataset in the `PAIRED_DATA_DICT` of `$PROJ_DIR/dpo/preference_datasets.py`. 


7. Conduct DPO training! Use `$PROJ_DIR/dpo/train_qt.sh`. This part of code is based on `https://github.com/eric-mitchell/direct-preference-optimization`. After training, Convert the trained model .pt file back to huggingface format using `checkpoint_pt_to_hf.py`.

8. Run evaluation using `run_eval_combined.sh`.
