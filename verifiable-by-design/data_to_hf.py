from datasets import Dataset
import json
import os

# Define paths
train_json_path = "/scratch/akranjcevic/verifiable-by-design/data/nq_NNE/train/novo_nordisk_dataset_full_200.json"
train_arrow_dir = "/scratch/akranjcevic/verifiable-by-design/data/nq_NNE/train/"

dev_json_path = "/scratch/akranjcevic/verifiable-by-design/data/nq_NNE/dev/dev_different_questions.json"
dev_arrow_dir = "/scratch/akranjcevic/verifiable-by-design/data/nq_NNE/dev/"

# Function to load JSON and save as arrow dataset
def json_to_arrow(json_path, output_dir):
    # Open and parse each line as individual JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    # Convert to HF Dataset and save
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(output_dir)


# Convert train and dev
json_to_arrow(train_json_path, train_arrow_dir)
json_to_arrow(dev_json_path, dev_arrow_dir)