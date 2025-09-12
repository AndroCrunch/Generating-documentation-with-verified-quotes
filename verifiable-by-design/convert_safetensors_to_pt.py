import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Function to convert safetensors to .pt format
def convert_safetensors_to_pt(model_dir, model_name, output_dir):
    # Load the model configuration and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load the safetensors
    safetensors_dir = model_dir
    model_weights = {}
    for filename in sorted(os.listdir(safetensors_dir)):
        if filename.endswith('.safetensors'):
            safetensors_path = os.path.join(safetensors_dir, filename)
            model_weights.update(load_file(safetensors_path, device="cpu"))

    # Load the model weights
    model.load_state_dict(model_weights)
    
    # Save the model as .pt
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    model.save_pretrained(output_dir)  # Hugging Face save
    torch.save(model.state_dict(), f"{output_dir}/pytorch_model.pt")  # PyTorch .pt save

    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model successfully converted to .pt and saved to {output_dir}")

# Paths (adjusted according to your setup)
model_dir = "/scratch/akranjcevic/verifiable-by-design/outputs/phi-3-mini-4k-dpo-gpu/akranjcevic/phi-3-mini-4k-instruct-DPO-qt_phi-3-mini-4k-instruct_bo32_dq0.10_dl0.10-concise_nosys-GPU_2025-06-11_17-32-35_528910/LATEST"
model_name = "/scratch/akranjcevic/models/phi-3-mini-4k-instruct/snapshots/0a67737cc96d2554230f90338b163bc6380a2a85"
output_dir = "/scratch/akranjcevic/verifiable-by-design/outputs/phi-3-mini-4k-dpo-gpu/akranjcevic/phi-3-mini-4k-instruct-DPO-qt_phi-3-mini-4k-instruct_bo32_dq0.10_dl0.10-concise_nosys-GPU_2025-06-11_17-32-35_528910/LATEST"

# Convert the model
convert_safetensors_to_pt(model_dir, model_name, output_dir)
