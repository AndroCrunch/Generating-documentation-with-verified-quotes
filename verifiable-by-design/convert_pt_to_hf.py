#!/usr/bin/env python3
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def convert_pt_to_hf(policy_path: str, base_model_path: str, output_dir: str):
    # Load the policy checkpoint
    ckpt = torch.load(policy_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)

    # Load base config and instantiate model
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config)

    # Load the fine-tuned weights
    model.load_state_dict(state_dict, strict=False)

    # Load tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # Save to Hugging Face format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Converted policy.pt → HF format at: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python convert_pt_to_hf.py <policy.pt> <base_model_dir> <output_dir>")
        sys.exit(1)

    policy_path, base_model_path, output_dir = sys.argv[1:]
    convert_pt_to_hf(policy_path, base_model_path, output_dir)
