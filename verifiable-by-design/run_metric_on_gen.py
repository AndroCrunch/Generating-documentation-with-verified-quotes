import sys
import os
import argparse
from typing import List

# Fix for local module imports
sys.path.append(os.path.join(os.path.dirname(__file__), "BARTScore"))

from jack_utils import load_json, save_json, add_suffix_before_ext, flatten_lol
from quip_api import batch_quip_api
from bart_score import BARTScorer

import numpy as np
from tqdm import tqdm
import evaluate
import torch
from datasets import load_from_disk

# ── Instantiate BART-base ONCE (no .load(...) call here) ──
# Changed from "facebook/bart-large-cnn" to "facebook/bart-base" to reduce memory footprint
GLOBAL_BART = BARTScorer(device="cpu", checkpoint="facebook/bart-base")


def load_data(gen_file: str):
    if gen_file.endswith(".json"):
        return load_json(gen_file)
    else:
        # assume it's a HuggingFace dataset
        gen = load_from_disk(gen_file)
        pred = gen["response"]
        if isinstance(pred[0], str):
            pred = [[p] for p in pred]
        return {
            "pred": pred,
            "refs": gen["reference"],
            "args": {"prompt_before": "", "prompt_after": ""},
            "full_prompts": gen["prompt"],
        }


def metric_name_mapping(metric: str) -> str:
    if metric == "bartscore":
        return "bartscore"
    elif metric == "quip":
        return "quip_reports"
    elif metric.startswith("ppl_"):
        return metric
    elif metric == "mauve":
        return metric
    else:
        return None


def run_quip(ctx, preds, refs, batch_size=64, simple_report_format=True, **kwargs):
    print("Running QUIP API")
    return batch_quip_api(preds, batch_size=batch_size, simple_report_format=simple_report_format, **kwargs)


def run_bartscore(ctx, preds, refs, batch_size=32, **kwargs):
    # Reuse the single GLOBAL_BART instance (no .load call here)
    min_ref = min(len(r) for r in refs)
    refs = [r[:min_ref] for r in refs]
    print("Calculating BARTScore, #refs:", len(refs[0]))
    return GLOBAL_BART.multi_ref_score(preds, refs, agg="max", batch_size=batch_size)


def run_ppl(ctx, preds, refs, batch_size=4, **kwargs):
    assert "model" in kwargs
    perp = evaluate.load("./cond_perplexity.py", module_type="measurement")
    res = perp.compute(
        context=ctx,
        continuation=preds,
        model_id=kwargs["model"],
        batch_size=batch_size,
        torch_dtype=torch.bfloat16,
    )
    print(f"mean perplexity on {kwargs['model']}: {res['mean_perplexity']}")
    return res["perplexities"]


def run_mauve(ctx, preds, refs, **kwargs):
    m = evaluate.load("mauve")
    out = m.compute(predictions=preds, references=[r[0] for r in refs], device_id=0)
    return out.mauve


def run_metric_on_gen_batch(gen_file: str, metrics: List[str], suffix: str, start_idx: int, end_idx: int):
    gen = load_data(gen_file)

    total = len(gen["pred"])
    end = end_idx if end_idx is not None else total

    print(f"Processing examples from index {start_idx} to {end} (total {total})")

    gen["pred"] = gen["pred"][start_idx:end]
    gen["refs"] = gen["refs"][start_idx:end]
    gen["full_prompts"] = gen["full_prompts"][start_idx:end]

    filepath = gen_file + ("" if gen_file.endswith(".json") else ".json")

    for metric in metrics:
        n = len(gen["pred"][0])
        # flatten
        pred_flat = [p for ps in gen["pred"] for p in ps]
        ref_flat = flatten_lol([[r] * n for r in gen["refs"]])
        prompt_flat = flatten_lol([[pr] * n for pr in gen["full_prompts"]])
        ctx = [gen["args"]["prompt_before"] + p + gen["args"]["prompt_after"] for p in prompt_flat]

        kwargs = {}
        if metric == "quip":
            fn = run_quip
        elif metric == "bartscore":
            fn = run_bartscore
        elif metric.startswith("ppl_"):
            fn = run_ppl
            kwargs["model"] = "_".join(metric.split("_")[1:])
        elif metric == "mauve":
            fn = run_mauve
        else:
            raise NotImplementedError(metric)

        scores = fn(ctx, pred_flat, ref_flat, **kwargs)
        gen[metric_name_mapping(metric)] = [scores[i : i + n] for i in range(0, len(scores), n)]

        filepath = add_suffix_before_ext(filepath, f"_{metric}")

    filepath = add_suffix_before_ext(filepath, suffix)
    print(f"Saving to {filepath}")
    save_json(gen, filepath)


def check_modes(modes):
    for m in modes:
        if metric_name_mapping(m) is None:
            raise ValueError(f"unknown metric {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes", "-m", nargs="+", required=True, help="metric mode(s)"
    )
    parser.add_argument(
        "--outname_suffix", "-s", default="", help="suffix for output file"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="first example (inclusive)"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="one-past-last example"
    )
    parser.add_argument(
        "--num_parts", type=int, default=None, help="split the dataset into this many equal parts"
    )
    parser.add_argument(
        "--part_idx", type=int, default=None, help="which part (1-based) to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for QUIP API calls"
    )
    parser.add_argument("input_files", nargs="+", help="data-folder or .json file")
    args = parser.parse_args()

    check_modes(args.modes)

    for infile in tqdm(args.input_files, desc="Files"):
        print("Processing", infile)

        # decide start/end indices
        if args.num_parts is not None and args.part_idx is not None:
            if args.part_idx < 1 or args.part_idx > args.num_parts:
                raise ValueError(f"part_idx must be in [1, {args.num_parts}]")
            data = load_data(infile)
            total = len(data["pred"])
            size = total // args.num_parts
            start = size * (args.part_idx - 1)
            end = total if args.part_idx == args.num_parts else size * args.part_idx
        else:
            start = args.start_idx or 0
            end = args.end_idx

        run_metric_on_gen_batch(infile, args.modes, args.outname_suffix, start, end)
