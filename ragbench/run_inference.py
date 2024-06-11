"""
export HF_TOKEN=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
export OPENAI_API_KEY=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
export PYTHONPATH="${PYTHONPATH}:/Users/masha/Documents/galileo/rungalileo/ragbench"

python run_inference.py --dataset delucionqa --model trulens --output results

"""
import os
import json
import argparse
import logging
import asyncio
from typing import Dict, Optional

from datasets import load_dataset
from trulens_async import AsyncTrulensOpenAI
from constants import HUGGINGFACE_REPO_NAME, RAGBenchFields, TrulensFields, RagasFields
from evaluation import calculate_metrics
from inference import trulens_annotate_dataset, ragas_annotate_dataset

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the component dataset to run inference on')
    parser.add_argument('--model', help='Model to use for inference')
    parser.add_argument('--output', help='Path for output files')
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()

    return args

def validate_env_setup():
    """Check that all environment variables are set
    """
    if "HF_TOKEN" not in os.environ:
        raise Exception("HF_TOKEN environment variable is not set. HF_TOKEN is required to download data from Huggingface. Please set and re-run the script")

    if "OPENAI_API_KEY" not in os.environ:
        raise Exception("OPENAI_API_KEY environment variable is not set. OPENAI_API_KEY is required for inference. Please set and re-run the script")


def download_dataset(dataset_name:str, split:Optional[str] = "test"):
    """Download specified dataset from Huggingface
    """
    return load_dataset(HUGGINGFACE_REPO_NAME, dataset_name, split=split)

def _get_adherence_pred_column(model):
    if model == "trulens":
        return TrulensFields.TRULENS_GROUNDEDNESS
    elif model == "ragas":
        return RagasFields.OUTPUT_FAITHFULNESS

def _get_context_relevance_pred_column(model):
    if model == "trulens":
        return TrulensFields.TRULENS_CONTEXT_RELEVANCE
    elif model == "ragas":
        return RagasFields.OUTPUT_CONTEXT_RELEVANCE


if __name__ == "__main__":
    validate_env_setup()
    args = parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=False)

    # Download data from Huggingface
    ds = download_dataset(args.dataset)

    # Run Inference
    output_path = os.path.join(args.output, f"{args.dataset}_{args.model}.jsonl")
    if not os.path.exists(output_path) or args.force:
        if args.model == "trulens":
            loop = asyncio.get_event_loop()
            ds = loop.run_until_complete(trulens_annotate_dataset(ds, output_path))
        elif args.model == "ragas":
            ds = ragas_annotate_dataset(ds, output_path)
        
    # Calculate Metrics
    results_path = os.path.join(args.output, f"{args.dataset}_{args.model}.jsonl")
    annotated_ds = load_dataset("json", data_files= results_path)['train']
    metrics = calculate_metrics(
        annotated_ds,
        pred_adherence=_get_adherence_pred_column(args.model),
        pred_context_releavance=_get_context_relevance_pred_column(args.model)
    )
    print(json.dumps(metrics, indent=4))

    
    # Save Results
    metrics_output_path = os.path.join(args.output, f"{args.dataset}_{args.model}_metrics.json")
    with open(metrics_output_path, "w") as f:
        f.write(json.dumps(metrics, indent=4))
