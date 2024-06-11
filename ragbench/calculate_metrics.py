"""
Reproduce RAGBench paper metrics
"""
import os
import numpy as np
import json
import argparse
import pandas as pd
from typing import Optional

from datasets import load_dataset
from constants import HUGGINGFACE_REPO_NAME, RAGBenchFields
from evaluation import rmse, auroc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the component dataset to evaluate', nargs='+', default=[])
    # parser.add_argument('--output', help='Path for output files')

    args = parser.parse_args()

    return args

def validate_env_setup():
    """Check that all environment variables are set
    """
    if "HF_TOKEN" not in os.environ:
        raise Exception("HF_TOKEN environment variable is not set. HF_TOKEN is required to download data from Huggingface. Please set and re-run the script")

def download_dataset(dataset_name:str, split:Optional[str] = "test"):
    """Download specified dataset from Huggingface
    """
    return load_dataset(HUGGINGFACE_REPO_NAME, dataset_name, split=split)

def _hallucination_auroc_from_adherence(trues, preds):
    """Convert adherence labels to hallucination labels
    """
    trues = np.array(trues)
    # drop occasional nulls
    eval_idx = ~np.isnan(trues)
    trues = ~trues[eval_idx]
    
    preds = 1 - np.array(preds, dtype=float)
    preds = preds[eval_idx]

    return auroc(trues, preds)

if __name__ == "__main__":
    validate_env_setup()
    args = parse_args()

    results = []

    for dataset_name in args.dataset:
        print(f"Evaluating {dataset_name}")
        ds = download_dataset(dataset_name)

        results.append({
            "dataset": dataset_name,

            "gpt35_hallucination": _hallucination_auroc_from_adherence(ds[RAGBenchFields.ADHERENCE], ds[RAGBenchFields.GPT35_ADHERENCE_PRED]),
            "gpt3_relevance": rmse(ds[RAGBenchFields.RELEVANCE], ds[RAGBenchFields.GPT35_RELEVANCE_PRED]),
            "gpt3_utilization": rmse(ds[RAGBenchFields.UTILIZATION], ds[RAGBenchFields.GPT35_UTILIZATION_PRED]),

            "ragas_hallucination": _hallucination_auroc_from_adherence(ds[RAGBenchFields.ADHERENCE], ds[RAGBenchFields.RAGAS_ADHERENCE_PRED]),
            "ragas_relevance": rmse(ds[RAGBenchFields.RELEVANCE], ds[RAGBenchFields.RAGAS_RELEVANCE_PRED]),

            "trulens_hallucination": _hallucination_auroc_from_adherence(ds[RAGBenchFields.ADHERENCE], ds[RAGBenchFields.TRULENS_ADHERENCE_PRED]),
            "trulens_relevance": rmse(ds[RAGBenchFields.RELEVANCE], ds[RAGBenchFields.TRULENS_RELEVANCE_PRED]),
        })

    print(json.dumps(results, indent=4))
    # results_df = pd.DataFrame(results)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(results_df)
    


