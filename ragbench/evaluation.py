import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score

from datasets import Dataset
from constants import RAGBenchFields


def rmse(trues: List[float], preds: List[float]) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between input ground truth (`trues`) and predictions (`preds`)
    """
    if len(trues) != len(preds):
        return None
    
    trues = np.array(trues)
    preds = np.array(preds, dtype=float)

    # Ignore Nulls in predictions
    eval_idx = ~np.isnan(preds)
    trues = trues[eval_idx]
    preds = preds[eval_idx]
    
    return np.sqrt(np.mean((preds - trues)**2))

def auroc(trues: List[bool], preds: List[float]) -> float:
    """
    Calculate Area Under Reciever Operator Characteristic Curve (AUROC) between input ground truth (`trues`) and predictions (`preds`)
    """
    eval_idx = ~np.isnan(preds)
    return roc_auc_score(trues[eval_idx], preds[eval_idx])


def calculate_metrics(
    annotatated_dataset: Dataset,
    pred_adherence:Optional[str] = None,
    pred_context_releavance:Optional[str] = None,
    pred_context_utilization:Optional[str] = None,
    ground_truth_adherence:str = RAGBenchFields.SUPPORTED, 
    ground_truth_context_relevance:str = RAGBenchFields.RELEVANCE,
    ground_truth_context_utilization:str = RAGBenchFields.UTILIZATION,
) -> Dict[str, float]:
    calculated_metrics = {}
    
    # Evaluate Hallucination Detection Task
    if pred_adherence:
        trues_hallucination = ~np.array(annotatated_dataset[ground_truth_adherence])
        preds_hallucination = 1 - np.array(annotatated_dataset[pred_adherence], dtype=float)
        calculated_metrics["hallucination_auroc"] = auroc(trues_hallucination, preds_hallucination)

    # Evaluate Context Relevance Task
    if pred_context_releavance:
        trues_relevance = annotatated_dataset[ground_truth_context_relevance]
        preds_relevance = annotatated_dataset[pred_context_releavance]
        calculated_metrics["relevance_rmse"] = rmse(trues_relevance, preds_relevance)

    # Evaluate Context utilization Task
    if pred_context_utilization:
        trues_utilization = annotatated_dataset[ground_truth_context_utilization]
        preds_utilization = annotatated_dataset[pred_context_utilization]
        calculated_metrics["utilization_rmse"] = rmse(trues_relevance, preds_relevance)


    return calculated_metrics