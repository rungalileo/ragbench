import os
import json
from typing import Dict, Optional

from datasets import Dataset
from trulens_async import AsyncTrulensOpenAI
from constants import RAGBenchFields, TrulensFields, RagasFields
from ragas import evaluate
from ragas.metrics import faithfulness, context_relevancy, answer_relevancy

async def trulens_annotate_dataset(hf_dataset: Dataset, output_path:Optional[str] = None) -> Dataset:
    """Annotate `hf_dataset` with Trulens metrics

    Returns:
        hf_dataset (Dataset) with 2 new columns: `trulens_goundedness` and `trulens_context_relevance`
    """
    print(f"Running Trulens Inference on {hf_dataset.num_rows} rows")

    async_trulens = AsyncTrulensOpenAI()

    contexts = hf_dataset[RAGBenchFields.CONTEXT]
    questions = hf_dataset[RAGBenchFields.QUESTION]
    responses = hf_dataset[RAGBenchFields.RESPONSE]
    # ids = hf_dataset[ID_COLUMN]
    ids = [f"{id}_{generator}" for id, generator in zip(hf_dataset['id'], hf_dataset['generation_model_name'])]

    trulens_results = await async_trulens.annotate(
        ids=ids,
        contexts=contexts,
        questions=questions,
        responses=responses,
        metrics=['groundedness', "context_relevance", "answer_relevance"],
        max_concurrent=100
    )

    groundedncess_annotations = [annotation.groundedness.value for annotation in trulens_results]
    context_relevance_annotations = [annotation.context_relevance.value for annotation in trulens_results]
    # answer_relevance_annotations = [annotation.answer_relevance.value for annotation in trulens_results]
    print(len(groundedncess_annotations))
    print(len(context_relevance_annotations))

    annotated_dataset = hf_dataset.add_column(TrulensFields.TRULENS_GROUNDEDNESS, groundedncess_annotations)
    annotated_dataset = annotated_dataset.add_column(TrulensFields.TRULENS_CONTEXT_RELEVANCE, context_relevance_annotations)
    # annotated_dataset = annotated_dataset.add_column("trulens_answer_relevance", answer_relevance_annotations)
    
    if output_path:
        annotated_dataset.to_json(output_path)

    return annotated_dataset


def ragas_annotate_dataset(hf_dataset: Dataset, output_path:Optional[str] = None) -> Dataset:
    """Annotate `hf_dataset` with Trulens metrics

    Returns:
        hf_dataset (Dataset) with 2 new columns: `trulens_goundedness` and `trulens_context_relevance`
    """
    print(f"Running RAGAS Inference on {hf_dataset.num_rows} rows")

    # prep column names
    hf_dataset = hf_dataset.rename_column(RAGBenchFields.CONTEXT, RagasFields.INPUT_CONTEXT)
    hf_dataset = hf_dataset.rename_column(RAGBenchFields.RESPONSE, RagasFields.INPUT_ANSWER)

    ragas_result = evaluate(hf_dataset, metrics=[faithfulness, context_relevancy])
    
    ragas_df = ragas_result.to_pandas()
    annotated_dataset = Dataset.from_pandas(ragas_df)
    
    if output_path:
        annotated_dataset.to_json(output_path)
    return annotated_dataset
