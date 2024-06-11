# RAGbench

- [dataset on Huggingface](https://huggingface.co/datasets/rungalileo/ragbench)

## Usage

## Reproduce RAGBench Baseline Metrics
To reproduce benchmark metrics on RAGBench, use `calculate_metrics.py`. For example, to reproduce GPT-3.5, RAGAS, Trulens for a set of RAGBench component datasets run:
```
python calculate_metrics.py --dataset hotpotqa msmarco hagrid expertqa
```

## Generate Baseline Results
Use the `run_inference.py` script to evaluate RAG eval frameworks on RAGBench. Input arguments:
- `dataset`: name of the RAGBench dataset to run inference on
- `model`: the model to evaluate (trulens or ragas)
- `output`: output directory to store results in

Run Trulens inference on HotpotQA subset:
```
python run_inference.py --dataset msmarco --model trulens --output results
```