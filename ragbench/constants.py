HUGGINGFACE_REPO_NAME = "rungalileo/ragbench"

class RAGBenchFields:
    ID = "id"
    CONTEXT = "documents"
    QUESTION = "question"
    RESPONSE = "response"
    SUPPORTED = "overall_supported"
    RELEVANCE = "relevance_score"
    UTILIZATION = "utilization_score"
    COMPLETENESS = "completeness_score"

class TrulensFields:
    TRULENS_GROUNDEDNESS="trulens_goundedness"
    TRULENS_CONTEXT_RELEVANCE="trulens_context_relevance"


class RagasFields:
    INPUT_CONTEXT="contexts"
    INPUT_QUESTOIN="question"
    INPUT_ANSWER="answer"

    OUTPUT_FAITHFULNESS="faithfulness"
    OUTPUT_CONTEXT_RELEVANCE="context_relevancy"
    OUTPUT_ANSWER_RELEVANCE="answer_relevancy"
