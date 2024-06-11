HUGGINGFACE_REPO_NAME = "rungalileo/ragbench"

class RAGBenchFields:
    ID = "id"
    CONTEXT = "documents"
    QUESTION = "question"
    RESPONSE = "response"
    SUPPORTED = "overall_supported"
    ADHERENCE = "adherence_score"
    RELEVANCE = "relevance_score"
    UTILIZATION = "utilization_score"
    COMPLETENESS = "completeness_score"

    RAGAS_ADHERENCE_PRED = "ragas_faithfulness"
    RAGAS_RELEVANCE_PRED = "ragas_context_relevance"

    TRULENS_ADHERENCE_PRED = "trulens_groundedness"
    TRULENS_RELEVANCE_PRED = "trulens_context_relevance"

    GPT35_ADHERENCE_PRED = "gpt3_adherence"
    GPT35_RELEVANCE_PRED = "gpt3_context_relevance"
    GPT35_UTILIZATION_PRED = "gpt35_utilization"
    

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

DEFAULT_OPENAI_MAX_CONCURRENT = 100