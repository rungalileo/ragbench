from dataclasses import dataclass

import asyncio
from typing import Coroutine, Dict, Sequence, List, Optional, Tuple, Literal

from tqdm.auto import trange

import openai
import trulens_eval
import nltk
from nltk import sent_tokenize
from trulens_eval.feedback import prompts
from trulens_eval.utils import generated as mod_generated_utils
import warnings
import numpy as np

from trulens_eval.feedback.provider.base import *
from trulens_eval.feedback.provider.openai import *

from pydantic import Field

@dataclass
class TrulensMetric:
    value: float
    explanation: Optional[str]
    error: Optional[str]


@dataclass
class TrulensAnnotation:
    id: str  # id of the annotated entry
    groundedness: Optional[TrulensMetric] = None
    context_relevance: Optional[TrulensMetric] = None
    answer_relevance: Optional[TrulensMetric] = None


def _limit_concurrency(
    coroutines: Sequence[Coroutine], concurrency: int
) -> List[Coroutine]:
    """Decorate coroutines to limit concurrency.
    Enforces a limit on the number of coroutines that can run concurrently in higher
    level asyncio-compatible concurrency managers like asyncio.gather(coroutines) and
    asyncio.as_completed(coroutines).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]


class AsyncTrulensOpenAI(
    trulens_eval.feedback.provider.openai.OpenAI
):
    async_client: openai.AsyncClient = Field(default_factory=openai.AsyncClient)
    already_downloaded_nltk: bool = False
    supported_metrics:List[str] = ["groundedness", "context_relevance", "answer_relevance"]

    async def annotate(
        self,
        ids: List[str],
        contexts: List[List[str]],
        questions: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
        metrics: List[str] = supported_metrics,
        max_concurrent: int = 100
    ) -> List[TrulensAnnotation]:
        for metric in metrics:
            if metric not in self.supported_metrics:
                print(f"{metric} is not a supported metric. Supported metrics: {self.supported_metrics}")
                return
        
        assert len(ids) == len(contexts)
        pbar = trange(len(ids) * len(metrics))

        coroutines = []
        for i in range(len(ids)):
            entry_id = ids[i]
            context = contexts[i]

            if "groundedness" in metrics and responses:
                coroutines.append(
                    self.agroundedness_measure_with_cot_reasons(
                        entry_id, context, responses[i], callback=pbar.update
                    )
                )
            
            if "context_relevance" in metrics and questions:
                coroutines.append(
                    self.acontext_relevance_with_cot_reasons(
                        entry_id, context, questions[i], callback=pbar.update
                    )
                )
            
            if "answer_relevance" in metrics and questions and responses:
                coroutines.append(
                    self.arelevance_with_cot_reasons(
                        entry_id, question=questions[i], response=responses[i], callback=pbar.update
                    )
                )

        coroutines = _limit_concurrency(coroutines, max_concurrent)
        result = await asyncio.gather(*coroutines)
        pbar.close()

        # Aggregate Results
        merged_results = {}
        for ann in result:
            if ann.id not in merged_results:
                merged_results[ann.id] = TrulensAnnotation(id=ann.id)
                
            if ann.groundedness:
                merged_results[ann.id].groundedness = ann.groundedness
            
            if ann.context_relevance:
                merged_results[ann.id].context_relevance = ann.context_relevance
            
            if ann.answer_relevance:
                merged_results[ann.id].answer_relevance = ann.answer_relevance

        return list(merged_results.values())

    async def context_relevance_with_cot(
        self,
        ids: List[str],
        contexts: List[List[str]],
        questions: List[str],
        max_concurrent: int = 100
    ):
        assert len(contexts) == len(questions)
        pbar = trange(len(contexts))

        coroutines = [
            self.acontext_relevance_with_cot_reasons(
                id, context, question, callback=pbar.update
            )
            for id, context, question in zip(ids, contexts, questions)
        ]
        coroutines = _limit_concurrency(coroutines, max_concurrent)
        result = await asyncio.gather(*coroutines)
        pbar.close()

        return result

    async def groundedness_measure_with_cot_reasons_multiple_inputs(
        self,
        sources: List[str],
        statements: List[str],
        max_concurrent: int = 100
    ):
        assert len(sources) == len(statements)

        pbar = trange(len(sources))

        coroutines = [
            self.agroundedness_measure_with_cot_reasons(
                source, statement, callback=pbar.update
            )
            for source, statement in zip(sources, statements)
        ]
        coroutines = _limit_concurrency(coroutines, max_concurrent)

        result = await asyncio.gather(*coroutines)

        pbar.close()

        return result

    async def agroundedness_measure_with_cot_reasons(
        self,
        entry_id,
        source: str,
        statement: str,
        callback=None,
    ) -> Tuple[float, dict]:
        """A measure to track if the source material supports each sentence in
        the statement using an LLM provider.

        The LLM will process the entire statement at once, using chain of
        thought methodology to emit the reasons.

        !!! example

            ```python
            from trulens_eval import Feedback
            from trulens_eval.feedback.provider.openai import OpenAI

            provider = OpenAI()

            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons)
                .on(context.collect()
                .on_output()
                )
            ```
        Args:
            source: The source that should support the statement.
            statement: The statement to check groundedness.

        Returns:
            Tuple[float, str]: A tuple containing a value between 0.0 (not grounded) and 1.0 (grounded) and a string containing the reasons for the evaluation.
        """
        if not self.already_downloaded_nltk:
            nltk.download('punkt')
            self.already_downloaded_nltk = True

        groundedness_scores = {}
        reasons_str = ""
        error = None

        try:
            hypotheses = sent_tokenize(statement)
            system_prompt = prompts.LLM_GROUNDEDNESS_SYSTEM
            for i, hypothesis in enumerate(hypotheses):
                user_prompt = prompts.LLM_GROUNDEDNESS_USER.format(
                    premise=f"{source}", hypothesis=f"{hypothesis}"
                )
                score, reason = await self.agenerate_score_and_reasons(
                    system_prompt, user_prompt
                )
                groundedness_scores[f"statement_{i}"] = score
                reasons_str += f"STATEMENT {i}:\n{reason['reason']}\n"

            # Calculate the average groundedness score from the scores dictionary
            average_groundedness_score = float(
                np.mean(list(groundedness_scores.values()))
            )
        except openai.BadRequestError as e:
            average_groundedness_score = None
            reasons_str = "Input token length exceeds LLM context length limit"
            error = e.code  # context_length_exceeded
        except Exception as e:
            average_groundedness_score = None
            reasons_str = "Error parsing trulens response"
            error = "error"

        if callback is not None:
            callback()
        # return average_groundedness_score, {"reasons": reasons_str}, error
    
        return TrulensAnnotation(
                id = entry_id,
                groundedness = TrulensMetric(
                    value = average_groundedness_score,
                    explanation = reasons_str,
                    error =  error,
                )
            )
    
    async def acontext_relevance_with_cot_reasons(
        self,
        entry_id: str,
        context: List[str],
        question: str,
        temperature: float = 0.0,
        callback=None,
    ) -> Tuple[float, dict]:
        """ Async verson of trulens_eval.feedback.provider.base.context_relevance_with_cot_reasons
        """
        system_prompt = prompts.CONTEXT_RELEVANCE_SYSTEM
        user_prompt = str.format(
            prompts.CONTEXT_RELEVANCE_USER, question=question, context=context
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )

        try:
            score, reason = await self.agenerate_score_and_reasons(
                system_prompt, user_prompt, temperature=temperature
            )
            error = None
            reason = reason['reason']
        except openai.BadRequestError as e:
            error = e.code 
            score = None
            reason = "Input token length exceeds LLM context length limit"
        
        if callback is not None:
            callback()

        return TrulensAnnotation(
            id=entry_id,
            context_relevance=TrulensMetric(
                value=score,
                explanation=reason,
                error=error
            )
        )

    async def arelevance_with_cot_reasons(
        self,
        entry_id: str,
        question: str,
        response: str,
        temperature: float = 0.0,
        callback=None,
    ) -> Tuple[float, dict]:
        """ Async verson of trulens_eval.feedback.provider.base.context_relevance_with_cot_reasons
        """
        system_prompt = prompts.ANSWER_RELEVANCE_SYSTEM
        user_prompt = str.format(
            prompts.ANSWER_RELEVANCE_USER, prompt=question, response=response
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )

        try:
            score, reason = await self.agenerate_score_and_reasons(
                system_prompt, user_prompt, temperature=temperature
            )
            error = None
            reason = reason['reason']
        except openai.BadRequestError as e:
            error = e.code 
            score = None
            reason = "Input token length exceeds LLM context length limit"
        
        if callback is not None:
            callback()

        return TrulensAnnotation(
            id=entry_id,
            answer_relevance=TrulensMetric(
                value=score,
                explanation=reason,
                error=error
            )
        )


    async def _acreate_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[Sequence[Dict]] = None,
        **kwargs
    ) -> str:
        if 'model' not in kwargs:
            kwargs['model'] = self.model_engine

        if 'temperature' not in kwargs:
            kwargs['temperature'] = 0.0

        if 'seed' not in kwargs:
            kwargs['seed'] = 123

        if messages is not None:
            # completion = self.endpoint.client.chat.completions.create(
            #     messages=messages, **kwargs
            # )
            completion = await self.async_client.chat.completions.create(
                messages=messages, **kwargs
            )

        elif prompt is not None:
            # completion = self.endpoint.client.chat.completions.create(
            #     messages=[{
            #         "role": "system",
            #         "content": prompt
            #     }], **kwargs
            # )
            completion = await self.async_client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": prompt
                }], **kwargs
            )

        else:
            raise ValueError("`prompt` or `messages` must be specified.")

        return completion.choices[0].message.content

    async def agenerate_score_and_reasons(
        self,
        system_prompt: str,
        user_prompt: Optional[str] = None,
        normalize: float = 10.0,
        temperature: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Base method to generate a score and reason, used for evaluation.

        Args:
            system_prompt: A pre-formatted system prompt.

            user_prompt: An optional user prompt. Defaults to None.

            normalize: The normalization factor for the score.

            temperature: The temperature for the LLM response.

        Returns:
            The score on a 0-1 scale.

            Reason metadata if returned by the LLM.
        """
        assert self.endpoint is not None, "Endpoint is not set."

        llm_messages = [{"role": "system", "content": system_prompt}]
        if user_prompt is not None:
            llm_messages.append({"role": "user", "content": user_prompt})

        response = await self._acreate_chat_completion(
            messages=llm_messages,
            temperature=temperature,
        )

        if "Supporting Evidence" in response:
            score = -1
            supporting_evidence = None
            criteria = None
            for line in response.split('\n'):
                if "Score" in line:
                    try:
                        score = mod_generated_utils.re_0_10_rating(line) / normalize
                    except:
                        score = None

                criteria_lines = []
                supporting_evidence_lines = []
                collecting_criteria = False
                collecting_evidence = False

                for line in response.split('\n'):
                    if "Criteria:" in line:
                        criteria_lines.append(
                            line.split("Criteria:", 1)[1].strip()
                        )
                        collecting_criteria = True
                        collecting_evidence = False
                    elif "Supporting Evidence:" in line:
                        supporting_evidence_lines.append(
                            line.split("Supporting Evidence:", 1)[1].strip()
                        )
                        collecting_evidence = True
                        collecting_criteria = False
                    elif collecting_criteria:
                        if "Supporting Evidence:" not in line:
                            criteria_lines.append(line.strip())
                        else:
                            collecting_criteria = False
                    elif collecting_evidence:
                        if "Criteria:" not in line:
                            supporting_evidence_lines.append(line.strip())
                        else:
                            collecting_evidence = False

                criteria = "\n".join(criteria_lines).strip()
                supporting_evidence = "\n".join(supporting_evidence_lines
                                               ).strip()
            reasons = {
                'reason':
                    (
                        f"{'Criteria: ' + str(criteria)}\n"
                        f"{'Supporting Evidence: ' + str(supporting_evidence)}"
                    )
            }
            return score, reasons

        else:
            score = mod_generated_utils.re_0_10_rating(response) / normalize
            warnings.warn(
                "No supporting evidence provided. Returning score only.",
                UserWarning
            )
            return score, {}
