# Tools for evaluating the quality of the RAG pipeline.

Metrics:

retrieval_precision: how many of the retrieved chunks are actually relevant to the question
retrieval_recall: how much of the information needed for the answer was successfully retrieved
context_relevance: how closely the retrieved chunks relate to the user's question
groundedness: whether the generated answer is supported by the retrieved context
answer_correctness: how closely the generated answer matches the expected answer

The evaluator combines LLM-based scoring with semantic similarity checks to
measure both retrieval quality and answer quality.

Inspired by common RAG evaluation approaches such as RAGAS, but implemented
with a lighter dependency footprint.
"""
import asyncio
import json
from typing import Dict, List, Optional

import openai

from app.core.config import settings


class RAGEvaluator:
    """
    Evaluate a RAG pipeline output across 5 core metrics.
    Uses gpt-4o-mini as judge (cost-efficient, ~$0.0001 per evaluation).
    """

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.judge_model = "gpt-4o-mini"

    async def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_contexts: List[str],
        expected_answer: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run all RAG metrics concurrently."""
        tasks = [
            self._evaluate_retrieval_precision(question, retrieved_contexts),
            self._evaluate_context_relevance(question, retrieved_contexts),
            self._evaluate_groundedness(answer, retrieved_contexts),
        ]

        if expected_answer:
            tasks.append(self._evaluate_answer_correctness(answer, expected_answer))
            tasks.append(self._evaluate_retrieval_recall(expected_answer, retrieved_contexts))

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        keys = ["retrieval_precision", "context_relevance", "groundedness"]
        if expected_answer:
            keys += ["answer_correctness", "retrieval_recall"]

        for key, result in zip(keys, results_list):
            if isinstance(result, Exception):
                results[key] = None
            else:
                results[key] = result

        return results

    async def _judge(self, prompt: str) -> float:
        """Call LLM judge and parse 0.0-1.0 score from response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert RAG evaluator. "
                            "Respond ONLY with a JSON object: {\"score\": <float 0.0-1.0>, \"reasoning\": \"<brief explanation>\"}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=200,
            )
            text = response.choices[0].message.content or ""
            data = json.loads(text.strip())
            return float(data["score"])
        except Exception:
            return 0.0

    async def _evaluate_retrieval_precision(
        self, question: str, contexts: List[str]
    ) -> float:
        """What fraction of retrieved chunks are relevant to the question?"""
        scores = []
        for ctx in contexts[:5]:  # cap at 5 to control cost
            score = await self._judge(
                f"Question: {question}\n\nRetrieved chunk: {ctx[:500]}\n\n"
                "Is this chunk relevant to answering the question? Score 1.0=highly relevant, 0.0=irrelevant."
            )
            scores.append(score)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    async def _evaluate_context_relevance(
        self, question: str, contexts: List[str]
    ) -> float:
        """Overall relevance of the retrieved context set to the question."""
        context_str = "\n\n---\n\n".join(contexts[:3])
        return await self._judge(
            f"Question: {question}\n\nRetrieved context:\n{context_str[:1500]}\n\n"
            "How relevant is this retrieved context to answering the question? "
            "Score 1.0=perfectly relevant, 0.0=completely irrelevant."
        )

    async def _evaluate_groundedness(self, answer: str, contexts: List[str]) -> float:
        """Is the answer grounded in (supported by) the retrieved context?"""
        context_str = "\n\n---\n\n".join(contexts[:3])
        return await self._judge(
            f"Answer: {answer}\n\nRetrieved context:\n{context_str[:2000]}\n\n"
            "Is every factual claim in the answer supported by the retrieved context? "
            "Score 1.0=fully grounded (no hallucinations), 0.0=unsupported claims."
        )

    async def _evaluate_answer_correctness(
        self, answer: str, expected_answer: str
    ) -> float:
        """How correct is the answer compared to the expected answer?"""
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            emb_a = model.encode(answer, convert_to_tensor=True)
            emb_b = model.encode(expected_answer, convert_to_tensor=True)
            score = float(util.cos_sim(emb_a, emb_b)[0][0])
            return round(max(0.0, min(1.0, score)), 4)
        except ImportError:
            # Fallback to LLM judge if sentence-transformers not available
            return await self._judge(
                f"Expected answer: {expected_answer}\n\nActual answer: {answer}\n\n"
                "How correct is the actual answer compared to the expected answer? "
                "Score 1.0=identical meaning, 0.0=completely wrong."
            )

    async def _evaluate_retrieval_recall(
        self, expected_answer: str, contexts: List[str]
    ) -> float:
        """What fraction of the information needed (from expected_answer) was retrieved?"""
        context_str = "\n\n".join(contexts[:5])
        return await self._judge(
            f"Expected answer: {expected_answer}\n\nRetrieved context:\n{context_str[:2000]}\n\n"
            "Does the retrieved context contain enough information to construct the expected answer? "
            "Score 1.0=all necessary info present, 0.0=critical information missing."
        )
