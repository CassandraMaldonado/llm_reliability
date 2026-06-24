"""app/api/v1/endpoints/rag.py — RAG pipeline evaluation endpoints."""
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models import User, RAGEvaluation
from app.repositories import RAGRepository
from app.schemas.rag import (
    RAGEvaluationCreate, RAGEvaluationResponse,
    RAGCompareRequest, RAGCompareResponse, RAGGroupStats,
)
from app.schemas.common import PaginatedResponse

router = APIRouter()


async def _compute_rag_metrics(eval_id: uuid.UUID, session: AsyncSession):
    """
    Background task: compute RAG metrics after submission.
    
    Metrics computed:
    - retrieval_precision: fraction of retrieved chunks that are relevant
    - retrieval_recall: fraction of relevant chunks that were retrieved
    - context_relevance: how relevant retrieved context is to the question
    - groundedness: how grounded the answer is in the retrieved context
    - answer_correctness: compared to expected_answer if provided
    
    Uses LLM-as-judge for context_relevance and groundedness.
    Uses semantic similarity for answer_correctness.
    """
    from sqlalchemy import select
    from app.evaluators.rag_evaluator import RAGEvaluator

    result = await session.execute(
        select(RAGEvaluation).where(RAGEvaluation.id == eval_id)
    )
    rag_eval = result.scalar_one_or_none()
    if not rag_eval:
        return

    evaluator = RAGEvaluator()
    scores = await evaluator.evaluate(
        question=rag_eval.question,
        answer=rag_eval.answer,
        retrieved_contexts=rag_eval.retrieved_contexts,
        expected_answer=rag_eval.expected_answer,
    )

    for key, value in scores.items():
        if hasattr(rag_eval, key):
            setattr(rag_eval, key, value)

    await session.commit()


@router.post("/evaluations", response_model=RAGEvaluationResponse, status_code=202)
async def submit_rag_evaluation(
    data: RAGEvaluationCreate,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Submit a RAG pipeline output for evaluation.
    Returns 202 Accepted — metrics are computed asynchronously.
    Poll the GET endpoint to check when scores are populated.
    """
    rag_eval = RAGEvaluation(
        organization_id=current_user.organization_id,
        **data.model_dump(),
    )
    repo = RAGRepository(session)
    rag_eval = await repo.create(rag_eval)
    await session.commit()
    await session.refresh(rag_eval)

    # Queue metric computation as background task
    background_tasks.add_task(_compute_rag_metrics, rag_eval.id, session)

    return rag_eval


@router.get("/evaluations", response_model=PaginatedResponse[RAGEvaluationResponse])
async def list_rag_evaluations(
    run_id: Optional[uuid.UUID] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    repo = RAGRepository(session)
    filters = {}
    if run_id:
        filters["experiment_run_id"] = run_id
    offset = (page - 1) * page_size
    items, total = await repo.list(current_user.organization_id, offset, page_size, **filters)
    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.post("/compare", response_model=RAGCompareResponse)
async def compare_rag_configs(
    data: RAGCompareRequest,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Compare RAG configurations grouped by embedding_model, chunk_size, or retrieval_strategy.
    Identifies which config performs best across retrieval and generation metrics.
    """
    repo = RAGRepository(session)
    groups_data = await repo.get_grouped_stats(
        current_user.organization_id, data.group_by, data.evaluation_ids
    )

    if not groups_data:
        raise HTTPException(status_code=404, detail="No evaluations found for given IDs")

    groups = []
    for g in groups_data:
        scores = [
            g.get("avg_retrieval_precision") or 0,
            g.get("avg_retrieval_recall") or 0,
            g.get("avg_context_relevance") or 0,
            g.get("avg_groundedness") or 0,
            g.get("avg_answer_correctness") or 0,
        ]
        overall = sum(scores) / len([s for s in scores if s > 0]) if any(s > 0 for s in scores) else 0
        groups.append(RAGGroupStats(
            group_value=str(g.get("group_value", "unknown")),
            count=g.get("count", 0),
            avg_retrieval_precision=float(g.get("avg_retrieval_precision") or 0),
            avg_retrieval_recall=float(g.get("avg_retrieval_recall") or 0),
            avg_context_relevance=float(g.get("avg_context_relevance") or 0),
            avg_groundedness=float(g.get("avg_groundedness") or 0),
            avg_answer_correctness=float(g.get("avg_answer_correctness") or 0),
            overall_score=round(overall, 4),
        ))

    groups.sort(key=lambda g: g.overall_score, reverse=True)
    winner = groups[0].group_value if groups else "unknown"

    return RAGCompareResponse(
        group_by=data.group_by,
        groups=groups,
        winner=winner,
        recommendation=(
            f"Based on evaluation data, '{winner}' achieves the highest composite RAG score. "
            f"Consider this {data.group_by} for production."
        ),
    )
