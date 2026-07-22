
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# RAG pipeline result for evaluation.
class RAGEvaluationCreate(BaseModel):
    experiment_run_id: Optional[uuid.UUID] = None
    dataset_row_id: Optional[uuid.UUID] = None
    question: str
    answer: str
    retrieved_contexts: List[str] = Field(min_length=1)
    expected_answer: Optional[str] = None

    # RAG configuration metadata
    embedding_model: Optional[str] = None    #text-embedding-3-small, bge-large, etc.
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    retrieval_strategy: Optional[str] = None  #dense, sparse, hybrid, mmr.
    top_k: Optional[int] = None
    reranker: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    organization_id: uuid.UUID
    experiment_run_id: Optional[uuid.UUID]
    question: str
    answer: str
    retrieved_contexts: List[str]

    # Computed metrics
    retrieval_precision: Optional[float]
    retrieval_recall: Optional[float]
    context_relevance: Optional[float]
    groundedness: Optional[float]
    answer_correctness: Optional[float]

    # Config
    embedding_model: Optional[str]
    chunk_size: Optional[int]
    retrieval_strategy: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime


class RAGCompareRequest(BaseModel):
    """Compare multiple RAG configurations on the same dataset."""
    evaluation_ids: List[uuid.UUID] = Field(min_length=2, max_length=20)
    group_by: str = Field(
        default="embedding_model",
        description="embedding_model, chunk_size, or retrieval_strategy"
    )


class RAGGroupStats(BaseModel):
    group_value: str
    count: int
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_context_relevance: float
    avg_groundedness: float
    avg_answer_correctness: float
    overall_score: float  # weighted composite


class RAGCompareResponse(BaseModel):
    group_by: str
    groups: List[RAGGroupStats]
    winner: str   # group_value of the best performing config
    recommendation: str  # human-readable advice
