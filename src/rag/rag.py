from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from qdrant_client import models
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.utils.utility import make_qdrant_client
from src.utils.ragas_eval import evaluate_rag

class RAGSearchResult(BaseModel):
    query: str                         # user query
    subject: Optional[str]             # subject
    chapter: Optional[str]             # chapter
    has_context: bool
    total_hits: int                    # number of chunks retrieved
    context_text: str                  # context retrieved
    sources: List[Dict[str, Any]]
    ragas_score: float = 0.0          
    ragas_context_precision: float = 0.0  # context precision metric
    ragas_faithfulness: float = 0.0       # faithfulness metric
    ragas_verdict: str = ""            # human-readable verdict

@tool
def rag_search(
    query: str,
    config: RunnableConfig,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search the student's uploaded study documents using semantic similarity.

    This is the PRIMARY tool. Always call this tool first before attempting any
    web search. It retrieves the most relevant text chunks from the student's own
    study materials stored in their personal Qdrant vector collection.

    Use this tool to:
    - Find explanations, definitions, derivations, or examples from the student's notes.
    - Ground your answer in the student's own uploaded content.
    - Optionally narrow the search to a specific subject or chapter for more focused results.

    IMPORTANT — After calling this tool, check the returned `ragas_score` (0.0–1.0):
      - ragas_score >= 0.5: Context is sufficient. Answer from it; do NOT call web_search.
      - ragas_score < 0.5:  Context is weak or missing. You MUST call web_search next
                            to supplement the answer with up-to-date web information.

    Args:
        query (str): The semantic search query. Should be the student's question or
            a rephrasing of the core concept being looked up. Use natural language,
            not just keywords. If a first call returns weak results, retry with a
            rephrased or more specific query before giving up.
        subject (Optional[str]): The MongoDB ObjectId of the subject to filter results by.
            Use this when the student's query is clearly scoped to a specific subject.
            Leave as None to search across all of the student's documents.
        chapter (Optional[str]): The MongoDB ObjectId of the chapter to filter results by.
            Use this when the student's query is scoped to a specific chapter within a subject.
            Leave as None to search across all chapters.
        limit (int): Maximum number of document chunks to retrieve. Defaults to 5.
            Increase only if the query is broad and more context is needed for a complete answer.

    Returns:
        A dict containing:
        - query (str): The original query passed to this tool.
        - has_context (bool): True if at least one relevant chunk was found.
        - total_hits (int): Number of document chunks retrieved.
        - context_text (str): The full retrieved context as a single string,
            chunks separated by '---'. Use this as the primary source for your answer.
        - sources (list): Metadata for each retrieved chunk.
        - ragas_score (float): Combined RAGAS faithfulness + context_precision score (0–1).
            A score < 0.5 means context is insufficient — you MUST escalate to web_search.
        - ragas_context_precision (float): Context precision sub-score (0–1).
        - ragas_faithfulness (float): Faithfulness sub-score (0–1).
        - ragas_verdict (str): Human-readable explanation of the RAGAS score.
    """
    collection_name: str = config.get("configurable", {}).get("collection_name", "")
    
    if not subject:
        subject = config.get("configurable", {}).get("subject")
    if not chapter:
        chapter = config.get("configurable", {}).get("chapter")

    if not collection_name:
        return RAGSearchResult(
            query=query, subject=subject, chapter=chapter,
            has_context=False, total_hits=0,
            context_text="Error: No collection configured for this user.",
            sources=[],
        ).model_dump()

    try:
        conditions = []
        if subject:
            conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
        if chapter:
            conditions.append(FieldCondition(key="chapter", match=MatchValue(value=chapter)))

        search_filter = Filter(must=conditions) if conditions else None
        qdrant_client = make_qdrant_client()

        def search_qdrant():
            try:
                return qdrant_client.query_points(
                    collection_name=collection_name,
                    prefetch=[
                        models.Prefetch(
                            query=models.Document(
                                text=query,
                                model="BAAI/bge-small-en-v1.5",
                            ),
                            using="dense",
                            limit=limit * 2,         # over-fetch before fusion
                        ),
                        models.Prefetch(
                            query=models.Document(
                                text=query,
                                model="Qdrant/bm25",
                            ),
                            using="sparse",
                            limit=limit * 2,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                ).points
            except Exception as e:
                if "Not found: Collection" in str(e):
                    return []
                raise

        results = search_qdrant()
        
        '''example results - 
        [
            {
                "id": 101,
                "score": 0.92,
                "payload": {
                    "text": "A transaction is a sequence of database operations treated as one unit.",
                    "subject": "DBMS",
                    "chapter": "Transactions",
                    "source": "dbms_notes.pdf"
                }
            },{},{}...{}
        ]
        '''
        
        sources = []
        context_chunks = []

        for idx, point in enumerate(results, start=1):
            payload = point.payload
            text = payload.get("text").strip()
            if text:
                context_chunks.append(text)
            sources.append({
                "rank": idx,
                "score": getattr(point, "score", None),
                "subject": payload.get("subject"),
                "chapter": payload.get("chapter"),
                "source": payload.get("source"),
                "text_preview": text[:300] if text else "",
            })

        context_text = (
            "\n\n---\n\n".join(context_chunks)
            if context_chunks
            else "No relevant context found in your uploaded documents."
        )

        ragas_result = evaluate_rag(
            query=query,
            context_text=context_text,
            has_context=bool(context_chunks),
        )

        return RAGSearchResult(
            query=query,
            subject=subject,
            chapter=chapter,
            has_context=bool(context_chunks),
            total_hits=len(results),
            context_text=context_text,
            sources=sources,
            ragas_score=ragas_result["combined"],
            ragas_context_precision=ragas_result["context_precision"],
            ragas_faithfulness=ragas_result["faithfulness"],
            ragas_verdict=ragas_result["verdict"],
        ).model_dump()

    except Exception as e:
        return RAGSearchResult(
            query=query, subject=subject, chapter=chapter,
            has_context=False, total_hits=0,
            context_text=f"Search failed: {str(e)}",
            sources=[],
            ragas_score=0.0,
            ragas_context_precision=0.0,
            ragas_faithfulness=0.0,
            ragas_verdict="Search failed — you MUST call web_search.",
        ).model_dump()
        