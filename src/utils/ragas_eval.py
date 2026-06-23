"""
RAGAS v0.4 evaluation module.

Computes two reference-free metrics using the official ragas library:
  - LLMContextPrecisionWithoutReference: Is the retrieved context relevant to the question?
  - Faithfulness:                        Can the answer be grounded solely in the retrieved context?

Both metrics use llama-3.1-8b-instant via Groq (already in the dependency stack).
The combined score (0.0–1.0) is returned alongside individual metric values.

Score thresholds:
  >= 0.7  →  Strong — answer fully from documents
  >= 0.5  →  Adequate — answer from documents, web search optional
  < 0.5   →  Weak — MUST escalate to web_search
"""

import asyncio
from typing import Optional

from langchain_groq import ChatGroq
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness
from src.utils.utility import logger

from src.constants.constants import TEMPARATURE
GROQ_MODEL = "llama-3.1-8b-instant"

def _get_evaluator_llm():
    """Lazy-init the ragas-wrapped Groq LLM."""
    global _evaluator_llm
    if _evaluator_llm is None:
        _evaluator_llm = LangchainLLMWrapper(
            ChatGroq(model=GROQ_MODEL, temperature=TEMPARATURE)
        )
    return _evaluator_llm

async def _async_score(
    query: str,
    contexts: list[str],
    response: str,
) -> dict:
    """
    Run context_precision and faithfulness async and return both raw scores.
    Falls back to 0.0 on any metric failure.
    """
    llm = _get_evaluator_llm()

    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=contexts,
        response=response,
    )

    cp_score: float = 0.0
    faith_score: float = 0.0

    # Context Precision (reference-free)
    try:
        metric_cp = LLMContextPrecisionWithoutReference(llm=llm)
        cp_score = await metric_cp.single_turn_ascore(sample)
        if cp_score is None:
            cp_score = 0.0
        logger.info(f"[RAGAS] context_precision = {cp_score:.3f}")
    except Exception as e:
        logger.warning(f"[RAGAS] context_precision failed: {e}")

    # Faithfulness
    try:
        metric_f = Faithfulness(llm=llm)
        faith_score = await metric_f.single_turn_ascore(sample)
        if faith_score is None:
            faith_score = 0.0
        logger.info(f"[RAGAS] faithfulness      = {faith_score:.3f}")
    except Exception as e:
        logger.warning(f"[RAGAS] faithfulness failed: {e}")

    combined = round((cp_score + faith_score) / 2.0, 3)
    combined = max(0.0, min(1.0, combined))
    logger.info(f"[RAGAS] combined_score    = {combined:.3f}")

    return {
        "context_precision": round(float(cp_score), 3),
        "faithfulness": round(float(faith_score), 3),
        "combined": combined,
    }

def evaluate_rag(
    query: str,
    context_text: str,
    has_context: bool,
    response: str = "",
) -> dict:
    """
    Evaluate a RAG retrieval using RAGAS v0.4 metrics.

    Args:
        query:        The user's question.
        context_text: The full concatenated retrieved context string.
        has_context:  Whether any chunks were retrieved at all.
        response:     (Optional) A placeholder / preliminary answer for faithfulness scoring.
                      If not provided, the context_text itself is used as a proxy response.

    Returns a dict:
        {
            "context_precision": float,   # 0–1
            "faithfulness":      float,   # 0–1
            "combined":          float,   # average of the two, 0–1
            "verdict":           str,     # human-readable label
        }
    """
    empty_result = {
        "context_precision": 0.0,
        "faithfulness": 0.0,
        "combined": 0.0,
        "verdict": "No relevant context found — escalate to web_search.",
    }

    if not has_context or not context_text or "No relevant context found" in context_text:
        logger.debug("[RAGAS] Skipping evaluation — no context retrieved.")
        return empty_result

    # Use first 2000 chars to avoid massive prompts
    contexts = [context_text[:2000].strip()]

    # If no response is provided, use the context as a proxy (faithfulness checks
    # whether the response is grounded; context ≈ ideal response for scoring purposes)
    proxy_response = response.strip() if response.strip() else contexts[0]

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g. FastAPI async handler)
                # Use a thread executor to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _async_score(query, contexts, proxy_response))
                    scores = future.result(timeout=30)
            else:
                scores = loop.run_until_complete(_async_score(query, contexts, proxy_response))
        except RuntimeError:
            scores = asyncio.run(_async_score(query, contexts, proxy_response))

    except Exception as e:
        logger.error(f"[RAGAS] Evaluation failed entirely: {e}", exc_info=True)
        return empty_result

    combined = scores["combined"]
    if combined >= 0.7:
        verdict = f"Strong (score={combined:.2f}) — answer from documents."
    elif combined >= 0.5:
        verdict = f"Adequate (score={combined:.2f}) — answer from documents; web search optional."
    elif combined > 0.0:
        verdict = f"Weak (score={combined:.2f}) — context insufficient, MUST call web_search."
    else:
        verdict = "No signal (score=0.0) — MUST call web_search."

    logger.info(
        f"[RAGAS] ┌─ query            : {query[:80]}\n"
        f"        ├─ context_precision: {scores['context_precision']:.3f}\n"
        f"        ├─ faithfulness     : {scores['faithfulness']:.3f}\n"
        f"        ├─ combined         : {combined:.3f}\n"
        f"        └─ verdict          : {verdict}"
    )

    return {**scores, "verdict": verdict}
