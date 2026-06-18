import os
import uuid
import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client.models import PointStruct, VectorParams, Distance

from src.utils.utility import logger, retry_config

DENSE_MODEL = "BAAI/bge-small-en-v1.5"

def make_qdrant_ltm_client() -> QdrantClient:
    """
    Creates a Qdrant client specifically pointing to the LTM database (qdrant2).
    Falls back to normal QDRANT_URL if QDRANT_LTM_URL is not set.
    """
    url = os.getenv("QDRANT_LTM_URL") or os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    logger.info(f"[LTM] Connecting to Qdrant LTM at: {url}")
    return QdrantClient(url=url, api_key=api_key)

@retry_config("Qdrant Ensure LTM Collection")
def ensure_ltm_collection(client: QdrantClient, collection_name: str):
    """
    Ensures that the LTM collection for a given user exists.
    LTM uses the same BAAI/bge-small-en-v1.5 model for dense semantic similarity.
    """
    try:
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=384,                  # BAAI/bge-small-en-v1.5 dimension
                        distance=Distance.COSINE,
                    )
                }
            )
            logger.info(f"[LTM] Created collection '{collection_name}' in LTM Qdrant")
    except Exception as e:
        logger.error(f"[LTM] Failed to ensure collection: {e}")
        raise

def search_ltm(collection_name: str, query: str, limit: int = 3) -> list[dict]:
    """
    Search the user's long-term memory for relevant profile traits or preferences.
    To ensure crucial identity facts (like the user's name) are always present, we query Qdrant
    and retrieve matches, plus any facts classified under the 'fact' or 'preference' category.
    """
    client = make_qdrant_ltm_client()
    try:
        ensure_ltm_collection(client, collection_name)
        
        # Search semantically
        results = client.query_points(
            collection_name=collection_name,
            query=models.Document(
                text=query,
                model=DENSE_MODEL,
            ),
            using="dense",
            limit=limit,
            with_payload=True
        ).points

        # Fetch generic facts (e.g. user's name) to guarantee they are present in system prompt
        general_facts = []
        try:
            scroll_res = client.scroll(
                collection_name=collection_name,
                limit=10,
                with_payload=True
            )[0]
            for point in scroll_res:
                p = point.payload or {}
                # Capture critical facts like name
                if p.get("category") == "fact" or "name" in p.get("text", "").lower():
                    general_facts.append({
                        "id": point.id,
                        "text": p.get("text", ""),
                        "category": p.get("category", "fact"),
                        "score": 1.0
                    })
        except Exception as e:
            logger.debug(f"[LTM] Generic facts scroll failed: {e}")

        memories = []
        seen_texts = set()
        
        # Add general facts first (like name)
        for gf in general_facts:
            txt = gf["text"]
            if txt and txt not in seen_texts:
                memories.append(gf)
                seen_texts.add(txt)

        # Add semantic search results
        for r in results:
            payload = r.payload or {}
            text = payload.get("text", "")
            if text and text not in seen_texts:
                memories.append({
                    "id": r.id,
                    "text": text,
                    "category": payload.get("category", "preference"),
                    "score": getattr(r, "score", None)
                })
                seen_texts.add(text)

        return memories[:limit + 3]
    except Exception as e:
        logger.error(f"[LTM] Search failed: {e}")
        return []


class MemoryEntity(BaseModel):
    memory_text: str = Field(
        description="A concise summary of user preference, schedule/event, goal, or learning difficulty. E.g., 'User prefers step-by-step calculus explanations.'"
    )
    category: Literal["preference", "schedule", "goal", "difficulty", "fact"] = Field(
        description="The type of memory being captured."
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment towards the preference or topic."
    )

class MemoryExtractionResponse(BaseModel):
    memories: List[MemoryEntity] = Field(
        default_factory=list,
        description="List of extracted memories from the conversation."
    )

def extract_and_save_memories(messages: list, collection_name: str):
    """
    Extracts user profile details, preferences, and episodic memories from the conversation history,
    then saves them to Qdrant LTM collection while resolving conflicts.
    """
    if not messages:
        return

    try:
        # 1. Format messages into text for the LLM
        formatted_history = []
        for m in messages:
            # Handle both dictionary formats and LangChain message objects
            if isinstance(m, dict):
                role = m.get("role", "unknown")
                content = m.get("content", "")
            else:
                # Fallback for LangChain BaseMessage / Pydantic objects
                role = getattr(m, "type", getattr(m, "role", "unknown"))
                content = getattr(m, "content", "")
                
            # Normalize role names
            if role in ("human", "user"):
                role = "user"
            elif role in ("ai", "assistant"):
                role = "assistant"
                
            if content and role in ("user", "assistant"):
                formatted_history.append(f"{role.capitalize()}: {content}")
        
        history_text = "\n".join(formatted_history)

        # 2. Invoke ChatGroq with structured output
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        structured_llm = llm.with_structured_output(MemoryExtractionResponse)

        prompt = (
            "Analyze the conversation transcript below. Identify and extract any long-term information "
            "about the user: their name, identity details, learning preferences, explicit schedules/tasks (e.g. exams, quizzes, assignment deadlines), "
            "academic goals, or persistent conceptual difficulties. "
            "Be specific (e.g. 'User's name is Aditya', 'User has a math quiz on Monday'). "
            "Ignore temporary status updates, general chit-chat, or greetings.\n\n"
            f"Conversation:\n{history_text}\n\n"
            "Return a list of extracted memories. If nothing of long-term importance is found, return an empty list."
        )

        response: MemoryExtractionResponse = structured_llm.invoke(prompt)
        extracted = response.memories if response else []

        if not extracted:
            logger.debug("[LTM] No memories extracted from conversation chunk.")
            return

        client = make_qdrant_ltm_client()
        ensure_ltm_collection(client, collection_name)

        # 3. For each extracted memory, perform conflict resolution
        for item in extracted:
            text = item.memory_text.strip()
            if not text:
                continue

            # Query Qdrant for semantic duplicates/conflicts
            existing_matches = search_ltm(collection_name, text, limit=3)
            conflict_point_id = None

            for match in existing_matches:
                score = match.get("score") or 0.0
                # If similarity is highly relevant (>= 0.70), check if they conflict or overlap
                if score >= 0.70:
                    conflict_point_id = match.get("id")
                    break

            # If an overlap or conflict is found, let's ask LLM to merge them
            if conflict_point_id:
                try:
                    # Fetch current stored value
                    stored_matches = client.retrieve(
                        collection_name=collection_name,
                        ids=[conflict_point_id],
                        with_payload=True
                    )
                    if stored_matches:
                        old_text = stored_matches[0].payload.get("text", "")
                        
                        merge_prompt = (
                            "You are a memory consolidation engine.\n"
                            f"Existing Memory: '{old_text}'\n"
                            f"New Incoming Observation: '{text}'\n\n"
                            "Synthesize these two observations into a single updated, consolidated fact. "
                            "If the new observation makes the old one outdated (e.g., test dates changed, preferences updated), "
                            "use the new information. Keep the response under 15 words.\n"
                            "Consolidated Memory:"
                        )
                        merge_res = llm.invoke(merge_prompt)
                        text = merge_res.content.strip()
                except Exception as e:
                    logger.error(f"[LTM] Failed to consolidate memory: {e}")

            point_id = conflict_point_id or str(uuid.uuid4())
            
            # Upsert into Qdrant LTM
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": models.Document(text=text, model=DENSE_MODEL)
                        },
                        payload={
                            "text": text,
                            "category": item.category,
                            "sentiment": item.sentiment,
                        }
                    )
                ]
            )
            logger.info(f"[LTM] Upserted memory item: '{text}' (ID: {point_id})")

    except Exception as e:
        logger.error(f"[LTM] Memory extraction task failed: {e}", exc_info=True)


