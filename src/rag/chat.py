import json
import asyncio
import os
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq
from src.deep_agents.deep_agents import agent
from src.utils.utility import logger, mongo_db
from src.utils.ltm_utils import search_ltm, extract_and_save_memories
from bson import ObjectId
from fastapi import HTTPException

def generate_rolling_summary(old_messages: list) -> str:
    if not old_messages:
        return ""
    try:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        text_to_summarize = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in old_messages])
        prompt = (
            "Summarize the following chat history. Focus on key facts, user preferences, "
            "and topics discussed. The summary MUST be a maximum of 200 words.\n\n"
            f"{text_to_summarize}"
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate rolling summary: {e}")
        return ""


def _extract_text(content) -> str:
    """
    Safely extract a plain string from any LangChain message content.
    Handles: str, list[str], list[{"type":"text","text":"..."}], etc.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _is_ai_message(msg) -> bool:
    t = type(msg).__name__
    if "AI" in t or "Assistant" in t:
        return True
    if isinstance(msg, dict):
        return msg.get("role") in ("assistant", "ai")
    return False


def _is_tool_message(msg) -> bool:
    t = type(msg).__name__
    if "Tool" in t:
        return True
    if isinstance(msg, dict):
        return msg.get("role") == "tool"
    return False


async def chat(req, collection_name: str):
    """
    Streams agent events as Server-Sent Events (SSE).

    Uses stream_mode="values" so every chunk contains the FULL accumulated
    message list — guaranteeing complete content on every message, including
    the final AIMessage answer.

    Events:
      {"type": "tool_call",   "tool": "rag_search", "query": "..."}
      {"type": "tool_result", "tool": "rag_search", "hits": 5, "has_context": true}
      {"type": "tool_call",   "tool": "web_search",  "query": "..."}
      {"type": "tool_result", "tool": "web_search",  "results": 3}
      {"type": "token",       "content": "..."}
      {"type": "done",        "full_text": "..."}
      {"type": "error",       "detail": "..."}
    """
    ltm_collection_name = f"user_{collection_name}"
    user_id_str = collection_name

    if mongo_db is not None:
        try:
            user_doc = await mongo_db.users.find_one({"_id": ObjectId(user_id_str)})
            if user_doc:
                tokens_used = user_doc.get("tokensUsed", 0)
                token_limit = user_doc.get("tokenLimit", 500000)
                if tokens_used >= token_limit:
                    raise HTTPException(status_code=403, detail="Token limit exhausted. You have used all your allocated AI tokens.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to check token limit for user {user_id_str}: {e}")

    config = {
        "configurable": {
            "collection_name": collection_name,
            "subject": req.subject,
            "chapter": req.chapter,
        }
    }

    def event_generator():
        full_answer_parts: list[str] = []
        processed_up_to: int = 0
        history = []

        try:
            raw_history = req.messages if isinstance(req.messages, list) else []
            clean_history = [
                {"role": m.get("role"), "content": m.get("content", "").strip()}
                for m in raw_history
                if isinstance(m, dict)
                and m.get("role") in ("user", "assistant")
                and m.get("content", "").strip()
            ]
            
            old_messages = clean_history[:-8]
            recent_history = clean_history[-8:]
            
            try:
                ltm_memories = search_ltm(ltm_collection_name, req.query, limit=3)
                if ltm_memories:
                    profile_facts = "\n".join([f"- {m['text']}" for m in ltm_memories])
                    history.append({
                        "role": "system", 
                        "content": f"User Long-Term Profile & Preferences:\n{profile_facts}\nUse these preferences and facts to customize your explanations and answers."
                    })
            except Exception as e:
                logger.error(f"[LTM] Failed to inject user profile: {e}")

            if old_messages:
                summary = generate_rolling_summary(old_messages)
                if summary:
                    history.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
            
            if req.subject or req.chapter:
                scope_msg = "Search scope: "
                if req.subject:
                    scope_msg += f"Subject={req.subject}"
                if req.chapter:
                    scope_msg += f", Chapter={req.chapter}"
                history.append({"role": "system", "content": f"{scope_msg}. Use these identifiers when searching your documents to find the most relevant material for this specific topic."})
            
            history.extend(recent_history)
            history.append({"role": "user", "content": req.query})

            processed_up_to = len(history)

            for chunk in agent.stream(
                {"messages": history},
                config=config,
                stream_mode="values",   
            ):
                if not isinstance(chunk, dict):
                    continue

                all_messages = chunk.get("messages") or []
                if not isinstance(all_messages, list):
                    all_messages = [all_messages]

                new_messages = all_messages[processed_up_to:]
                processed_up_to = len(all_messages)

                for msg in new_messages:
                    if msg is None:
                        continue

                    logger.debug(
                        f"[STREAM] type={type(msg).__name__} "
                        f"tool_calls={bool(getattr(msg, 'tool_calls', None))} "
                        f"content_len={len(_extract_text(getattr(msg, 'content', None)))}"
                    )

                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get("name", "unknown")
                                args = tc.get("args", {}) or {}
                            else:
                                tool_name = getattr(tc, "name", "unknown")
                                args = getattr(tc, "args", {}) or {}
                            if not isinstance(args, dict):
                                args = {}
                            yield _sse({
                                "type": "tool_call",
                                "tool": tool_name,
                                "query": args.get("query", ""),
                            })
                        continue

                    if _is_tool_message(msg):
                        tool_name = getattr(msg, "name", None) or (
                            msg.get("name") if isinstance(msg, dict) else None
                        )
                        content_str = _extract_text(
                            getattr(msg, "content", None) or
                            (msg.get("content") if isinstance(msg, dict) else None)
                        )

                        if tool_name == "rag_search":
                            try:
                                result = json.loads(content_str)
                                hits                  = result.get("total_hits", 0) if isinstance(result, dict) else 0
                                has_ctx               = result.get("has_context", False) if isinstance(result, dict) else False
                                ragas_score           = result.get("ragas_score", 0.0) if isinstance(result, dict) else 0.0
                                ragas_cp              = result.get("ragas_context_precision", 0.0) if isinstance(result, dict) else 0.0
                                ragas_faith           = result.get("ragas_faithfulness", 0.0) if isinstance(result, dict) else 0.0
                                ragas_verdict         = result.get("ragas_verdict", "") if isinstance(result, dict) else ""
                            except Exception:
                                hits, has_ctx, ragas_score, ragas_cp, ragas_faith, ragas_verdict = 0, False, 0.0, 0.0, 0.0, ""
                            yield _sse({
                                "type": "tool_result",
                                "tool": "rag_search",
                                "hits": hits,
                                "has_context": has_ctx,
                                "ragas_score": ragas_score,
                                "ragas_context_precision": ragas_cp,
                                "ragas_faithfulness": ragas_faith,
                                "ragas_verdict": ragas_verdict,
                            })


                        elif tool_name == "web_search":
                            try:
                                result = json.loads(content_str)
                                n = len(result.get("results", [])) if isinstance(result, dict) else 0
                            except Exception:
                                n = 0
                            yield _sse({"type": "tool_result", "tool": "web_search", "results": n})
                        continue

                    if _is_ai_message(msg):
                        raw = getattr(msg, "content", None) or (
                            msg.get("content") if isinstance(msg, dict) else None
                        )
                        text = _extract_text(raw)

                        if not text:
                            extra = getattr(msg, "additional_kwargs", {})
                            text = _extract_text(
                                extra.get("text") or
                                extra.get("output") or
                                extra.get("answer") or ""
                            )

                        if not text:
                            logger.debug("[STREAM] AI message still empty after fallback, skipping")
                            continue

                        current_streamed_len = len(full_answer_parts)
                        new_chars = text[current_streamed_len:]
                        if new_chars:
                            logger.debug(f"[STREAM] Incremental capture: '{new_chars}'")
                            for char in new_chars:
                                full_answer_parts.append(char)
                                yield _sse({"type": "token", "content": char})

            final_text = "".join(full_answer_parts)
            if not final_text:
                logger.error("[STREAM] No answer found in any chunk")
                yield _sse({"type": "error", "detail": "Agent returned an empty response."})
            else:
                yield _sse({"type": "done", "full_text": final_text})
                
                messages_to_analyze = history + [{"role": "assistant", "content": final_text}]
                try:
                    import threading
                    def _bg_task():
                        try:
                            extract_and_save_memories(messages_to_analyze, ltm_collection_name)
                        except Exception as ex:
                            logger.error(f"[LTM] Bg task error: {ex}")
                    
                    threading.Thread(target=_bg_task, daemon=True).start()
                except Exception as e:
                    logger.error(f"[LTM] Background memory extraction dispatch failed: {e}")

            if mongo_db is not None:
                try:
                    prompt_chars = sum(len(m.get("content", "")) for m in history)
                    response_chars = len(final_text)
                    estimated_tokens = max(1, (prompt_chars + response_chars) // 4)
                    
                    await mongo_db.users.update_one(
                        {"_id": ObjectId(user_id_str)},
                        {"$inc": {"tokensUsed": estimated_tokens}}
                    )
                except Exception as e:
                    logger.error(f"Failed to update token usage for user {user_id_str}: {e}")

        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield _sse({"type": "error", "detail": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
