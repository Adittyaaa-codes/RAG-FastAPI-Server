import os
import tempfile
import uuid
import jwt

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from utility import make_qdrant_client, ensure_collection, embed_text, embed_texts, load_and_chunk, logger, retry_config
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant_client = make_qdrant_client()

security = HTTPBearer()

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        secret_key = os.getenv("JWT_SECRET")
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        user_id = payload.get("_id") or payload.get("user_id") or payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User ID not found in token")
        return f"user_{user_id}".replace("-", "_")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

@app.post("/index")
async def index_doc(
    file: UploadFile = File(...),
    subject: str = Form(...),
    chapter: str = Form(...),
    collection_name: str = Depends(get_current_user_id)
):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)

    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    chunks = load_and_chunk(tmp_path)
    texts = [c.page_content for c in chunks]

    print(f"[INDEX] {len(texts)} chunks from '{file.filename}'")

    try:
        embeddings = embed_texts(texts)

        ensure_collection(qdrant_client, collection_name)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text,
                    "subject": subject,
                    "chapter": chapter,
                    "source": file.filename,
                    "user_collection": collection_name
                }
            )
            for text, embedding in zip(texts, embeddings)
        ]

        @retry_config("Qdrant Upload")
        def upload_to_qdrant():
            qdrant_client.upload_points(collection_name=collection_name, points=points)
        
        upload_to_qdrant()

        count = qdrant_client.count(collection_name=collection_name).count
        logger.info(f"[INDEX] Success: {len(points)} points uploaded to {collection_name}. Total: {count}")

        return {"message": "indexed", "chunks": len(points), "total_in_db": count}
    except Exception as e:
        logger.error(f"[INDEX] Pipeline failed: {str(e)}")
        return {"error": "Indexing failed", "detail": str(e)}, 500

class ChatRequest(BaseModel):
    query: str
    subject: str = None
    chapter: str = None

@app.get("/list_docs")
async def list_documents(collection_name: str = Depends(get_current_user_id)):
    try:
        try:
            records, _ = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=True
            )
        except Exception as e:
            if "Not found: Collection" in str(e):
                return {"documents": [], "count": 0}
            raise e
        
        sources = set()
        for record in records:
            if record.payload:
                source = record.payload.get('source')
                if source:
                    sources.add(source)
        
        return {
            "documents": list(sources),
            "count": len(sources)
        }
        
    except Exception as e:
        logger.error(f"[LIST_DOCS] Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_docs/{filename}")
async def delete_document(
    filename: str,
    collection_name: str = Depends(get_current_user_id)
):
    collection_name = collection_name

    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for this user"
        )

    result = qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filename)
                    )
                ]
            )
        )
    )

    if result.status == models.UpdateStatus.COMPLETED:
        return {
            "success": True,
            "filename": filename,
            "collection": collection_name,
            "message": f"'{filename}' deleted successfully"
        }

    raise HTTPException(status_code=500, detail="Deletion did not complete")


@app.post("/chat")
def chat(
    req: ChatRequest,
    collection_name: str = Depends(get_current_user_id)
):
    try:
        query_vector = embed_text(req.query)

        conditions = []
        if req.subject:
            conditions.append(FieldCondition(key="subject", match=MatchValue(value=req.subject)))
        if req.chapter:
            conditions.append(FieldCondition(key="chapter", match=MatchValue(value=req.chapter)))

        search_filter = Filter(must=conditions) if conditions else None

        @retry_config("Qdrant Search")
        def search_qdrant():
            try:
                return qdrant_client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=5,
                    with_payload=True
                ).points
            except Exception as e:
                # If the collection doesn't exist, the user hasn't uploaded anything yet
                if "Not found: Collection" in str(e):
                    return []
                raise

        results = search_qdrant()
        logger.info(f"[RETRIEVAL] Found {len(results)} relevant chunks.")

        context = "No relevant context found." if not results else "\n\n---\n\n".join([r.payload["text"] for r in results])

        prompt = f"""Answer the question using ONLY the context below.
                If the answer is not in the context, say you don't know.

                CONTEXT:
                {context}

                QUESTION: {req.query}"""

        @retry_config("OpenAI Chat Completion")
        def get_llm_response():
            return openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

        response = get_llm_response()
        answer = response.choices[0].message.content
        return {"answer": answer, "chunks_used": len(results)}

    except Exception as e:
        logger.error(f"[CHAT] Pipeline failed: {str(e)}")
        return {"error": "Chat generation failed", "detail": str(e)}, 500


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)
