import os
import jwt
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from src.utils.utility import make_qdrant_client, ensure_collection, embed_text, embed_texts, load_and_chunk, logger, retry_config
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer()

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials

    if not token or token.strip() == "":
        raise HTTPException(status_code=401, detail="Authorization token is missing or empty")

    if token.count(".") != 2:
        raise HTTPException(status_code=401, detail="Malformed token: a valid JWT must have 3 dot-separated segments")

    secret_key = os.getenv("JWT_SECRET")
    if not secret_key:
        raise HTTPException(status_code=500, detail="Server misconfiguration: JWT_SECRET is not set")

    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    user_id = payload.get("_id") or payload.get("user_id") or payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token payload")

    return f"user_{user_id}".replace("-", "_")


from src.rag.index import index
@app.post("/index")
async def index_doc(
    file: UploadFile = File(...),
    subject: str = Form(...),
    chapter: str = Form(...),
    collection_name: str = Depends(get_current_user_id)
):
    
    return await index(file=file,collection_name=collection_name,subject=subject,chapter=chapter)


from src.rag.list_docs import list_docs
@app.get("/list_docs")
async def list_documents(collection_name: str = Depends(get_current_user_id)):
    return await list_docs(collection_name=collection_name)


from src.rag.delete_docs import delete_docs

class DeleteDocsRequest(BaseModel):
    subject: str
    chapter: Optional[str] = None

@app.delete("/delete_docs")
async def delete_by_hierarchy(
    req: DeleteDocsRequest,
    collection_name: str = Depends(get_current_user_id)
):
    """
    Delete documents by subject and optional chapter hierarchy.
    Used by Node backend to sync deletions with Qdrant when subjects/chapters are deleted.
    """
    return await delete_docs(
        collection_name=collection_name,
        subject=req.subject,
        chapter=req.chapter
    )

class ChatRequest(BaseModel):
    query: str
    subject: Optional[str] = None
    chapter: Optional[str] = None
    messages: list = []
    
from src.rag.chat import chat
@app.post("/chat")
async def chat_endpoint(
    req:ChatRequest,
    collection_name: str = Depends(get_current_user_id)
):
    return await chat(req,collection_name)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)