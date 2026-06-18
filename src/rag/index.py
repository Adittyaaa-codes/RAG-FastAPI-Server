import os
import uuid
import tempfile

from fastapi import HTTPException
from qdrant_client import models
from qdrant_client.models import PointStruct
from src.utils.utility import load_and_chunk, ensure_collection, make_qdrant_client

# FastEmbed model IDs — must match what's used in rag.py hybrid search
DENSE_MODEL  = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

async def index(file, collection_name: str, subject: str, chapter: str):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, file.filename)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        chunks = load_and_chunk(tmp_path)
        texts  = [chunk.page_content for chunk in chunks]

        if not texts:
            return {"message": "No content found to index", "chunks": 0, "total_in_db": 0}

        qdrant_client = make_qdrant_client()
        ensure_collection(qdrant_client, collection_name)

        # Qdrant's FastEmbed handles embedding locally — no external API call needed.
        # models.Document tells Qdrant which model to use per vector space.
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense":  models.Document(text=text, model=DENSE_MODEL),
                    "sparse": models.Document(text=text, model=SPARSE_MODEL),
                },
                payload={
                    "text":             text,
                    "subject":          subject,
                    "chapter":          chapter,
                    "source":           file.filename,
                    "user_collection":  collection_name,
                },
            )
            for text in texts
        ]

        qdrant_client.upload_points(
            collection_name=collection_name,
            points=points,
        )

        count = qdrant_client.count(collection_name=collection_name).count

        return {
            "message":    "indexed",
            "chunks":     len(points),
            "total_in_db": count,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
