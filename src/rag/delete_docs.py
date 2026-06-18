from fastapi import HTTPException
from src.utils.utility import make_qdrant_client
from qdrant_client.models import FilterSelector, Filter, FieldCondition, MatchValue, UpdateStatus
from typing import Optional

async def delete_docs(collection_name: str, subject: str, chapter: Optional[str] = None):
    """
    Delete documents from Qdrant by subject (and optionally by chapter).
    
    Args:
        collection_name: The user's collection name in Qdrant
        subject: The subject name to delete
        chapter: Optional chapter name. If provided, only deletes that chapter's materials.
                If None, deletes all materials for the subject.
    
    Returns:
        Dict with success status and deletion details
    """
    qdrant_client = make_qdrant_client()
    
    try:
        qdrant_client.get_collection(collection_name)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for this user"
        )
    
    # Build filter conditions
    must_conditions = [
        FieldCondition(
            key="subject",
            match=MatchValue(value=subject)
        )
    ]
    
    # If chapter is specified, add it to the filter
    if chapter:
        must_conditions.append(
            FieldCondition(
                key="chapter",
                match=MatchValue(value=chapter)
            )
        )
    
    # Delete matching points
    result = qdrant_client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(must=must_conditions)
        )
    )
    
    if result.status == UpdateStatus.COMPLETED:
        delete_type = f"chapter '{chapter}' in subject '{subject}'" if chapter else f"subject '{subject}'"
        return {
            "success": True,
            "subject": subject,
            "chapter": chapter,
            "collection": collection_name,
            "message": f"Successfully deleted {delete_type}"
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete from Qdrant"
        )
