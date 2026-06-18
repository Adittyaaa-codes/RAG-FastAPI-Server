from src.utils.utility import make_qdrant_client

async def list_docs(collection_name):
    try:
        try:
            qdrant_client = make_qdrant_client()
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
        return {
            "error": "Listing failed",
            "detail": str(e),
        }