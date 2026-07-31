import logging
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

class TourismRetriever:
    """Retriever class that coordinates query embedding, semantic search, and formatting."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k
        
    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieves top K semantically matching tourism chunks.
        Returns structured dicts containing:
          - content: str
          - metadata: dict
          - score: float
        """
        limit = top_k if top_k is not None else self.top_k
        
        try:
            if self.vector_store.is_empty():
                logger.warning("Vector store is empty. No documents to retrieve.")
                return []
                
            results = self.vector_store.similarity_search(query, top_k=limit)
            return results
        except Exception as e:
            logger.error(f"Error during semantic retrieval: {e}", exc_info=True)
            return []
            
    def format_evidence(self, results: list[dict]) -> str:
        """Helper to format retrieval results into a readable text block for LLM prompts."""
        if not results:
            return "No matching evidence found in the tourism knowledge base."
            
        formatted_blocks = []
        for idx, res in enumerate(results):
            meta = res.get("metadata", {})
            title = meta.get("title", meta.get("destination", "Unknown Destination"))
            filename = meta.get("source_file", "unknown")
            score = res.get("score", 0.0)
            
            block = (
                f"Source [{idx+1}]: {title} (File: {filename}, Similarity Score: {score:.2f})\n"
                f"Content: {res['content']}\n"
            )
            formatted_blocks.append(block)
            
        return "\n---\n".join(formatted_blocks)
