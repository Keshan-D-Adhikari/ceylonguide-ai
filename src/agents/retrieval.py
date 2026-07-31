import logging
from src.rag.embeddings import SentenceTransformerEmbeddings
from src.rag.vector_store import VectorStore
from src.rag.retriever import TourismRetriever
from src.utils import config

logger = logging.getLogger(__name__)

# Initialize singletons for embeddings and vector store
try:
    _embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    _vector_store = VectorStore(persist_dir=config.CHROMA_DIR, embedding_model=_embeddings)
    _retriever = TourismRetriever(vector_store=_vector_store, top_k=config.TOP_K)
except Exception as e:
    logger.error(f"Failed to initialize RAG components: {e}")
    _retriever = None

def retrieve_tourism_info(user_query: str, interests: list[str] = None, top_k: int = None) -> list[dict]:
    """
    Retrieves relevant tourism documents from ChromaDB based on query and interests.
    Used by the Destination Retrieval Agent.
    """
    if _retriever is None:
        logger.error("Retriever is not initialized.")
        return []
        
    # Combine user query and interests to formulate a strong semantic query
    search_terms = [user_query]
    if interests:
        search_terms.extend(interests)
    combined_query = " ".join(search_terms)
    
    try:
        results = _retriever.retrieve(combined_query, top_k=top_k)
        return results
    except Exception as e:
        logger.error(f"Retrieval agent error: {e}")
        return []
