from abc import ABC, abstractmethod

class BaseEmbeddings(ABC):
    """Abstract base class for creating and retrieving embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed list of texts."""
        pass
        
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        pass

class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Local embeddings wrapper using Sentence-Transformers from Hugging Face."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        
    def _lazy_init(self):
        """Lazily initialize the model to save startup time when not needed."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package is missing. Please install it using 'pip install sentence-transformers'."
                )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._lazy_init()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [list(map(float, emb)) for emb in embeddings]

    def embed_query(self, text: str) -> list[float]:
        self._lazy_init()
        embedding = self.model.encode(text, show_progress_bar=False)
        return list(map(float, embedding))
