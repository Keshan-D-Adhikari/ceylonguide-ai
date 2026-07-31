import os
import chromadb
from src.rag.loaders import Document
from src.rag.embeddings import BaseEmbeddings

class VectorStore:
    """Wrapper around ChromaDB for storing and retrieving document embeddings."""
    
    def __init__(self, persist_dir: str, embedding_model: BaseEmbeddings, collection_name: str = "ceylonguide_tourism"):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
    def _init_db(self):
        """Initializes the ChromaDB client and gets or creates the collection."""
        if self.client is None:
            # Ensure path exists
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Use cosine similarity
            )

    def is_empty(self) -> bool:
        """Checks if the collection is empty or doesn't exist yet."""
        try:
            self._init_db()
            return self.collection.count() == 0
        except Exception:
            return True

    def add_documents(self, documents: list[Document]):
        """Embeds and adds documents to the vector store."""
        if not documents:
            return
            
        self._init_db()
        
        texts = [doc.page_content for doc in documents]
        # Generate embeddings
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Build collections payload
        ids = []
        metadatas = []
        for idx, doc in enumerate(documents):
            # Create a unique ID
            source_file = doc.metadata.get("source_file", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", idx)
            doc_id = f"{source_file}_chunk_{chunk_idx}"
            ids.append(doc_id)
            
            # Flatten lists in metadata for ChromaDB (which only supports primitive types)
            meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, list):
                    meta[k] = ",".join(v)
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)
            
        # Add to Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    def similarity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs a semantic search in ChromaDB."""
        self._init_db()
        
        if self.is_empty():
            return []
            
        query_vector = self.embedding_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        # Format output
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            ids = results["ids"][0]
            
            for idx in range(len(docs)):
                # Convert flattened categories back to list
                meta = metadatas[idx].copy()
                if "categories" in meta and isinstance(meta["categories"], str):
                    meta["categories"] = [cat.strip() for cat in meta["categories"].split(",") if cat.strip()]
                    
                formatted_results.append({
                    "id": ids[idx],
                    "content": docs[idx],
                    "metadata": meta,
                    "distance": distances[idx],
                    "score": 1.0 - distances[idx] # Simple score mapping for cosine distance
                })
                
        return formatted_results

    def reset_store(self):
        """Clears the collection to re-ingest data."""
        self._init_db()
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
