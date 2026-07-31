import os
import sys

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import config
from src.rag.loaders import load_directory
from src.rag.chunking import chunk_documents
from src.rag.embeddings import SentenceTransformerEmbeddings
from src.rag.vector_store import VectorStore

def main():
    print("=" * 60)
    print("CeylonGuide AI - Knowledge Base Ingestion Pipeline")
    print("=" * 60)
    
    # 1. Setup paths
    raw_dir = "data/raw"
    chroma_dir = config.CHROMA_DIR
    
    print(f"Reading raw documents from: {raw_dir}")
    print(f"ChromaDB target directory:  {chroma_dir}")
    
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        print(f"Error: Raw data directory '{raw_dir}' is empty or does not exist.")
        sys.exit(1)
        
    # 2. Load documents
    print("Loading documents...")
    documents = load_directory(raw_dir)
    print(f"Successfully loaded {len(documents)} document(s).")
    
    # 3. Chunk documents
    print(f"Chunking documents (Size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP})...")
    chunks = chunk_documents(documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"Generated {len(chunks)} text chunks.")
    
    # 4. Initialize embedding model and vector store
    print(f"Loading local embedding model: {config.EMBEDDING_MODEL}...")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
        # Trigger lazy load
        embeddings._lazy_init()
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Please ensure sentence-transformers and PyTorch are installed.")
        sys.exit(1)
        
    print("Initializing ChromaDB collection...")
    vector_store = VectorStore(persist_dir=chroma_dir, embedding_model=embeddings)
    
    # Reset collection to avoid duplicates
    print("Resetting existing collection if any...")
    vector_store.reset_store()
    
    # 5. Index chunks
    print("Embedding and indexing chunks into ChromaDB (this might take a minute)...")
    try:
        vector_store.add_documents(chunks)
        print("=" * 60)
        print("INGESTION SUCCESSFUL SUMMARY:")
        print(f"- Documents loaded:       {len(documents)}")
        print(f"- Chunks generated:       {len(chunks)}")
        print(f"- Embedding model used:   {config.EMBEDDING_MODEL}")
        print(f"- Vector store location:  {chroma_dir}")
        print(f"- ChromaDB collections:  {vector_store.collection_name}")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"INGESTION FAILED: {e}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
