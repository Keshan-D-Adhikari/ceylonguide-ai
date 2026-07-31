import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import config
from src.rag.embeddings import SentenceTransformerEmbeddings
from src.rag.vector_store import VectorStore
from src.rag.retriever import TourismRetriever

def calculate_metrics(retrieved_files: list[str], expected_files: list[str], k: int) -> tuple[float, float, float]:
    """
    Computes Hit@K, Recall@K, and Reciprocal Rank (RR) for a single query.
    """
    top_k_retrieved = retrieved_files[:k]
    
    # 1. Hit@K
    hit = 1.0 if any(item in top_k_retrieved for item in expected_files) else 0.0
    
    # 2. Recall@K (based on unique files retrieved)
    retrieved_set = set(top_k_retrieved)
    expected_set = set(expected_files)
    recall = len(retrieved_set.intersection(expected_set)) / len(expected_set) if expected_set else 1.0
    
    # 3. Reciprocal Rank (RR)
    rr = 0.0
    for idx, item in enumerate(top_k_retrieved):
        if item in expected_files:
            rr = 1.0 / (idx + 1)
            break
            
    return hit, recall, rr

def run_evaluation():
    print("=" * 60)
    print("CeylonGuide AI - RAG Retrieval Offline Evaluation")
    print("=" * 60)
    
    query_file = os.path.join("evaluation", "queries.json")
    if not os.path.exists(query_file):
        print(f"Error: Dataset not found at '{query_file}'")
        sys.exit(1)
        
    # Check if vector DB is loaded
    if not os.path.exists(config.CHROMA_DIR) or len(os.listdir(config.CHROMA_DIR)) == 0:
        print("Error: ChromaDB vector store is empty or does not exist.")
        print("Please ingest the tourism guides first by running:")
        print("  python scripts/ingest.py")
        sys.exit(1)
        
    # Load dataset
    with open(query_file, "r") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} evaluation queries.")
    
    # Initialize retriever
    embeddings = SentenceTransformerEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = VectorStore(persist_dir=config.CHROMA_DIR, embedding_model=embeddings)
    retriever = TourismRetriever(vector_store=vector_store, top_k=config.TOP_K)
    
    total_hits = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    count = len(dataset)
    
    print("-" * 60)
    print(f"{'Query':<40} | {'Hit@K':<6} | {'Recall@K':<8} | {'RR':<5}")
    print("-" * 60)
    
    for item in dataset:
        query = item["query"]
        expected = item["expected_sources"]
        
        # Retrieve top K
        results = retriever.retrieve(query, top_k=config.TOP_K)
        retrieved_files = [res["metadata"].get("source_file") for res in results]
        
        # Calculate metrics for K=config.TOP_K (typically 5)
        hit, recall, rr = calculate_metrics(retrieved_files, expected, config.TOP_K)
        
        total_hits += hit
        total_recall += recall
        total_mrr += rr
        
        # Truncate query for output printing
        short_query = query if len(query) <= 37 else query[:34] + "..."
        print(f"{short_query:<40} | {hit:<6.1f} | {recall:<8.2f} | {rr:<5.2f}")
        
    avg_hit = total_hits / count
    avg_recall = total_recall / count
    mrr = total_mrr / count
    
    print("=" * 60)
    print("AGGREGATE RETRIEVAL METRICS:")
    print(f"- Average Hit@{config.TOP_K}:     {avg_hit * 100:.1f}%")
    print(f"- Average Recall@{config.TOP_K}:  {avg_recall * 100:.1f}%")
    print(f"- Mean Reciprocal Rank (MRR): {mrr:.3f}")
    print("=" * 60)

if __name__ == "__main__":
    run_evaluation()
