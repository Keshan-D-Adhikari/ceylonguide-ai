from src.rag.loaders import Document

def validate_settings(chunk_size: int, chunk_overlap: int):
    """Validates chunk size and overlap settings."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

def chunk_document(doc: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """
    Splits a single Document into smaller Document chunks based on characters.
    Ensures that split is done cleanly on spaces where possible.
    """
    validate_settings(chunk_size, chunk_overlap)
    
    text = doc.page_content
    metadata = doc.metadata.copy()
    chunks = []
    
    # If content fits in a single chunk, return it
    if len(text) <= chunk_size:
        metadata["chunk_index"] = 0
        chunks.append(Document(page_content=text, metadata=metadata))
        return chunks
        
    start = 0
    chunk_index = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we are not at the end of the text, try to find a clean break point (space or newline)
        if end < len(text):
            # Look backwards up to 100 characters for a space/newline to split
            break_index = text.rfind(" ", start + chunk_size - 100, end)
            if break_index == -1:
                # If no space in the last 100 chars, try to look at newlines
                break_index = text.rfind("\n", start + chunk_size - 100, end)
            if break_index != -1 and break_index > start:
                end = break_index
                
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))
            chunk_index += 1
            
        # Advance starting point based on actual end and overlap
        start = end - chunk_overlap
        if start >= len(text) or end >= len(text):
            break
            
    return chunks

def chunk_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """Splits a list of Documents into chunks."""
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
