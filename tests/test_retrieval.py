import os
import unittest
from unittest.mock import MagicMock
from src.rag.loaders import parse_yaml_frontmatter, clean_text, Document, load_document
from src.rag.chunking import chunk_document, validate_settings
from src.rag.embeddings import BaseEmbeddings
from src.rag.vector_store import VectorStore

class MockEmbeddings(BaseEmbeddings):
    """Simple mock embeddings class to avoid downloading model during unit testing."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Return dummy 384-dimensional vectors (standard for MiniLM)
        return [[0.1] * 384 for _ in texts]
        
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384

class TestRetrievalAndParsing(unittest.TestCase):
    
    def test_parse_yaml_frontmatter(self):
        raw_md = """---
title: Test Title
province: Western
categories:
  - beaches
  - relaxation
source_name: Test Source
---
# Main Header
Body text goes here."""
        
        metadata, body = parse_yaml_frontmatter(raw_md)
        
        self.assertEqual(metadata.get("title"), "Test Title")
        self.assertEqual(metadata.get("province"), "Western")
        self.assertEqual(metadata.get("categories"), ["beaches", "relaxation"])
        self.assertEqual(metadata.get("source_name"), "Test Source")
        self.assertIn("Body text goes here.", body)
        self.assertNotIn("title: Test Title", body)

    def test_clean_text(self):
        text = "Hello \t World\n\n\n\nNew Line"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "Hello World\n\nNew Line")

    def test_chunk_validation(self):
        with self.assertRaises(ValueError):
            validate_settings(-10, 5)
        with self.assertRaises(ValueError):
            validate_settings(100, 100)
        with self.assertRaises(ValueError):
            validate_settings(100, 120)

    def test_chunk_document(self):
        content = "One Two Three Four Five Six Seven Eight Nine Ten"
        # Content length is 47 characters
        doc = Document(
            page_content=content, 
            metadata={"title": "Test"}
        )
        
        # Test large chunk size (no split)
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, content)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

        # Test small chunk size (must split)
        chunks_split = chunk_document(doc, chunk_size=20, chunk_overlap=5)
        self.assertTrue(len(chunks_split) > 1)
        # Ensure metadata is preserved
        for chunk in chunks_split:
            self.assertEqual(chunk.metadata["title"], "Test")
            self.assertIn("chunk_index", chunk.metadata)

    def test_vector_store_mock(self):
        # Test vector store initialization and operation using MockEmbeddings
        # to avoid filesystem dependency issues in temporary tests
        mock_emb = MockEmbeddings()
        persist_dir = "data/test_chroma"
        
        # We can mock the client to avoid writing to disk in pytest
        store = VectorStore(persist_dir=persist_dir, embedding_model=mock_emb, collection_name="test_col")
        store.client = MagicMock()
        store.collection = MagicMock()
        
        store.collection.count.return_value = 0
        self.assertTrue(store.is_empty())
        
        # Add test doc
        doc = Document("Sample text", {"source_file": "sample.md", "title": "Sample"})
        store.add_documents([doc])
        
        store.collection.add.assert_called_once()
        
if __name__ == "__main__":
    unittest.main()
