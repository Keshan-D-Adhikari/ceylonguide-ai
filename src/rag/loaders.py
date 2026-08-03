import os
import re
import logging

logger = logging.getLogger(__name__)

class Document:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(metadata={self.metadata})"

def parse_yaml_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parses simple YAML front matter from a markdown file.
    Returns metadata dict and remaining text content.
    """
    metadata = {}
    remaining_content = content
    
    # Matches starting and ending --- for front matter
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if match:
        front_matter = match.group(1)
        remaining_content = content[match.end():]
        
        current_key = None
        for line in front_matter.split("\n"):
            # Check for list items
            if line.strip().startswith("- "):
                val = line.strip("- ").strip()
                if current_key:
                    if not isinstance(metadata.get(current_key), list):
                        metadata[current_key] = []
                    metadata[current_key].append(val)
                continue
            
            # Check for key-value pair
            kv_match = re.match(r"^(\w+):\s*(.*)$", line)
            if kv_match:
                key = kv_match.group(1).strip()
                val = kv_match.group(2).strip()
                # Clean up quotes
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                metadata[key] = val
                current_key = key
                
    return metadata, remaining_content

def clean_text(text: str) -> str:
    """Performs basic cleaning of whitespace and control characters."""
    text = re.sub(r"\r\n", "\n", text)
    # Collapse multiple consecutive newlines or spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def load_document(file_path: str) -> Document:
    """Loads a markdown or txt file, parses metadata and cleans text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
        
    metadata, body = parse_yaml_frontmatter(raw_content)
    cleaned_body = clean_text(body)
    
    # Enrich metadata with file system properties
    metadata["source_file"] = filename
    if "title" not in metadata:
        metadata["title"] = os.path.splitext(filename)[0].replace("_", " ").title()
        
    return Document(page_content=cleaned_body, metadata=metadata)

def load_directory(dir_path: str) -> list[Document]:
    """Loads all markdown and txt files in a directory."""
    documents = []
    if not os.path.exists(dir_path):
        return documents
        
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".md") or filename.endswith(".txt"):
            file_path = os.path.join(dir_path, filename)
            try:
                doc = load_document(file_path)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                
    return documents
