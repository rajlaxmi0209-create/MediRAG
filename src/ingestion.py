"""
Document Ingestion Module for MediRAG.
Handles PDF loading, text extraction per page, and recursive character chunking.
"""

import os
from typing import List, Dict, Any
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.

    Args:
        pdf_path (str): Filepath to the target PDF.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'text', 'page_number', and 'source'.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

    documents = []
    filename = os.path.basename(pdf_path)

    # Open the PDF using the official pymupdf interface
    doc = pymupdf.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()

        # Skip empty pages
        if not text:
            continue

        documents.append({
            "text": text,
            "page_number": page_index + 1,  # 1-based indexing for display
            "source": filename
        })

    doc.close()
    return documents


def chunk_documents(
    extracted_pages: List[Dict[str, Any]],
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits page-level documents into smaller, overlapping chunks while
    preserving source and page metadata.

    Args:
        extracted_pages (List[Dict[str, Any]]): Output from extract_text_from_pdf.
        chunk_size (int): Maximum character length per chunk.
        chunk_overlap (int): Overlap character length between sequential chunks.

    Returns:
        List[Dict[str, Any]]: List of chunk dicts containing text and metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    chunk_counter = 0

    for page_doc in extracted_pages:
        page_text = page_doc["text"]
        source = page_doc["source"]
        page_number = page_doc["page_number"]

        text_pieces = splitter.split_text(page_text)

        for piece in text_pieces:
            cleaned_piece = piece.strip()
            if cleaned_piece:
                chunks.append({
                    "chunk_id": f"{source}_p{page_number}_c{chunk_counter}",
                    "text": cleaned_piece,
                    "source": source,
                    "page_number": page_number
                })
                chunk_counter += 1

    return chunks