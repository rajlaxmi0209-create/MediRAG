"""
Vector Store and Embeddings Module for MediRAG.
Handles local HuggingFace embeddings via BAAI/bge-small-en-v1.5 and ChromaDB persistence.
"""

import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer


class MedicalVectorStore:
    """
    Manages embedding generation and ChromaDB vector collection.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "medirag_knowledge_base",
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        """
        Initializes local embedding model and persistent ChromaDB client.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name

        # Load local embedding model (CPU-friendly, ~130MB)
        self.embedder = SentenceTransformer(model_name)

        # Initialize persistent ChromaDB client
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create the vector collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Generates embeddings for chunks and upserts them into ChromaDB.

        Args:
            chunks (List[Dict[str, Any]]): Processed chunks from ingestion module.

        Returns:
            int: Number of chunks added.
        """
        if not chunks:
            return 0

        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page_number": int(chunk["page_number"])
            }
            for chunk in chunks
        ]

        # Generate dense embeddings (384 dimensions)
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()

        # Store in ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs cosine similarity search for the user query.

        Args:
            query (str): The search query.
            top_k (int): Number of most relevant passages to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chunks with text, metadata, and similarity score.
        """
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_items = []
        if results and results.get("documents") and len(results["documents"]) > 0:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for i in range(len(docs)):
                retrieved_items.append({
                    "text": docs[i],
                    "source": metas[i].get("source", "Unknown"),
                    "page_number": metas[i].get("page_number", 0),
                    "score": round(1.0 - distances[i], 4)
                })

        return retrieved_items