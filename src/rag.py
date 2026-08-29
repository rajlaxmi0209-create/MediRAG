"""
RAG Orchestration Pipeline for MediRAG.
Coordinates retrieval, prompt construction, LLM response, and source formatting.
"""

from typing import Dict, Any, Optional
from src.vector_store import MedicalVectorStore
from src.llm import LLMClient
from src.prompts import build_rag_prompt


class MediRAGPipeline:
    """End-to-end pipeline connecting vector retrieval to LLM generation."""

    def __init__(
        self, 
        vector_store: MedicalVectorStore, 
        llm_provider: str = "gemini",
        model_name: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.llm = LLMClient(provider=llm_provider, model_name=model_name)

    def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a user query.
        """
        retrieved_chunks = self.vector_store.similarity_search(query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found in the database. Please ingest documents first.",
                "sources": []
            }

        context_blocks = []
        for c in retrieved_chunks:
            block = f"[Document: {c['source']} | Page: {c['page_number']}]\n{c['text']}"
            context_blocks.append(block)
        
        context_str = "\n\n---\n\n".join(context_blocks)
        prompt = build_rag_prompt(context=context_str, question=query)
        response_text = self.llm.generate(prompt)

        return {
            "query": query,
            "answer": response_text,
            "sources": retrieved_chunks
        }