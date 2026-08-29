"""
LLM Abstraction Layer for MediRAG.
Supports Google Gemini API (Primary) and local Ollama instance (Optional).
"""

import os
from typing import Optional
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class LLMClient:
    """Unified client for invoking Gemini or Ollama."""

    def __init__(self, provider: str = "gemini", model_name: Optional[str] = None):
        self.provider = provider.lower()

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
            
            self.client = genai.Client(api_key=api_key)
            # Use a verified model from your active API list
            self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

        elif self.provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'gemini' or 'ollama'.")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        """Generates a text response from the configured LLM."""
        if self.provider == "gemini":
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    )
                )
                return response.text.strip()
            except Exception as e:
                return f"Gemini API Error: {str(e)}"

        elif self.provider == "ollama":
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
                res = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
                res.raise_for_status()
                return res.json().get("response", "").strip()
            except Exception as e:
                return f"Ollama Connection Error: {str(e)}. Ensure Ollama is running on {self.base_url}."