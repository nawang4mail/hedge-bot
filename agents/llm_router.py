"""
LLM Router — single source of truth for the local LLM connection.

All four agents import get_llm() from here.  Swapping models or backends
(e.g., Ollama → LM Studio) requires changing only this file.
"""
from functools import lru_cache
from langchain_ollama import ChatOllama
from config import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.1) -> ChatOllama:
    """
    Return a cached ChatOllama instance pointed at the local server.

    temperature=0.1 keeps outputs deterministic for trading signals;
    raise to ~0.7 for exploratory research tasks if desired.

    To switch to LM Studio (OpenAI-compatible endpoint), replace with:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            model="local-model",
        )
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=temperature,
        # Keep responses short and structured — critical for token efficiency
        format="json",
    )
