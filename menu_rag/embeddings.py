"""Mistral embeddings via LangChain."""

from __future__ import annotations

from langchain_mistralai import MistralAIEmbeddings

from .config import Settings


def build_embeddings(settings: Settings) -> MistralAIEmbeddings:
    return MistralAIEmbeddings(
        api_key=settings.mistral_api_key,
        model=settings.mistral_model,
        max_retries=5,
    )


def probe_dimension(embeddings: MistralAIEmbeddings) -> int:
    """Ask the model for one vector rather than hardcoding 1024, so switching
    MISTRAL_EMBED_MODEL doesn't silently create a mismatched index."""
    return len(embeddings.embed_query("dimension probe"))
