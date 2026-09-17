"""Pinecone index lifecycle and the LangChain vector store wrapper."""

from __future__ import annotations

import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


def get_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(client: Pinecone, settings: Settings, dimension: int):
    """Creates the serverless index on first run; on later runs just checks that
    the existing index's dimension still matches the embedding model."""
    name = settings.pinecone_index

    if not client.has_index(name):
        print(f"Creating Pinecone index '{name}' (dimension={dimension}, metric=cosine)...")
        client.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not client.describe_index(name).status.get("ready", False):
            time.sleep(1)
        print(f"Index '{name}' is ready.")
    else:
        existing = client.describe_index(name).dimension
        if existing != dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {existing} but "
                f"model '{settings.mistral_model}' produces {dimension}. "
                "Delete the index or point PINECONE_INDEX at a new name."
            )
        print(f"Using existing Pinecone index '{name}' (dimension={dimension}).")

    return client.Index(name)


def get_vector_store(settings: Settings, embeddings, index) -> PineconeVectorStore:
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=settings.pinecone_namespace,
    )
