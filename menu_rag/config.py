"""Environment-backed settings.

Values come from rag/.env; DATABASE_URL falls back to backend/.env so the two
projects can't drift apart on which database they point at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BACKEND_ENV = ROOT.parent / "backend" / ".env"

load_dotenv(ROOT / ".env")
if BACKEND_ENV.exists():
    # Only fills in variables .env didn't already set.
    load_dotenv(BACKEND_ENV)


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    mistral_api_key: str
    mistral_model: str
    mistral_chat_model: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_namespace: str
    embed_batch_size: int
    embed_batch_delay: float


def load_settings() -> Settings:
    return Settings(
        database_url=_require("DATABASE_URL"),
        mistral_api_key=_require("MISTRAL_API_KEY"),
        mistral_model=os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed"),
        mistral_chat_model=os.getenv("MISTRAL_CHAT_MODEL", "ministral-8b-latest"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index=os.getenv("PINECONE_INDEX", "milli-milli-menu"),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "items"),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        embed_batch_delay=float(os.getenv("EMBED_BATCH_DELAY_SECONDS", "1.0")),
    )
