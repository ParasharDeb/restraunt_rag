"""Embed every row of the Prisma `items` table with Mistral and upsert it into Pinecone.

    python ingest.py              # embed everything
    python ingest.py --limit 20   # smoke test on a handful of rows
    python ingest.py --dry-run    # print the text that would be embedded, call nothing

Vector ids are the Prisma item ids, so re-running updates rows in place.
"""

from __future__ import annotations

import argparse
import time

from menu_rag.config import load_settings
from menu_rag.db import fetch_items
from menu_rag.documents import to_documents
from menu_rag.embeddings import build_embeddings, probe_dimension
from menu_rag.store import ensure_index, get_client, get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index menu items into Pinecone.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N items.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the text that would be embedded without calling Mistral or Pinecone.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings()

    print("Reading items from Postgres...")
    items = fetch_items(settings.database_url)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No items found — nothing to index.")
        return

    documents = to_documents(items)
    print(f"Prepared {len(documents)} documents.")

    if args.dry_run:
        for doc in documents[:5]:
            print("\n" + "-" * 70)
            print(f"id: {doc.id}")
            print(doc.page_content)
            print(f"metadata: {doc.metadata}")
        print(f"\nDry run — {len(documents)} documents ready, nothing sent.")
        return

    embeddings = build_embeddings(settings)
    dimension = probe_dimension(embeddings)
    print(f"Embedding model '{settings.mistral_model}' -> {dimension} dimensions.")

    index = ensure_index(get_client(settings), settings, dimension)
    store = get_vector_store(settings, embeddings, index)

    batch_size = max(1, settings.embed_batch_size)
    total = len(documents)
    done = 0

    for start in range(0, total, batch_size):
        batch = documents[start : start + batch_size]
        store.add_documents(documents=batch, ids=[doc.id for doc in batch])
        done += len(batch)
        print(f"  upserted {done}/{total}")
        if done < total and settings.embed_batch_delay > 0:
            # Mistral's free tier rate limits aggressively; pace the batches.
            time.sleep(settings.embed_batch_delay)

    stats = index.describe_index_stats()
    namespace_count = stats.get("namespaces", {}).get(settings.pinecone_namespace, {}).get("vector_count")
    print(
        f"\nDone. {done} items embedded into index '{settings.pinecone_index}' "
        f"namespace '{settings.pinecone_namespace}' (vector count: {namespace_count})."
    )


if __name__ == "__main__":
    main()
