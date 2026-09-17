"""Search the Pinecone index — the quickest way to sanity-check an ingest.

    python query.py "something spicy and vegetarian to share"
    python query.py "light starter" --k 10 --course Starter --diet Vegeterian
"""

from __future__ import annotations

import argparse

from menu_rag.config import load_settings
from menu_rag.embeddings import build_embeddings
from menu_rag.store import get_client, get_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Similarity search over the menu index.")
    parser.add_argument("query", help="Natural language query.")
    parser.add_argument("--k", type=int, default=5, help="Number of results.")
    parser.add_argument("--course", default=None, help="Filter by CourseEnum value, e.g. Starter.")
    parser.add_argument("--diet", default=None, help="Filter by DietPrefEnum value, e.g. Vegeterian.")
    parser.add_argument("--cuisine", default=None, help="Filter by CuisineEnum value, e.g. Indian.")
    parser.add_argument("--max-spice", type=int, default=None, help="Only items at or below this spice level.")
    return parser.parse_args()


def build_filter(args: argparse.Namespace) -> dict | None:
    filters: dict = {}
    if args.course:
        filters["course"] = args.course
    if args.diet:
        filters["diet"] = args.diet
    if args.cuisine:
        filters["cuisine"] = args.cuisine
    if args.max_spice is not None:
        filters["spice"] = {"$lte": args.max_spice}
    return filters or None


def main() -> None:
    args = parse_args()
    settings = load_settings()

    embeddings = build_embeddings(settings)
    index = get_client(settings).Index(settings.pinecone_index)
    store = get_vector_store(settings, embeddings, index)

    results = store.similarity_search_with_score(args.query, k=args.k, filter=build_filter(args))

    if not results:
        print("No matches.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"\n{rank}. {meta.get('name')}  (score {score:.4f})")
        print(f"   {meta.get('course')} | {meta.get('cuisine')} | {meta.get('diet')} | spice {meta.get('spice')}")
        print(f"   {doc.page_content[:200]}")


if __name__ == "__main__":
    main()
