"""Re-derive spice, taste tags, cuisine, course, protein and diet with Mistral.

    python enrich.py --dry-run --limit 10   # show what would change, write nothing
    python enrich.py --limit 40             # enrich the first 40 items
    python enrich.py                        # enrich everything

Run order matters: `bun run db:seed:items` re-runs the heuristic classifier and
would clobber these values, so the sequence is seed -> enrich -> ingest.
"""

from __future__ import annotations

import argparse
import time

from menu_rag.config import load_settings
from menu_rag.db import fetch_items
from menu_rag.enrichment import build_llm, enrich_batch, to_db_row, write_enrichments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-enrich the menu items table.")
    p.add_argument("--limit", type=int, default=None, help="Only process the first N items.")
    p.add_argument("--batch-size", type=int, default=15, help="Dishes per Mistral call.")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between calls (rate limits).")
    p.add_argument("--dry-run", action="store_true", help="Print proposed changes, write nothing.")
    return p.parse_args()


def describe_change(item, row: dict) -> str | None:
    """One line per field that would actually change, diffed against the
    normalised row so a dry run shows exactly what a real run would write."""
    diffs = []
    if row["spice"] != item.spice:
        diffs.append(f"spice {item.spice}->{row['spice']} (conf {row['conf']:.2f})")
    if row["cuisine"] != item.cuisine:
        diffs.append(f"cuisine {item.cuisine}->{row['cuisine']}")
    if row["course"] != item.course:
        diffs.append(f"course {item.course}->{row['course']}")
    if row["protein"] != item.protein:
        diffs.append(f"protein {item.protein}->{row['protein']}")
    if row["diet"] != item.diet:
        diffs.append(f"diet {item.diet}->{row['diet']}")
    if sorted(row["tags"]) != sorted(item.taste_tags):
        diffs.append(f"tags {item.taste_tags}->{row['tags']}")
    return "; ".join(diffs) if diffs else None


def main() -> None:
    args = parse_args()
    settings = load_settings()

    print("Reading items from Postgres...")
    items = fetch_items(settings.database_url)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No items found.")
        return

    llm = build_llm(settings)
    print(f"Enriching {len(items)} items with '{settings.mistral_chat_model}'"
          f"{' (dry run)' if args.dry_run else ''}...\n")

    batch_size = max(1, args.batch_size)
    written = changed = failed = 0

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        try:
            enriched = enrich_batch(llm, batch)
        except Exception as exc:  # one bad batch shouldn't lose the whole run
            failed += len(batch)
            print(f"  [{start + len(batch)}/{len(items)}] batch failed: {exc}")
            continue

        by_id = {i.id: i for i in batch}
        rows = [to_db_row(by_id[item_id], e) for item_id, e in enriched.items()]

        for row in rows:
            change = describe_change(by_id[row["id"]], row)
            if change:
                changed += 1
                if args.dry_run:
                    print(f"  {by_id[row['id']].name[:42]:44} {change}")

        if not args.dry_run:
            written += write_enrichments(settings.database_url, rows)

        done = min(start + batch_size, len(items))
        if not args.dry_run:
            print(f"  [{done}/{len(items)}] written {written}, changed {changed}")
        if done < len(items) and args.delay > 0:
            time.sleep(args.delay)

    verb = "would change" if args.dry_run else "changed"
    print(f"\nDone. {verb} {changed}/{len(items)} items."
          + (f" Wrote {written}." if not args.dry_run else "")
          + (f" {failed} failed." if failed else ""))
    if not args.dry_run:
        print("Next: python ingest.py  (Pinecone still holds the old metadata until you do)")


if __name__ == "__main__":
    main()
