"""Reads the rows Prisma manages. Prisma has no Python client here, so this
talks to the same Postgres directly and mirrors the column names in
backend/prisma/schema.prisma.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class Item:
    id: str
    name: str
    desc: str | None
    cuisine: str
    course: str
    diet: str
    allergens: str | None
    protein: str
    spice: int
    taste_tags: list[str]
    serves: list[int]


# "desc" is a SQL reserved word, hence the quoting.
ITEMS_QUERY = """
    SELECT id, name, "desc", cuisine, course, diet, allergens,
           protein, spice, taste_tags, serves
    FROM items
    ORDER BY name
"""


def fetch_items(database_url: str) -> list[Item]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(ITEMS_QUERY)
            rows = cur.fetchall()

    return [
        Item(
            id=row["id"],
            name=row["name"],
            desc=row["desc"],
            cuisine=str(row["cuisine"]),
            course=str(row["course"]),
            diet=str(row["diet"]),
            allergens=row["allergens"],
            protein=str(row["protein"]),
            spice=row["spice"] or 0,
            taste_tags=list(row["taste_tags"] or []),
            serves=list(row["serves"] or []),
        )
        for row in rows
    ]
