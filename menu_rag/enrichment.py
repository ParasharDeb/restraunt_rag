"""LLM enrichment for menu items.

The heuristic classifier in backend/prisma/classifyItem.ts cannot express the
attributes this RAG needs. `detectSpice` returns 0 before running a single regex
for anything outside the Food bucket, which zeroes 366 of 435 rows, and the one
strong signal it has (a literal "spicy" tag) exists on 9 items. The result is
spice 0:421, 1:7, 3:1, 4:6 -- unusable for "spicy" vs "not spicy" queries.

This module asks Mistral to read each dish the way a cook would and fill in
spice, taste tags, cuisine, course, protein and diet. It writes back to the same
Prisma-managed columns, plus `spice_confidence` so that "never assessed" stays
distinguishable from a confident zero.
"""

from __future__ import annotations

import random
import time
from typing import Literal

import psycopg
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from .config import Settings
from .db import Item
from .hardrules import apply_hard_rules

# The LLM is never asked to spell the Prisma diet enum. `Vegeterian` and
# `Non_vegeterian` are misspelled in schema.prisma, and a model told to emit
# them will "correct" them to `Vegetarian` a fair share of the time. It emits
# diner-language tokens instead and we map to the DB spelling here.
DIET_TO_DB = {
    "veg": "Vegeterian",
    "nonveg": "Non_vegeterian",
    "egg": "Eggeterian",
    "fish": "OnlyFish",
    "jain": "Jain",
}

# A closed vocabulary. Left open, the model invents a long tail ("zesty",
# "umami-forward") that nothing can filter on.
TASTE_VOCAB = [
    "spicy", "sweet", "tangy", "smoky", "creamy", "crispy", "cheesy",
    "garlicky", "citrusy", "comforting", "refreshing", "rich", "light",
    "savoury", "nutty", "herby", "bitter", "fruity",
]

# The model reaches for US spellings and occasionally echoes a POS merchandising
# tag ("Boozy") instead of a taste. Normalise what we can, drop the rest.
TASTE_ALIASES = {
    "savory": "savoury",
    "zesty": "tangy",
    "sour": "tangy",
    "hot": "spicy",
    "buttery": "rich",
    "creamy rich": "creamy",
    "fresh": "refreshing",
}


def clean_tags(tags: list[str]) -> list[str]:
    """Normalise to the closed vocabulary, dedupe, keep at most 4."""
    out: list[str] = []
    for raw in tags:
        t = TASTE_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if t in TASTE_VOCAB and t not in out:
            out.append(t)
    return out[:4]


CUISINES = ["Indian", "Asian", "Italian", "Continental", "Beverage", "Other"]
COURSES = ["Starter", "MainCourse", "Bread", "Salad", "Dessert",
           "Beverage", "Alcohol", "Shisha", "Sides"]
PROTEINS = ["Chicken", "Mutton", "Fish", "Prawns", "Egg", "Paneer", "Tofu", "None"]

# The model reaches for real-world synonyms and for Course values in the Cuisine
# slot. Map what is unambiguous; anything else falls back to the existing row.
CUISINE_ALIASES = {
    "alcohol": "Beverage", "drinks": "Beverage", "drink": "Beverage",
    "cocktail": "Beverage", "beverages": "Beverage", "bar": "Beverage",
    "shisha": "Other", "hookah": "Other",
    "chinese": "Asian", "thai": "Asian", "japanese": "Asian", "oriental": "Asian",
    "north indian": "Indian", "punjabi": "Indian", "desi": "Indian", "mughlai": "Indian",
    "american": "Continental", "european": "Continental", "french": "Continental",
    "mexican": "Continental", "western": "Continental",
}
COURSE_ALIASES = {
    "main": "MainCourse", "main course": "MainCourse", "maincourse": "MainCourse",
    "entree": "MainCourse", "appetizer": "Starter", "appetiser": "Starter",
    "snack": "Starter", "side": "Sides", "breads": "Bread",
    "drink": "Beverage", "mocktail": "Beverage", "juice": "Beverage",
    "liquor": "Alcohol", "wine": "Alcohol", "beer": "Alcohol", "spirits": "Alcohol",
    "hookah": "Shisha", "desserts": "Dessert",
}


def _pick(value: str, allowed: list[str], aliases: dict[str, str], fallback: str) -> str:
    """Case-insensitive match, then alias, then give up and keep what we had."""
    v = (value or "").strip()
    for a in allowed:
        if v.lower() == a.lower():
            return a
    mapped = aliases.get(v.lower())
    return mapped if mapped in allowed else fallback


Cuisine = Literal["Indian", "Asian", "Italian", "Continental", "Beverage", "Other"]
Course = Literal[
    "Starter", "MainCourse", "Bread", "Salad", "Dessert",
    "Beverage", "Alcohol", "Shisha", "Sides",
]
Protein = Literal["Chicken", "Mutton", "Fish", "Prawns", "Egg", "Paneer", "Tofu", "None"]
Diet = Literal["veg", "nonveg", "egg", "fish", "jain"]


class ItemEnrichment(BaseModel):
    """One dish, as judged by the model.

    The enum-ish fields are plain `str` on purpose. A Literal makes pydantic reject
    the WHOLE batch when the model returns one stray value (it likes to answer
    `cuisine: "Alcohol"`, which is a Course, not a Cuisine). Accepting the string
    and normalising it in Python salvages the other 14 dishes in the batch.
    """

    index: int = Field(description="The item's index from the numbered input list.")
    spice: int = Field(ge=0, le=5, description="0 = no heat, 5 = very hot.")
    spice_confidence: float = Field(
        ge=0.0, le=1.0,
        description="How sure you are about the spice level, given how little the name/description says.",
    )
    taste_tags: list[str] = Field(description="Between 0 and 4 tags from the allowed vocabulary.")
    cuisine: str = Field(description=f"One of: {', '.join(CUISINES)}")
    course: str = Field(description=f"One of: {', '.join(COURSES)}")
    protein: str = Field(description=f"One of: {', '.join(PROTEINS)}")
    diet: str = Field(description="One of: veg, nonveg, egg, fish, jain")


class BatchEnrichment(BaseModel):
    items: list[ItemEnrichment]


SYSTEM_PROMPT = f"""You are a chef cataloguing a restaurant menu in India. For each numbered dish
you are given a name, an optional description, and the merchandising tags the POS carried.

Return one object per dish, echoing its `index`.

SPICE is the most important field, and the reason for this task. Judge heat from what the dish
ACTUALLY IS, not from whether the word "chilli" appears:
- Indian curries, masalas, tikka, kadhai, kolhapuri, achari, vindaloo, rogan josh carry real heat (3-4).
- Tandoori and kebab marinades are moderately spiced (2-3).
- Chilli/Schezwan/peri-peri/Hot Garlic dishes are hot (4-5).
- Jalapeno, wasabi, gochujang, mustard, black pepper all add heat (2-4).
- Breads, desserts, salads, plain rice, most Italian and Continental dishes are 0-1.
- Alcohol, shisha and soft drinks are 0.
Set `spice_confidence` honestly: ~0.9 when the dish name makes the heat obvious, ~0.5 when you are
inferring from cuisine alone, ~0.2 when the name is opaque (e.g. a brand name or "Chef's Special").

TASTE_TAGS: choose ONLY from this list, at most 4, and only ones you are confident about.
Never copy a POS tag (Boozy, Bar, Veg, Trending, Bestseller, New, Chef's Special) into taste_tags:
{", ".join(TASTE_VOCAB)}

COURSE: use Alcohol for spirits/wine/beer/cocktails, Shisha for hookah and paan, Beverage for
soft drinks/juices/mocktails, and the food courses for food. Bottles and 30ml pours are Alcohol.

CUISINE is a FOOD STYLE and never a drink type. "Alcohol", "Shisha" and "Cocktail" are NOT
cuisines. For any drink (spirit, wine, beer, cocktail, juice, soft drink) use cuisine "Beverage".
For shisha/hookah/paan use cuisine "Other".

DIET: "veg" (no meat, no egg), "nonveg" (meat/poultry/seafood), "egg" (egg but no meat),
"fish" (fish/seafood but no other meat), "jain" (no onion, no garlic, no root vegetables).
Note the POS tags are unreliable here -- many meat dishes are tagged "Veg". Trust the dish name.

PROTEIN: the main protein, or "None" for drinks, breads, and vegetable dishes without paneer/tofu."""


def build_llm(settings: Settings, model: str | None = None) -> ChatMistralAI:
    return ChatMistralAI(
        api_key=settings.mistral_api_key,
        model=model or settings.mistral_chat_model,
        temperature=0,
        max_retries=5,
    )


def render_item(idx: int, item: Item) -> str:
    bits = [f"{idx}. {item.name}"]
    if item.desc and item.desc.strip():
        bits.append(f"   description: {item.desc.strip()}")
    if item.allergens and item.allergens.strip():
        bits.append(f"   pos tags: {item.allergens.strip()}")
    return "\n".join(bits)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limited" in text


def enrich_batch(llm, items: list[Item], max_attempts: int = 6) -> dict[str, ItemEnrichment]:
    """Returns {item_id: enrichment} for one batch, keyed back by list position.

    Mistral's free tier rate limits hard enough that the client's own retries get
    exhausted, so back off explicitly: 4s, 8s, 16s, 32s, 64s with jitter.
    """
    listing = "\n".join(render_item(i, it) for i, it in enumerate(items))
    structured = llm.with_structured_output(BatchEnrichment)

    for attempt in range(max_attempts):
        try:
            result = structured.invoke(
                [("system", SYSTEM_PROMPT), ("human", f"Catalogue these {len(items)} dishes:\n\n{listing}")]
            )
            break
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == max_attempts - 1:
                raise
            wait = 2 ** (attempt + 2) + random.uniform(0, 1.5)
            print(f"      rate limited, retrying in {wait:.1f}s (attempt {attempt + 2}/{max_attempts})")
            time.sleep(wait)

    out: dict[str, ItemEnrichment] = {}
    for enriched in result.items:
        if 0 <= enriched.index < len(items):
            out[items[enriched.index].id] = enriched
    return out


DIET_ALIASES = {
    "vegetarian": "veg", "pure veg": "veg", "vegan": "veg",
    "non-veg": "nonveg", "non veg": "nonveg", "nonvegetarian": "nonveg",
    "non-vegetarian": "nonveg", "meat": "nonveg",
    "eggetarian": "egg", "pescatarian": "fish", "seafood": "fish", "onlyfish": "fish",
}


def to_db_row(item: Item, e: ItemEnrichment) -> dict:
    """Normalise one enrichment into a DB-ready row.

    Every field falls back to the item's current value rather than failing, so a
    single odd answer costs one attribute instead of the whole dish.
    """
    diet_token = DIET_ALIASES.get((e.diet or "").strip().lower(), (e.diet or "").strip().lower())
    diet = DIET_TO_DB.get(diet_token, item.diet)

    spice = max(0, min(5, e.spice))
    conf = max(0.0, min(1.0, e.spice_confidence))
    cuisine = _pick(e.cuisine, CUISINES, CUISINE_ALIASES, item.cuisine)
    course = _pick(e.course, COURSES, COURSE_ALIASES, item.course)

    row = {
        "id": item.id,
        "spice": spice,
        "conf": conf,
        "tags": clean_tags(e.taste_tags),
        "cuisine": cuisine,
        "course": course,
        "protein": _pick(e.protein, PROTEINS, {}, item.protein),
        "diet": diet,
    }

    # Where the dish name states a fact outright, the name outranks the model.
    return apply_hard_rules(row, item.name, item.desc, item.allergens)


UPDATE_SQL = """
    UPDATE items SET
        spice            = %(spice)s,
        spice_confidence = %(conf)s,
        taste_tags       = %(tags)s,
        cuisine          = %(cuisine)s::"CuisineEnum",
        course           = %(course)s::"CourseEnum",
        protein          = %(protein)s::"ProteinEnum",
        diet             = %(diet)s::"DietPrefEnum",
        updated_at       = now()
    WHERE id = %(id)s
"""


def write_enrichments(database_url: str, rows: list[dict]) -> int:
    """Writes already-normalised rows (see `to_db_row`) back to Postgres."""
    if not rows:
        return 0

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(UPDATE_SQL, rows)
        conn.commit()
    return len(rows)
