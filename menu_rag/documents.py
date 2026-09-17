"""Turns an Item row into the sentence we embed plus Pinecone-safe metadata.

Queries arrive as natural language ("something spicy and vegetarian to share"),
so the embedded text is prose rather than a field dump — the enum values are
spelled out the way a diner would say them.
"""

from __future__ import annotations

from langchain_core.documents import Document

from .db import Item

DIET_LABELS = {
    "Vegeterian": "vegetarian",
    "Non_vegeterian": "non-vegetarian",
    "Eggeterian": "eggetarian (contains egg)",
    "OnlyFish": "pescatarian (fish only)",
    "Jain": "Jain (no onion, no garlic, no root vegetables)",
}

COURSE_LABELS = {
    "Starter": "starter / appetiser",
    "MainCourse": "main course",
    "Bread": "bread",
    "Salad": "salad",
    "Dessert": "dessert",
    "Beverage": "beverage / drink",
    "Alcohol": "alcoholic drink",
    "Shisha": "shisha / hookah",
    "Sides": "side dish",
}

SPICE_LABELS = {
    0: "not spicy at all",
    1: "very mildly spiced",
    2: "mildly spiced",
    3: "medium spicy",
    4: "hot and spicy",
    5: "very spicy",
}


# Below this, we say nothing about heat rather than guess.
SPICE_MIN_CONFIDENCE = 0.5


def _serves_phrase(serves: list[int]) -> str | None:
    if not serves:
        return None
    low, high = min(serves), max(serves)
    if low == high:
        return f"Serves {low} {'person' if low == 1 else 'people'}."
    return f"Serves {low} to {high} people."


def build_text(item: Item) -> str:
    """The string that gets embedded."""
    parts: list[str] = [f"{item.name}."]

    if item.desc and item.desc.strip():
        parts.append(item.desc.strip().rstrip(".") + ".")

    course = COURSE_LABELS.get(item.course, item.course)
    article = "An" if course[0].lower() in "aeiou" else "A"
    parts.append(f"{article} {course} from the {item.cuisine} menu.")
    parts.append(f"Suitable for {DIET_LABELS.get(item.diet, item.diet)} diners.")

    if item.protein != "None":
        parts.append(f"Main protein: {item.protein.lower()}.")

    # Only state heat when the enrichment was reasonably sure. The old classifier
    # defaulted everything to 0, so every dish claimed "not spicy at all" -- an
    # assertion that was simply false for a tikka masala and poisoned retrieval.
    if item.spice_confidence is not None and item.spice_confidence >= SPICE_MIN_CONFIDENCE:
        parts.append(f"Spice level {item.spice} out of 5, {SPICE_LABELS.get(item.spice, 'medium spicy')}.")

    if item.taste_tags:
        parts.append(f"Tastes {', '.join(tag.lower() for tag in item.taste_tags)}.")

    serves = _serves_phrase(item.serves)
    if serves:
        parts.append(serves)

    if item.allergens and item.allergens.strip():
        parts.append(f"Allergens and tags: {item.allergens.strip()}.")

    return " ".join(parts)


def build_metadata(item: Item) -> dict:
    """Pinecone only stores str / number / bool / list-of-str, so serves is
    flattened into strings plus numeric min/max that filters can range over."""
    metadata: dict = {
        "item_id": item.id,
        "name": item.name,
        "cuisine": item.cuisine,
        "course": item.course,
        "diet": item.diet,
        "protein": item.protein,
        "spice": item.spice,
        "spice_confidence": item.spice_confidence if item.spice_confidence is not None else 0.0,
        "taste_tags": [tag for tag in item.taste_tags if tag],
        "serves": [str(n) for n in item.serves],
    }
    if item.serves:
        metadata["serves_min"] = min(item.serves)
        metadata["serves_max"] = max(item.serves)
    if item.desc and item.desc.strip():
        metadata["desc"] = item.desc.strip()
    if item.allergens and item.allergens.strip():
        metadata["allergens"] = item.allergens.strip()
    return metadata


def to_document(item: Item) -> Document:
    # id == the Prisma row id, so re-running the ingest upserts instead of duplicating.
    return Document(id=item.id, page_content=build_text(item), metadata=build_metadata(item))


def to_documents(items: list[Item]) -> list[Document]:
    return [to_document(item) for item in items]
