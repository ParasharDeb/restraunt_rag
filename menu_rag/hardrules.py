"""Deterministic guards that outrank the LLM.

ministral-8b is good at judging heat and taste but unreliable on the facts the
dish name states outright. Left alone it labelled "Hot Garlic Fish" and "Fish
Finger" as Vegeterian, "Tandoori Royal Veg Platter" as Non_vegeterian, and filed
"Steam Rice" under course Alcohol.

A vegetarian being recommended fish is not a ranking miss, it is a broken
promise, so where the name is unambiguous the name wins.
"""

from __future__ import annotations

import re

# Deliberately does not include "egg": egg is handled separately, and several
# vegetarian dishes mention it only in a description of the batter.
MEAT_RE = re.compile(
    r"\b(chicken|murgh|mutton|lamb|keema|tangdi|fish|prawn|shrimp|crab|"
    r"bacon|ham|pepperoni|salami|meat|seekh)\b",
    re.I,
)
FISH_ONLY_RE = re.compile(r"\b(fish|prawn|shrimp|crab)\b", re.I)
EGG_RE = re.compile(r"\begg(s)?\b", re.I)
# "(N-V)" and "Non Veg" mark a non-vegetarian variant of an otherwise veg dish.
NONVEG_MARK_RE = re.compile(r"non[\s_-]*veg|\(\s*n\s*-?\s*v\s*\)", re.I)
VEG_MARK_RE = re.compile(r"\bveg\b|\bvegetarian\b|\(\s*veg\s*\)", re.I)

FOOD_COURSES = {"Starter", "MainCourse", "Bread", "Salad", "Dessert", "Sides"}
DRINK_COURSES = {"Beverage", "Alcohol", "Shisha"}

# Used only to rescue a food dish that the model filed under a drink course.
COURSE_HINTS = [
    (re.compile(r"\b(naan|roti|kulcha|paratha|bread|laccha)\b", re.I), "Bread"),
    (re.compile(r"\b(salad)\b", re.I), "Salad"),
    (re.compile(r"\b(cake|tiramisu|brownie|ice\s*cream|kulfi|dessert|pudding)\b", re.I), "Dessert"),
    (re.compile(r"\b(rice|pulao|biryani|curry|gravy|dal|makhni|steak|pasta|pizza|noodle)\b", re.I), "MainCourse"),
    (re.compile(r"\b(fries|platter|nachos|wings|pops|tikki|kebab|tikka|wonton|dimsum|bao)\b", re.I), "Starter"),
]


def apply_hard_rules(row: dict, name: str, desc: str | None, allergens: str | None) -> dict:
    """Overrides `row` (a to_db_row result) wherever the source text is decisive."""
    text = f"{name} {desc or ''}"
    tags = (allergens or "").lower()
    out = dict(row)

    # 1. The POS tags are authoritative about what the thing IS.
    if "boozy" in tags:
        out.update(course="Alcohol", cuisine="Beverage", spice=0, conf=1.0)
        return out
    if "lounge" in tags:
        out.update(course="Shisha", cuisine="Other", spice=0, conf=1.0)
        return out

    # 2. Diet. Order matters: an explicit marker in the NAME is the strongest
    # signal there is, and it must beat meat words found in a description.
    # "Tandoori Royal Veg Platter" describes a "mushroom veg seekh kebab" --
    # reading `seekh` as meat there turned a veg platter non-vegetarian.
    if NONVEG_MARK_RE.search(name):
        out["diet"] = "Non_vegeterian"
    elif VEG_MARK_RE.search(name):
        out["diet"] = "Vegeterian"
        if out["protein"] in ("Chicken", "Mutton", "Fish", "Prawns"):
            out["protein"] = "None"
    elif MEAT_RE.search(text):
        only_fish = bool(FISH_ONLY_RE.search(text)) and not re.search(
            r"(chicken|murgh|mutton|lamb|keema|tangdi|bacon|ham|pepperoni|salami|seekh)",
            text, re.I,
        )
        out["diet"] = "OnlyFish" if only_fish else "Non_vegeterian"
        if out["protein"] == "None":
            for pattern, protein in (
                (r"chicken|murgh|tangdi", "Chicken"), (r"mutton|lamb|keema|seekh", "Mutton"),
                (r"prawn|shrimp", "Prawns"), (r"fish|crab", "Fish"),
            ):
                if re.search(pattern, text, re.I):
                    out["protein"] = protein
                    break
    elif EGG_RE.search(text) and out["diet"] == "Vegeterian":
        out["diet"] = "Eggeterian"

    # 3. A dish with no drink tag does not belong in a drink course.
    if out["course"] in DRINK_COURSES:
        for pattern, course in COURSE_HINTS:
            if pattern.search(name):
                out["course"] = course
                if out["cuisine"] == "Beverage":
                    out["cuisine"] = "Indian" if re.search(
                        r"\b(naan|roti|kulcha|dal|makhni|tikka|kebab|biryani|pulao|masala|paneer)\b", name, re.I
                    ) else "Other"
                break

    return out
