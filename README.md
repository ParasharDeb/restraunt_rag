# milli-milli RAG

Embeds the menu (`items` table in the Prisma/Neon database) with **Mistral**
via **LangChain**, and upserts the vectors into **Pinecone**.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

Fill in `.env`:

| Variable | Where to get it |
| --- | --- |
| `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| `PINECONE_API_KEY` | https://app.pinecone.io -> API Keys |
| `DATABASE_URL` | inherited from `../backend/.env`; set here only to override |

The Pinecone index is created on first run (serverless, cosine, dimension
probed from the embedding model - 1024 for `mistral-embed`).

## Use

Run order matters. `bun run db:seed:items` re-runs the heuristic classifier in
`backend/prisma/classifyItem.ts` and would clobber the enrichment, so:

**seed -> enrich -> ingest**

```bash
.venv/Scripts/python.exe enrich.py --dry-run --limit 10   # preview the LLM's judgements
.venv/Scripts/python.exe enrich.py                        # re-derive spice/tags/cuisine/diet
.venv/Scripts/python.exe ingest.py --dry-run     # show what would be embedded
.venv/Scripts/python.exe ingest.py --limit 20    # smoke test
.venv/Scripts/python.exe ingest.py               # index everything

.venv/Scripts/python.exe query.py "something spicy and vegetarian to share"
.venv/Scripts/python.exe query.py "light starter" --k 10 --course Starter --max-spice 2
```

Vector ids are the Prisma `Item.id`, so re-running `ingest.py` updates rows in
place instead of duplicating them. Run it again after `bun run db:seed:items`.

## Layout

| File | Role |
| --- | --- |
| `menu_rag/config.py` | env-backed settings |
| `menu_rag/db.py` | reads `items` straight from Postgres (no Prisma Python client) |
| `menu_rag/documents.py` | row -> embedded sentence + Pinecone metadata |
| `menu_rag/embeddings.py` | `MistralAIEmbeddings` + dimension probe |
| `menu_rag/store.py` | index creation and `PineconeVectorStore` |
| `ingest.py` / `query.py` | entry points |

## Enrichment

The POS export never carried spice, taste or cuisine, and the heuristic classifier
could not recover them: `detectSpice` returns 0 before running a single regex for
anything outside the Food bucket, which zeroed 366 of 435 rows, and its one strong
signal (a literal `Spicy` tag) exists on 9 items. The result was
`spice 0:421, 1:7, 3:1, 4:6` -- so "spicy" and "not spicy" meant nothing.

`enrich.py` has Mistral read each dish and fill in spice, taste tags, cuisine,
course, protein and diet. Food-item spice is now spread across 0-4 instead of
collapsing onto 0.

`menu_rag/hardrules.py` runs after the model and overrides it wherever the source
text is decisive. This is not optional polish: left alone the model labelled
*Hot Garlic Fish* and *Fish Finger* as `Vegeterian`, and filed *Steam Rice* under
course `Alcohol`. A vegetarian recommended fish is a broken promise, not a ranking
miss, so a `Boozy`/`Lounge` POS tag and an explicit `Veg`/`Non Veg` marker in the
dish name both outrank the model.

`spice_confidence` records how sure the model was. `documents.py` omits the spice
sentence entirely below 0.5 rather than asserting "not spicy at all" about a dish
nobody ever assessed.

## What gets embedded

Enum columns are spelled out as prose so natural-language queries match, e.g.

> Butter Chicken. A main course from the Indian menu. Suitable for
> non-vegetarian diners. Main protein: chicken. Spice level 3 out of 5, medium
> spicy. Tastes creamy, rich. Serves 2 people. Allergens and tags: Dairy, Nuts.

Structured fields are also kept as metadata (`course`, `cuisine`, `diet`,
`protein`, `spice`, `taste_tags`, `serves_min`/`serves_max`) so retrieval can
combine semantic search with hard filters - see the flags on `query.py`.
