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

```bash
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

## What gets embedded

Enum columns are spelled out as prose so natural-language queries match, e.g.

> Butter Chicken. A main course from the Indian menu. Suitable for
> non-vegetarian diners. Main protein: chicken. Spice level 3 out of 5, medium
> spicy. Tastes creamy, rich. Serves 2 people. Allergens and tags: Dairy, Nuts.

Structured fields are also kept as metadata (`course`, `cuisine`, `diet`,
`protein`, `spice`, `taste_tags`, `serves_min`/`serves_max`) so retrieval can
combine semantic search with hard filters - see the flags on `query.py`.
