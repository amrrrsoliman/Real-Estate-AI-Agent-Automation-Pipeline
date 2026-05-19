"""
Build a local ChromaDB index from properties.csv using CPU embeddings (no API).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

BATCH_SIZE = 500
COLLECTION_NAME = "properties"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

ROOT = Path(__file__).resolve().parent
CSV_CANDIDATES = (
    ROOT / "properties.csv",
    ROOT / "data" / "raw" / "properties.csv",
)
CHROMA_PATH = ROOT / "database" / "real_estate_db"

COLUMNS = ("type", "title", "location", "bedroom", "bathroom", "size_sqm", "price")


def resolve_csv_path() -> Path:
    for path in CSV_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"properties.csv not found. Tried: {', '.join(str(p) for p in CSV_CANDIDATES)}"
    )


def build_text_for_embedding(row: pd.Series) -> str:
    parts = [
        f"Type: {row['type']}",
        f"Title: {row['title']}",
        f"Location: {row['location']}",
        f"Bedrooms: {row['bedroom']}",
        f"Bathrooms: {row['bathroom']}",
        f"Size sqm: {row['size_sqm']}",
        f"Price EGP: {row['price']}",
    ]
    return " | ".join(str(part) for part in parts)


def row_metadata(row: pd.Series) -> dict:
    meta = {col: str(row[col]).strip() if pd.notna(row[col]) else "" for col in COLUMNS}
    digits = "".join(c for c in meta.get("price", "") if c.isdigit())
    meta["price_egp"] = int(digits) if digits else 0
    return meta


def main() -> int:
    csv_path = resolve_csv_path()
    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: missing columns {missing}. Found: {list(df.columns)}")
        return 1

    df = df[list(COLUMNS)].fillna("")
    df = df[df.astype(str).apply(lambda r: "".join(r.values), axis=1).str.strip() != ""]
    df = df.reset_index(drop=True)
    total = len(df)
    print(f"Loaded {total} properties.")

    print(f"Loading embeddings model '{EMBEDDING_MODEL}' (CPU) ...")
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Dropping existing collection '{COLLECTION_NAME}' for a fresh index.")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Indexing into {CHROMA_PATH} (batches of {BATCH_SIZE}) ...\n")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        texts = [build_text_for_embedding(row) for _, row in batch.iterrows()]
        metadatas = [row_metadata(row) for _, row in batch.iterrows()]
        ids = [f"property_{i}" for i in batch.index]

        vectors = embedder.embed_documents(texts)
        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"[{end}/{total}] Indexed batch {start // BATCH_SIZE + 1} ({end - start} rows)")

    print(f"\nDone. {collection.count()} vectors in '{COLLECTION_NAME}' at {CHROMA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
