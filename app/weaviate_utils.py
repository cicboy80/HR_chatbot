import time
import hashlib
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.classes.query import MetadataQuery, Filter

from app.llm_utils import embed_texts, embed_text, expand_query

COLLECTION = "PDFDocument"


def connect(weaviate_url: str, weaviate_api_key: str):
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_api_key),
        additional_config=AdditionalConfig(
            timeout=Timeout(
                init=30,
                query=180,
                insert=180,
            )
        ),
        skip_init_checks=False,
    )


def ensure_schema(client):
    """
    Create collection once, do NOT delete data.
    BYO vectors (we supply vectors explicitly at insert time).
    """
    if client.collections.exists(COLLECTION):
        print(f"ℹ️ Weaviate collection '{COLLECTION}' exists")
        return

    dims = len(embed_text("dimension check"))
    print(f"🧭 Embedding dims (sanity check only): {dims}")

    client.collections.create(
        name=COLLECTION,
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            )
        ),
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="document_name", data_type=DataType.TEXT),
            Property(name="content_hash", data_type=DataType.TEXT),
        ],
    )

    print(f"✅ Created '{COLLECTION}' (BYO vectors, cosine)")

def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash_exists(col, content_hash: str) -> bool:
    res = col.query.fetch_objects(
        filters=Filter.by_property("content_hash").equal(content_hash),
        limit=1,
        return_properties=["content_hash"],
    )
    return bool(res.objects)

def insert_chunks(
    client,
    chunks: list[str],
    document_name: str,
    batch_size: int = 12,
    max_retries: int = 3,
):
    if not chunks:
        raise ValueError("No chunks to insert into Weaviate")

    col = client.collections.get(COLLECTION)

    # 1) dedupe within the current upload first
    unique_chunks = []
    seen_hashes = set()

    for i, chunk in enumerate(chunks):
        content_hash = chunk_hash(chunk)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        unique_chunks.append((i, chunk, content_hash))

    # 2) dedupe against existing DB contents
    chunks_to_insert = []
    skipped_existing = 0

    for i, chunk, content_hash in unique_chunks:
        if content_hash_exists(col, content_hash):
            skipped_existing += 1
            continue
        chunks_to_insert.append((i, chunk, content_hash))

    if not chunks_to_insert:
        print("ℹ️ No new chunks to insert; all chunks already exist")
        return {
            "inserted": 0,
            "skipped_existing": skipped_existing,
            "unique_in_upload": len(unique_chunks),
        }

    total = 0

    for batch_start in range(0, len(chunks_to_insert), batch_size):
        batch = chunks_to_insert[batch_start: batch_start + batch_size]
        batch_texts = [chunk for _, chunk, _ in batch]

        try:
            vectors = embed_texts(batch_texts)
        except Exception as e:
            raise RuntimeError(f"Embedding batch failed (size={len(batch_texts)}): {e}")

        objects = []
        for (orig_idx, chunk, content_hash), vec in zip(batch, vectors):
            objects.append(
                DataObject(
                    properties={
                        "text": chunk,
                        "chunk_index": orig_idx,
                        "document_name": document_name,
                        "content_hash": content_hash,
                    },
                    vector=vec,
                )
            )

        for attempt in range(1, max_retries + 1):
            try:
                result = col.data.insert_many(objects)

                if hasattr(result, "errors") and result.errors:
                    raise RuntimeError(f"Weaviate insert errors: {result.errors}")

                total += len(objects)
                break

            except Exception as e:
                if attempt == max_retries:
                    raise
                backoff = 2 ** (attempt - 1)
                print(
                    f"⚠️ Insert batch failed "
                    f"(attempt {attempt}/{max_retries}): {e} — retrying in {backoff}s"
                )
                time.sleep(backoff)

    print(f"✅ Inserted {total} new chunks into Weaviate")
    print(f"ℹ️ Skipped {skipped_existing} chunks already present in Weaviate")

    return {
        "inserted": total,
        "skipped_existing": skipped_existing,
        "unique_in_upload": len(unique_chunks),
    }

def search_weaviate(client, query: str, k: int = 20):
    col = client.collections.get(COLLECTION)

    expanded_query = expand_query(query)
    query_vec = embed_text(expanded_query)

    res = col.query.hybrid(
        query=expanded_query,
        vector=query_vec,
        alpha=0.65,
        limit=k,
        return_properties=["text", "chunk_index"],
        return_metadata=MetadataQuery(score=True),
    )

    if not res.objects:
        return []

    return [
        {
            "text": o.properties["text"],
            "chunk_index": o.properties.get("chunk_index"),
            "score": o.metadata.score if o.metadata else None,
        }
        for o in res.objects
    ]