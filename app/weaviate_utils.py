import time
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.config import Configure, DataType
from weaviate.classes.data import DataObject

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

    # Sanity check embedding dimensions (not enforced in schema)
    dims = len(embed_text("dimension check"))
    print(f"🧭 Embedding dims (sanity check only): {dims}")

    client.collections.create(
        name=COLLECTION,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric="cosine"
        ),
        properties=[
            {"name": "text", "data_type": DataType.TEXT},
            {"name": "page", "data_type": DataType.INT},
        ],
    )

    print(f"✅ Created '{COLLECTION}' (BYO vectors, cosine)")


def insert_chunks(
    client,
    chunks: list[str],
    batch_size: int = 12,
    max_retries: int = 3,
):
    if not chunks:
        raise ValueError("No chunks to insert into Weaviate")

    col = client.collections.get(COLLECTION)
    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        # 1) Batch embeddings
        try:
            vectors = embed_texts(batch)
        except Exception as e:
            raise RuntimeError(f"Embedding batch failed (size={len(batch)}): {e}")

        objects = []
        for j, (chunk, vec) in enumerate(zip(batch, vectors)):
            objects.append(
                DataObject(
                    properties={"text": chunk, "page": i + j},
                    vector=vec,
                )
            )

        # 2) Insert with retries
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

    print(f"✅ Inserted {total} chunks into Weaviate")


def search_weaviate(client, query: str, k: int = 12):
    col = client.collections.get(COLLECTION)

    expanded_query = expand_query(query)
    query_vec = embed_text(expanded_query)

    res = col.query.hybrid(
        query=expanded_query,
        vector=query_vec,
        alpha=0.3,
        limit=k,
        return_properties=["text", "page"],
    )

    if not res.objects:
        return []

    return [(o.properties["text"], o.metadata.distance) for o in res.objects]