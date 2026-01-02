import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import DataType
from weaviate.classes.data import DataObject
from app.llm_utils import embed_text, expand_query

# CONNECT

def connect(weaviate_url: str, WEAVIATE_API_KEY: str):
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        skip_init_checks=True
    )

# SCHEMA

def setup_schema(client):
    print("🧹 Resetting Weaviate schema...")

    client.collections.delete_all()

    client.collections.create(
        name="PDFDocument",
        vectorizer_config=None,
        vector_index_config={
            "distance": "cosine",
            "dimensions": 1536
        },  # ✅ COMMA FIXED HERE
        properties=[
            {"name": "text", "data_type": DataType.TEXT},
            {"name": "page", "data_type": DataType.INT}
        ]
    )

    print("✅ Fresh PDFDocument collection created with 1536 dimensions")

# SAFE BATCH INSERT

def insert_chunks(client, chunks, batch_size=64):
    pdf_chunks = client.collections.get("PDFDocument")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        objects = []
        for j, chunk in enumerate(batch):
            vec = embed_text(chunk)

            obj = DataObject(
                properties={
                    "text": chunk,
                    "page": i + j
                },
                vector=vec
            )

            objects.append(obj)

        pdf_chunks.data.insert_many(objects)

    print("✅ PDF successfully stored as vectors")

# SAFE HYBRID SEARCH

def search_weaviate(client, query: str, k=12):
    pdf_chunks = client.collections.get("PDFDocument")
    print("VECTOR COUNT:", pdf_chunks.aggregate.over_all().total_count)

    expanded_query = expand_query(query)
    query_vec = embed_text(expanded_query)

    result = pdf_chunks.query.hybrid(
        query=expanded_query,
        vector=query_vec,
        alpha=0.3,
        limit=k,
        return_properties=["text", "page"]
    )

    return [(o.properties["text"], o.metadata.distance) for o in result.objects]