# Import relevant libraries and modules
from openai import OpenAI
import re
from typing import List, Dict, Any

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
QUERY_EXPAND_MODEL = "gpt-4.1-mini"
RERANK_MODEL = "gpt-4o-mini"


# --- Embeddings ---

def embed_text(text: str) -> list[float]:
    """Create OpenAI embedding for ONE chunk."""
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Create OpenAI embeddings for MANY chunks in a single request (faster + more reliable).
    Returns a list of vectors aligned with `texts`.
    """
    if not texts:
        return []
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [d.embedding for d in response.data]


# --- Query expansion ---

def expand_query(query: str) -> str:
    """Use GPT to expand a short query into a more detailed search query."""
    try:
        prompt = f"""Expand the following short questions into a more detailed search query
that includes synonyms and related HR terms, but also restate the keywords clearly.

Examples:

Q: Who should I contact if I am sick?
Expanded: Who should I notify or contact if I am ill, unwell, or absent due to sickness — such as my Deputy Head or line manager.

Q: What do I do if I am late?
Expanded: What procedure should I follow if I expect to be late, delayed, or absent for work — who must I contact, for example my Deputy Head or line manager?

Now expand this query in the same way:
Q: {query}
Expanded:
"""
        response = client.chat.completions.create(
            model=QUERY_EXPAND_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Query expansion failed:", e)
        return query


# --- Reranking ---

def rerank_chunks_with_llm(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank retrieved chunks using GPT reasoning.
    `chunks` is a list of dicts:
    [{"text": "...", "chunk_index": 1, "distance": 0.12}, ...]
    Returns the same dicts ordered by relevance.
    """
    if not chunks:
        return []

    # Keep prompt small & consistent
    chunk_list_parts = []
    for i, chunk in enumerate(chunks):
        clean_text = chunk.get("text", "")[:400].strip().replace("\n", " ")
        chunk_index = chunk.get("chunk_index")
        distance = chunk.get("distance")
        chunk_list_parts.append(
            f"[{i+1}] (chunk={chunk_index}, distance={distance}) {clean_text}..."
        )
    chunk_list = "\n\n".join(chunk_list_parts)

    rerank_prompt = f"""
You are a precise HR assistant that ranks excerpts from a staff handbook
by how relevant they are to the user's question.

Question: {query}

Excerpts:
{chunk_list}

Return ONLY the list of excerpt numbers, separated by commas, in descending order of relevance.
Example: 3, 1, 2
""".strip()

    try:
        response = client.chat.completions.create(
            model=RERANK_MODEL,
            messages=[
                {"role": "system", "content": "You are a factual and consistent reranker."},
                {"role": "user", "content": rerank_prompt}
            ],
            temperature=0
        )
        text_output = response.choices[0].message.content.strip()
        print(f"🔎 Reranker raw output: {text_output}")

        # Extract numbers safely
        order = [int(x) for x in re.findall(r"\d+", text_output)]
        order = [i for i in order if 1 <= i <= len(chunks)]

        if not order:
            return chunks

        reranked = [chunks[i-1] for i in order]
        used = set(order)
        remaining = [chunks[i] for i in range(len(chunks)) if (i + 1) not in used]

        return reranked + remaining

    except Exception as e:
        print("⚠️ Rerank failed:", e)
        # Fallback: return original order
        return chunks
