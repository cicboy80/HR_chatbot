# Import relevant libraries and modules
from openai import OpenAI
import re

client = OpenAI()

# reate embeddings and Store in Vector DB
def embed_text(text: str) -> list[float]:
    """Create OpenAI embedding for ONE chunk"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Querying
def expand_query(query:str)->str:
    """Use GPT to expand a short query into a detailed query."""
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
            model = "gpt-4.1-mini",
            messages = [{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("⚠️ Query expansion failed:", e)
        return query
    
def rerank_chunks_with_llm(query:str, chunks:list[str]) -> list[str]:
    """
    Rerank retrieved chunks using GPT reasoning.
    Returns a list of chunks ordered in descending order
    """

    #Build a short reranking prompt
    chunk_list_parts = []
    for i, (text, _) in enumerate(chunks):
        clean_text = text[:400].strip().replace("\n", " ")
        chunk_list_parts.append(f"[{i+1}] {clean_text}...")
    chunk_list = "\n\n".join(chunk_list_parts)

    rerank_prompt = f"""
You are a precise HR assistant that ranks excerpts 
from a staff handbook by how relevant they are to the user's question.
You must rank excerpts that directly answer the user's question higher than those that merely discuss related topics.

Question: {query}

Excerpts:
{chunk_list}

Return only the list of excerpt numbers, separated by commas, in descending order of relevance. Example: 3, 1, 2
"""

    #Run LLM model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a factual and consistent reranker."},
            {"role": "user", "content": rerank_prompt}
        ],
        temperature = 0
    )
    
    text_output = response.choices[0].message.content.strip()
    print(f"🔎 Reranker raw output: {text_output}")  # optional

    # extract numbers safely
    order = [int(x) for x in re.findall(r'\d+', text_output )]
    order = [i for i in order if 1 <= i <= len(chunks)] #ensure valid range

    # fallback: if model fails to output indices, return original order
    if not order:
        order = list(range(1, len(chunks) + 1))
    
    # Return reordered text chunks
    ordered_chunks = [chunks[i-1][0] for i in order]
    return ordered_chunks