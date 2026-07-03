"""Retrieval-quality eval for the HR chatbot.

Runs each question in qa_pairs.json against the live Weaviate index and
reports hit-rate: whether any expected phrase appears in the retrieved
chunks. Costs a few cents in OpenAI calls (query expansion + embeddings;
plus reranking with --rerank).

Usage:
    python evals/run_eval.py            # hit@20 and hit@4 on raw retrieval
    python evals/run_eval.py --rerank   # also hit@4 after LLM reranking
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv

load_dotenv(BASE_DIR / "api_keys.env")

import os

from app.weaviate_utils import connect, search_weaviate  # noqa: E402


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def phrase_in_docs(phrases: list[str], docs: list[dict]) -> bool:
    blob = normalize(" ".join(d.get("text", "") for d in docs))
    return any(normalize(p) in blob for p in phrases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerank", action="store_true", help="also score hit@4 after LLM reranking")
    parser.add_argument("--k", type=int, default=20, help="retrieval depth (default 20, matching the app)")
    args = parser.parse_args()

    qa_pairs = json.loads((BASE_DIR / "evals" / "qa_pairs.json").read_text())

    client = connect(os.environ["WEAVIATE_URL"], os.environ["WEAVIATE_API_KEY"])
    try:
        hits_k = hits_4 = hits_rerank = 0
        failures = []

        for pair in qa_pairs:
            question = pair["question"]
            phrases = pair["expected_phrases"]
            retrieved = search_weaviate(client, question, k=args.k)

            hit_k = phrase_in_docs(phrases, retrieved)
            hit_4 = phrase_in_docs(phrases, retrieved[:4])
            hits_k += hit_k
            hits_4 += hit_4

            line = f"  hit@{args.k}={'Y' if hit_k else 'n'}  hit@4={'Y' if hit_4 else 'n'}"

            if args.rerank:
                from app.llm_utils import rerank_chunks_with_llm

                reranked = rerank_chunks_with_llm(question, retrieved)[:4]
                hit_r = phrase_in_docs(phrases, reranked)
                hits_rerank += hit_r
                line += f"  rerank@4={'Y' if hit_r else 'n'}"

            print(f"{line}  {question}")
            if not hit_k:
                failures.append(question)

        n = len(qa_pairs)
        print(f"\nhit@{args.k}: {hits_k}/{n} ({hits_k / n:.0%})")
        print(f"hit@4 (raw order): {hits_4}/{n} ({hits_4 / n:.0%})")
        if args.rerank:
            print(f"hit@4 (after rerank): {hits_rerank}/{n} ({hits_rerank / n:.0%})")
        if failures:
            print("\nMisses at full depth:")
            for q in failures:
                print(f"  - {q}")

        # treat < 70% hit@k as failure so this can gate CI/manual checks
        return 0 if hits_k / n >= 0.7 else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
