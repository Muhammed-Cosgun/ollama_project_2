
"""6. gün / Lab 1: Retrieve(k=8) -> lexical rerank(top_n=6)."""

from d5_05_search import search
from quality_utils import rerank_candidates


def retrieve_and_rerank(question, k=8, top_n=6):
    candidates = search(question, k=k)

    return rerank_candidates(
        question,
        candidates,
        top_n=top_n
    )


if __name__ == "__main__":
    q = "Kütüphane cumartesi kaçta kapanıyor?"

    ranked = retrieve_and_rerank(
        question=q,
        k=8,
        top_n=6
    )

    print("\n--- RERANKING SONUÇLARI ---")

    for i, (doc, score) in enumerate(ranked, 1):
        print(
            i,
            doc.metadata.get("source", "?"),
            "| lexical=",
            round(score, 3)
        )
        print("Metin:", doc.page_content[:220])

