"""6. gün / Lab 1: Retrieve(k=8) -> lexical rerank(top_n=3)."""
from d5_05_search import search
from quality_utils import rerank_candidates

def retrieve_and_rerank(question, k=8, top_n=3):
    candidates = search(question, k=k)
    return rerank_candidates(question, candidates, top_n=top_n)

if __name__ == "__main__":
    q = "Kütüphane cumartesi kaçta kapanıyor?"
    ranked = retrieve_and_rerank(q)
    for i, (doc, score) in enumerate(ranked, 1):
        print(i, doc.metadata["source"], "| lexical=", round(score, 3))
