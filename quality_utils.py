
"""Model veya LangChain gerektirmeyen değerlendirme ve sıralama yardımcıları."""

import re


STOP = {
    "ve", "ile", "için", "mi", "mı", "mu", "mü",
    "bir", "ne", "nerede", "nasıl", "hangi",
    "kadar", "kaç", "ben", "benim", "nedir",
    "bu", "de", "da", "olan", "olarak",
    "var", "mıydı", "mıyım", "olur", "yapılır"
}


def tokens(text):
    return {
        x
        for x in re.findall(
            r"[a-z0-9çğıöşü]+",
            text.lower()
        )
        if len(x) > 2 and x not in STOP
    }


def lexical_score(question, content):
    qs = tokens(question)

    return (
        len(qs & tokens(content)) / len(qs)
        if qs else 0.0
    )


def rerank_candidates(
    question,
    docs,
    top_n=6,
    lexical_weight=0.4
):
    candidates = []

    for rank, doc in enumerate(docs):
        lexical = lexical_score(
            question,
            doc.page_content
        )

        # Chroma sırasından yaklaşık puan oluştur
        semantic = 1 / (rank + 1)

        combined = (
            lexical_weight * lexical
            + (1 - lexical_weight) * semantic
        )

        candidates.append((doc, combined))

    return sorted(
        candidates,
        key=lambda pair: pair[1],
        reverse=True
    )[:top_n]


def citations_in_answer(answer):
    return set(
        re.findall(r"\[?(K\d+)\]?", answer)
    )


def validate_citations(answer, known_ids):
    got = citations_in_answer(answer)
    known = set(known_ids)

    return {
        "valid": bool(got) and got <= known,
        "used": sorted(got),
        "invalid": sorted(got - known),
        "missing": not bool(got)
    }


def hit_at_k(docs, expected_source):
    return int(
        any(
            doc.metadata.get("source") == expected_source
            for doc in docs
        )
    )
