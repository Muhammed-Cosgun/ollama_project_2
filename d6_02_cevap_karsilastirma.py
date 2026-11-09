
"""
Normal RAG ve Reranking'li RAG karşılaştırması.

Durum 1: Chroma (k=8) -> İlk 6 belge -> Qwen
Durum 2: Chroma (k=8) -> Lexical Reranking -> İlk 6 belge -> Qwen

Amaç: Reranking sonrası cevap değişiyor mu?
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import EMBED_MODEL, OLLAMA_URL
from lexical_similarity_test_2 import lexical_similarity


DB_PATH = "./chroma_efatura"
COLLECTION_NAME = "efatura_ornek"

K = 8
TOP_N = 6
LEXICAL_AGIRLIK = 0.4
LLM_MODEL = "qwen2.5:1.5b"


def cevap_uret(model, soru, belgeler):
    """Belgeleri kullanarak Qwen'den cevap üret."""

    baglam = "\n\n".join(
        belge.page_content for belge in belgeler
    )

    prompt = f"""
Sen e-Fatura teknik doğrulama asistanısın.

Yalnızca verilen belgeleri kullanarak cevap ver.
Cevap belgelerde yoksa "Belgede bu bilgi bulunmuyor." de.
Bilgi uydurma. Kısa ve açık cevap ver.

BELGELER:
{baglam}

SORU:
{soru}

CEVAP:
"""

    return model.invoke(prompt).content.strip()


def rerank_yap(soru, adaylar):
    """Lexical similarity ile belgeleri yeniden sırala."""

    puanli = []

    for sira, belge in enumerate(adaylar, 1):

        chroma_puani = 1 / sira
        lexical_puan = lexical_similarity(
            soru, belge.page_content
        )

        toplam_puan = (
            (1 - LEXICAL_AGIRLIK) * chroma_puani
            + LEXICAL_AGIRLIK * lexical_puan
        )

        puanli.append((toplam_puan, sira, belge))

    puanli.sort(key=lambda x: (-x[0], x[1]))

    return puanli[:TOP_N]


def main():

    embedding = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )

    db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=DB_PATH
    )

    kayit_sayisi = db._collection.count()

    if kayit_sayisi == 0:
        print("Veritabanı boş.")
        return

    model = ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0
    )

    soru = input(
        "Soru: "
    ).strip() or "TCKN kaç haneli olmalıdır?"

    # Her iki yöntem için aynı adaylar
    adaylar = db.similarity_search(soru, k=K)

    # Durum 1: Normal RAG
    normal_belgeler = adaylar[:TOP_N]

    # Durum 2: Reranking'li RAG
    rerank_sonuclari = rerank_yap(soru, adaylar)

    rerank_belgeler = [
        belge for _, _, belge in rerank_sonuclari
    ]

    # Aynı model ile cevap üret
    normal_cevap = cevap_uret(
        model, soru, normal_belgeler
    )

    rerank_cevap = cevap_uret(
        model, soru, rerank_belgeler
    )

    # Belge sıralarını belirle
    normal_siralar = list(
        range(1, len(normal_belgeler) + 1)
    )

    rerank_siralar = [
        eski_sira
        for _, eski_sira, _ in rerank_sonuclari
    ]

    # Kısa karşılaştırma raporu
    print("\n=== RAG KARŞILAŞTIRMA TESTİ ===")
    print(f"Soru: {soru}")
    print(f"Ayarlar: k={K}, top_n={TOP_N}, lexical={LEXICAL_AGIRLIK}")
    print(f"Veritabanındaki parça sayısı: {kayit_sayisi}")

    print("\n1. NORMAL RAG (Reranking yok)")
    print("Belge sıraları:", normal_siralar)
    print("Cevap:", normal_cevap)

    print("\n2. RERANKING'Lİ RAG")
    print("Belge sıraları:", rerank_siralar)
    print("Cevap:", rerank_cevap)

    print("\n=== SONUÇ ===")
    print(
        "Belge sıralaması değişti mi?",
        "EVET" if normal_siralar != rerank_siralar else "HAYIR"
    )
    print(
        "Cevap metni değişti mi?",
        "EVET" if normal_cevap != rerank_cevap else "HAYIR"
    )


if __name__ == "__main__":
    main()
