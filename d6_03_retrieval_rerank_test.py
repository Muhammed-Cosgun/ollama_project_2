
"""
6. Gün / Lab 4
E-Fatura Retrieval Hit@3 ve Rerank Top1 Testi

Retrieval: k=8
Hit@3: İlk 3 sonuçta doğru bilgi var mı?
Rerank Top1: Yeniden sıralamada ilk sonuç doğru mu?
"""

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL, OLLAMA_URL
from quality_utils import rerank_candidates


DB_PATH = "./chroma_efatura"
COLLECTION_NAME = "efatura_ornek"

K = 8
TOP_N = 6


# Sorular yeni dokuman.txt içeriğinden hazırlandı.
# Beklenen ifadeler, doğru bilgiye sahip parçaları
# belirlemek için kullanılıyor.

CASES = [
    (
        "Fatura numarasında hangi karakterler kullanılabilir?",
        "büyük harf ve rakamlardan"
    ),
    (
        "TCKN kaç haneli olmalıdır?",
        "11 haneden oluşmalıdır"
    ),
    (
        "XML belgesinde zorunlu alan eksikse sistem ne yapar?",
        "tahmin ederek doldurmaz"
    ),
    (
        "Fatura düzenleme tarihi hangi biçimde olmalıdır?",
        "YYYY-MM-DD"
    ),
    (
        "Fatura düzenleme saati hangi biçimde olmalıdır?",
        "HH:MM:SS"
    ),
    (
        "XSD doğrulaması neyi kontrol eder?",
        "şema yapısına uygunluğunu"
    ),
    (
        "Schematron doğrulaması ne için kullanılır?",
        "alanlar arası koşulların"
    ),
    (
        "XML belgesinde açılış ve kapanış etiketleri nasıl olmalıdır?",
        "birbiriyle uyumlu olmalıdır"
    ),
    (
        "Düzeltmeden sonra XML belgesine ne yapılmalıdır?",
        "yeniden doğrulanmalıdır"
    ),
    (
        "Orijinal XML belgesi nasıl korunmalıdır?",
        "değiştirilmeden korunmalıdır"
    ),
]


def dogru_bilgi_var_mi(belge, beklenen):
    """Belge parçası beklenen bilgiyi içeriyor mu?"""

    return beklenen.casefold() in belge.page_content.casefold()


def hit_at_3(belgeler, beklenen):
    """İlk 3 parçada beklenen bilgi var mı?"""

    return int(
        any(
            dogru_bilgi_var_mi(belge, beklenen)
            for belge in belgeler[:3]
        )
    )


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

    print("\n=== E-FATURA RETRIEVAL + RERANK TESTİ ===")
    print("Veritabanındaki parça sayısı:", kayit_sayisi)
    print(f"Ayarlar: k={K}, rerank top_n={TOP_N}")
    print("Test soru sayısı:", len(CASES))

    if kayit_sayisi == 0:
        print("Veritabanı boş. Test yapılamıyor.")
        return

    hits = 0
    top1 = 0

    for i, (soru, beklenen) in enumerate(CASES, 1):

        # 1. Retrieval: Chroma'dan 8 aday getir.
        docs = db.similarity_search(soru, k=K)

        # 2. Hit@3: Doğru bilgi ilk 3'te mi?
        hit = hit_at_3(docs, beklenen)
        hits += hit

        # 3. Reranking: 8 adayı yeniden sırala.
        ranked = rerank_candidates(
            soru,
            docs,
            top_n=TOP_N
        )

        # 4. Rerank Top1: İlk sonuç doğru mu?
        ilk_belge = ranked[0][0] if ranked else None

        dogru_top1 = int(
            ilk_belge is not None
            and dogru_bilgi_var_mi(ilk_belge, beklenen)
        )

        top1 += dogru_top1

        print(f"\n{i}. Soru: {soru}")
        print("Hit@3:", "BAŞARILI" if hit else "BAŞARISIZ")
        print(
            "Rerank Top1:",
            "BAŞARILI" if dogru_top1 else "BAŞARISIZ"
        )

    n = len(CASES)

    print("\n=== GENEL SONUÇ ===")
    print(f"Hit@3: {hits}/{n} (%{hits / n * 100:.1f})")
    print(f"Rerank Top1: {top1}/{n} (%{top1 / n * 100:.1f})")

    print("\nNot: Testler örnek eğitim dokümanına dayanmaktadır.")


if __name__ == "__main__":
    main()
