
import re

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, OLLAMA_URL


# Mevcut e-Fatura veritabanımız
DB_PATH = "./chroma_efatura"
COLLECTION_NAME = "efatura_ornek"


def kelimelere_ayir(metin):
    """Türkçe karakterleri koruyarak kelimelere ayır."""
    metin = metin.replace("İ", "i").replace("I", "ı").lower()

    return re.findall(
        r"[a-zçğıöşü0-9]+",
        metin
    )


def ucgramlar(metin):
    """Kelimelerden üç karakterlik gruplar oluştur."""
    gruplar = set()

    for kelime in kelimelere_ayir(metin):
        kelime = f"^{kelime}$"

        for i in range(len(kelime) - 2):
            gruplar.add(kelime[i:i + 3])

    return gruplar


def lexical_similarity(soru, metin):
    """Soru ile belge arasındaki karakter benzerliğini hesapla."""
    soru_gram = ucgramlar(soru)

    if not soru_gram:
        return 0.0

    belge_gram = ucgramlar(metin)

    return len(soru_gram & belge_gram) / len(soru_gram)


def ara_ve_sirala(
    db,
    soru,
    aday_sayisi=8,
    lexical_agirlik=0.4,
    sonuc_sayisi=6
):
    if aday_sayisi < 1 or sonuc_sayisi < 1:
        raise ValueError(
            "Aday ve sonuç sayısı pozitif olmalı."
        )

    if not 0 <= lexical_agirlik <= 1:
        raise ValueError(
            "Lexical ağırlık 0 ile 1 arasında olmalı."
        )

    # 1. RETRIEVAL
    adaylar = db.similarity_search(
        soru,
        k=aday_sayisi
    )

    print("\n--- CHROMA İLK SIRALAMA ---")

    for i, belge in enumerate(adaylar, 1):
        print(f"\n{i}. Belge")
        print("Metin:", belge.page_content.strip())

    # 2. RERANKING
    puanli = []

    for sira, belge in enumerate(adaylar, 1):

        anlamsal_sira_puani = 1 / sira

        lexical_puan = lexical_similarity(
            soru,
            belge.page_content
        )

        toplam_puan = (
            (1 - lexical_agirlik) * anlamsal_sira_puani
            + lexical_agirlik * lexical_puan
        )

        puanli.append(
            (toplam_puan, sira, lexical_puan, belge)
        )

    puanli.sort(
        key=lambda x: (-x[0], x[1])
    )

    return puanli[:sonuc_sayisi]


def main():

    print("=== E-FATURA LEXICAL RERANKING TESTİ ===")
    print("Ayarlar: k=8, top_n=6")

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
        print(
            "Veritabanı boş. Önce yeni_index.py çalıştır."
        )
        return

    print(
        "Veritabanındaki parça sayısı:",
        kayit_sayisi
    )

    soru = input(
        "\nSoru [TCKN kaç haneli olmalıdır?]: "
    ).strip()

    soru = soru or "TCKN kaç haneli olmalıdır?"

    k = int(
        input(
            "Chroma aday sayısı k [8]: "
        ).strip() or "8"
    )

    agirlik = float(
        (
            input(
                "Lexical ağırlık 0-1 [0.4]: "
            ).strip() or "0.4"
        ).replace(",", ".")
    )

    sonuc = ara_ve_sirala(
        db,
        soru,
        aday_sayisi=k,
        lexical_agirlik=agirlik,
        sonuc_sayisi=6
    )

    print("\n--- LEXICAL RERANKING SONUCU ---")

    for yeni_sira, (
        toplam,
        eski_sira,
        lexical,
        belge
    ) in enumerate(sonuc, 1):

        print(f"\n{yeni_sira}. Belge")
        print("Eski Chroma sırası:", eski_sira)
        print("Lexical puan:", round(lexical, 3))
        print("Birleşik puan:", round(toplam, 3))
        print("Metin:", belge.page_content.strip())


if __name__ == "__main__":
    main()
