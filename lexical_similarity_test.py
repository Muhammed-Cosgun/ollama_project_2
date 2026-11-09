"""Mevcut d5_05_search.py ve chroma_kampus ile lexical reranking deneyi."""
import re
from d5_05_search import search


def kelimelere_ayir(metin):
    """Türkçe I/İ harflerini koruyarak kelimelere ayır."""
    metin = metin.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[a-zçğıöşü0-9]+", metin)


def ucgramlar(metin):
    """Türkçe ek farklılıklarına daha toleranslı karakter üçlüleri."""
    gruplar = set()
    for kelime in kelimelere_ayir(metin):
        kelime = f"^{kelime}$"
        for i in range(len(kelime) - 2):
            gruplar.add(kelime[i:i + 3])
    return gruplar


def lexical_similarity(soru, metin):
    """Soru üçlülerinden belgede bulunanların oranı: [0,1]."""
    soru_gram = ucgramlar(soru)
    if not soru_gram:
        return 0.0
    belge_gram = ucgramlar(metin)
    return len(soru_gram & belge_gram) / len(soru_gram)


def ara_ve_sirala(soru, aday_sayisi=8, lexical_agirlik=0.4, sonuc_sayisi=3):
    if aday_sayisi < 1 or sonuc_sayisi < 1:
        raise ValueError("Aday ve sonuç sayısı pozitif olmalı")
    if not 0 <= lexical_agirlik <= 1:
        raise ValueError("Lexical ağırlık 0 ile 1 arasında olmalı")

    # Mevcut d5_05_search.py zaten chroma_kampus'a erişiyor.
    adaylar = search(soru, k=aday_sayisi)
    puanli = []
    for sira, belge in enumerate(adaylar, 1):
        # Mesafe değil: Chroma sırasından türetilmiş basit bir puan.
        anlamsal_sira_puani = 1 / sira
        lexical_puan = lexical_similarity(soru, belge.page_content)
        toplam = ((1 - lexical_agirlik) * anlamsal_sira_puani
                  + lexical_agirlik * lexical_puan)
        puanli.append((toplam, sira, lexical_puan, belge))

    puanli.sort(key=lambda item: (-item[0], item[1]))
    return puanli[:sonuc_sayisi]


def main():
    soru = input("Soru [Kaybolan kulaklığımı nereden bulabilirim?]: ").strip()
    soru = soru or "Kaybolan kulaklığımı nereden bulabilirim?"
    k = int(input("Chroma aday sayısı k [8]: ").strip() or "8")
    agirlik = float((input("Lexical ağırlık 0-1 [0.4]: ").strip() or "0.4").replace(",", "."))

    print("\nCHROMA'NIN İLK SIRALAMASI")
    adaylar = search(soru, k=k)
    for i, belge in enumerate(adaylar, 1):
        print(f"{i:>2}. {belge.metadata.get('source', '?')}")

    print("\nLEXICAL İLE YENİDEN SIRALAMA")
    for i, (toplam, eski_sira, lexical, belge) in enumerate(
        ara_ve_sirala(soru, aday_sayisi=k, lexical_agirlik=agirlik), 1
    ):
        print(f"{i}. {belge.metadata.get('source', '?')}"
              f" | Chroma sırası={eski_sira}"
              f" | lexical={lexical:.3f} | birleşik={toplam:.3f}")
        print("   Metin:", belge.page_content[:220].replace("\n", " "))


if __name__ == "__main__":
    main()
