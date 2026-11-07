"""6. gün / Lab 4: gerçek aramada recall@3 + rerank top1 ölçümü."""
from d5_05_search import search
from quality_utils import hit_at_k, rerank_candidates

CASES = [
    ("Kütüphane cumartesi kaçta kapanıyor?", "kutuphane.txt"),
    ("Kaybolan kulaklığımı nereden alabilirim?", "kayip_esya.txt"),
    ("Yeni kulüp kurmak için kaç imza gerekli?", "kulupler.txt"),
    ("Wi-Fi parolamı kiminle paylaşmalıyım?", "wifi.txt"),
    ("Staj belgelerini ne zaman yüklemeliyim?", "staj.txt"),
    ("Ring servisleri hafta içi kaç dakikada bir kalkar?", "ulasim.txt"),
    ("Yemekhane öğle yemeği hangi saatlerde?", "yemekhane.txt"),
    ("Bütünleme sınavına kimler katılabilir?", "sinavlar.txt"),
]

if __name__ == "__main__":
    hits = top1 = 0
    for query, expected in CASES:
        docs = search(query, k=8)
        hits += hit_at_k(docs[:3], expected)
        ranked = rerank_candidates(query, docs, top_n=3)
        found = ranked[0][0].metadata["source"] if ranked else None
        top1 += int(found == expected)
        print(("OK" if found == expected else "KONTROL"), "beklenen=", expected, "top1=", found)
    n = len(CASES)
    print("Hit@3 (soru başına beklenen kaynağın ilk 3'te olması):", f"{hits}/{n}")
    print("Rerank Top1 isabet:", f"{top1}/{n}")
    print("Not: Sonuçlar modele, chunk ayarına ve dil uyumuna göre değişebilir.")
