from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, OLLAMA_URL

# Chroma veritabanını aç
db = Chroma(
    collection_name="efatura_ornek",
    persist_directory=str(Path(__file__).resolve().parent / "chroma_efatura"),
    embedding_function=OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )
)

# Kullanıcıdan soru al
soru = input("Sorunuzu yazın: ")

# En ilgili 2 belge parçasını bul
sonuclar = db.similarity_search(soru, k=1)

# Sonuçları göster
for i, belge in enumerate(sonuclar, 1):
    print(f"\n--- Sonuç {i} ---")
    print(belge.page_content)
    print("Kaynak:", belge.metadata.get("source"))