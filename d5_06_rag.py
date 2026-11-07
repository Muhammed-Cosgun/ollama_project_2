
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from config import EMBED_MODEL, OLLAMA_URL

# 1. Chroma veritabanına bağlan
embedding = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)

db = Chroma(
    collection_name="efatura_ornek",
    persist_directory=str(Path(__file__).resolve().parent / "chroma_efatura"),
    embedding_function=embedding
)

# 2. Dil modelini hazırla
llm = ChatOllama(
    model="qwen2.5:1.5b",
    base_url=OLLAMA_URL,
    temperature=0
)

# 3. Kullanıcıdan soru al
soru = input("Sorunuzu yazın: ")

# 4. Chroma'dan ilgili bilgiyi getir
sonuclar = db.similarity_search(soru, k=1)

baglam = "\n\n".join(
    belge.page_content for belge in sonuclar
)

# 5. Modele soru ve belgeyi gönder
prompt = f"""
Yalnızca aşağıdaki belgeye dayanarak soruyu cevapla.
Belgede cevap yoksa "Belgede bu bilgi bulunmuyor." de.
Bilgi uydurma.

BELGE:
{baglam}

SORU:
{soru}
"""

yanit = llm.invoke(prompt)

# 6. Sonucu göster
print("\n--- YAPAY ZEKA YANITI ---")
print(yanit.content)

print("\n--- KAYNAK ---")
for belge in sonuclar:
    print(belge.metadata.get("source"))