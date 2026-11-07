from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, OLLAMA_URL

db = Chroma(
    collection_name="efatura_ornek",
    persist_directory="./chroma_efatura",
    embedding_function=OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL
    )
)

veriler = db.get()

print("Toplam kayıt:", len(veriler["ids"]))

for i, metin in enumerate(veriler["documents"]):
    print(f"\n--- Belge {i + 1} ---")
    print("ID:", veriler["ids"][i])
    print("Metin:", metin)
    print("Kaynak:", veriler["metadatas"][i])