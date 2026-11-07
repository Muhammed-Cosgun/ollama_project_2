"""5. gün / Lab 3: Gerçek Ollama embedding isteği."""
from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, OLLAMA_URL

def embedding_client():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

if __name__ == "__main__":
    emb = embedding_client()
    vector = emb.embed_query("Kulaklığımı kaybettim, nereye başvurmalıyım?")
    print("Model:", EMBED_MODEL)
    print("Boyut:", len(vector))
    print("İlk 5 değer:", [round(x, 4) for x in vector[:5]])