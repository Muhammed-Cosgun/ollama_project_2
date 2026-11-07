from langchain_ollama import OllamaEmbeddings
from config import EMBED_MODEL, OLLAMA_URL
model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vektor = model.embed_query(input("Metin gir: "))
print("Boyut:", len(vektor), "\nİlk 5 değer:", vektor[:5])