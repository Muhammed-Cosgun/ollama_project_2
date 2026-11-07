"""5. gün / Lab 5: anlamsal arama ve metadata filtreleme."""
from langchain_chroma import Chroma
from config import DB_DIR, COLLECTION
from d5_03_embeddings import embedding_client

def get_store():
    if not DB_DIR.exists():
        raise FileNotFoundError("Önce python d5_04_index.py çalıştırın")
    return Chroma(collection_name=COLLECTION, embedding_function=embedding_client(), persist_directory=str(DB_DIR))

def search(question, k=4, unit=None):
    filter_value = {"birim": unit} if unit else None
    return get_store().similarity_search(query=question, k=k, filter=filter_value)

if __name__ == "__main__":
    question = "Kaybolan kulaklığımı nerede bulabilirim?"
    print("Soru:", question)
    for idx, doc in enumerate(search(question), 1):
        print(idx, doc.metadata["source"], "|", doc.metadata["chunk_id"])
    print("\nFILTRE / Ogrenci_Isleri")
    for idx, doc in enumerate(search(question, unit="Ogrenci_Isleri"), 1):
        print(idx, doc.metadata["source"], "|", doc.metadata["birim"])
