from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMBED_MODEL, OLLAMA_URL

# 1. Metin dosyasını oku
metin = Path("data/dokuman.txt").read_text(encoding="utf-8")

# 2. Metni belgelere dönüştür ve parçalara ayır
belge = Document(page_content=metin, metadata={"source": "dokuman.txt"})

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
parcalar = splitter.split_documents([belge])

# 3. Ollama embedding modelini hazırla
embedding = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)

# 4. Chroma veritabanını oluştur
db = Chroma.from_documents(
    documents=parcalar,
    embedding=embedding,
    collection_name="efatura_ornek",
    persist_directory="./chroma_efatura"
)

# 5. Sonuçları göster
print("Kaydedilen parça sayısı:", len(parcalar))
print("Veritabanı klasörü: chroma_efatura")
print("Toplam kayıt sayısı:", len(db.get()["ids"]))