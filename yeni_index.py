
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBED_MODEL, OLLAMA_URL


# 1. AYARLAR

DB_PATH = Path("./chroma_efatura")
DOSYA_PATH = Path("data/dokuman.txt")

COLLECTION_NAME = "efatura_ornek"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


# 2. DOKÜMANI OKU

metin = DOSYA_PATH.read_text(encoding="utf-8")

print("Embedding modeli:", EMBED_MODEL)
print("Doküman karakter sayısı:", len(metin))


# 3. METNİ PARÇALARA AYIR

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

parcalar = splitter.split_text(metin)

belgeler = [
    Document(
        page_content=parca,
        metadata={
            "source": "dokuman.txt",
            "chunk_index": i,
        },
    )
    for i, parca in enumerate(parcalar)
]

print("Oluşturulan parça sayısı:", len(belgeler))


# 4. EMBEDDING MODELİNİ HAZIRLA

embedding = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL,
)


# 5. ESKİ CHROMA İNDEKSİNİ TEMİZLE

# Yalnızca bu projeye ait örnek indeks silinir.
if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

print("Eski indeks temizlendi.")


# 6. YENİ BELGELERİ CHROMA'YA KAYDET

print("Embedding oluşturuluyor ve Chroma'ya kaydediliyor...")

db = Chroma.from_documents(
    documents=belgeler,
    embedding=embedding,
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_PATH),
)


# 7. SONUCU KONTROL ET

kayit_sayisi = db._collection.count()

print("\n=== CHROMA KAYIT SONUCU ===")
print("Kaydedilen parça sayısı:", len(belgeler))
print("Chroma toplam kayıt:", kayit_sayisi)

if kayit_sayisi == len(belgeler):
    print("SONUÇ: Kayıt başarılı.")
else:
    print("UYARI: Kayıt sayıları uyuşmuyor.")
