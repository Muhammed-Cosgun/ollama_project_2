
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBED_MODEL, OLLAMA_URL


# 1. AYARLAR

DB_PATH = "./chroma_efatura"
COLLECTION_NAME = "efatura_ornek"

DOSYA_PATH = Path("data/kisiselBilgi.txt")
SOURCE_NAME = "kisiselBilgi.txt"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


# 2. BELGEYİ OKU

if not DOSYA_PATH.exists():
    raise FileNotFoundError(
        f"Dosya bulunamadı: {DOSYA_PATH}"
    )

metin = DOSYA_PATH.read_text(encoding="utf-8")

if not metin.strip():
    raise ValueError("Belge boş, kayıt yapılamaz.")

print("Belge okundu:", SOURCE_NAME)


# 3. METNİ PARÇALARA AYIR

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

parcalar = splitter.split_text(metin)

belgeler = [
    Document(
        page_content=parca,
        metadata={
            "source": SOURCE_NAME,
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


# 5. MEVCUT CHROMA VERİTABANINA BAĞLAN

db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
    persist_directory=DB_PATH,
)

onceki_kayit = db._collection.count()

print("Önceki toplam kayıt:", onceki_kayit)


# 6. AYNI BELGENİN ESKİ KAYITLARINI BUL

eski_kayitlar = db.get(
    where={"source": SOURCE_NAME}
)

eski_ids = eski_kayitlar["ids"]

print("Aynı belgeye ait eski parça:", len(eski_ids))


# 7. YENİ BELGE PARÇALARINI VEKTÖRLEŞTİR VE KAYDET

ids = [
    f"kisiselBilgi_chunk_{i}"
    for i in range(len(belgeler))
]

print("Embedding oluşturuluyor...")

# Önce embedding'leri oluşturuyoruz.
# Böylece model bağlantısında hata olursa
# mevcut kayıtlar silinmemiş olur.
metinler = [belge.page_content for belge in belgeler]

vektorler = embedding.embed_documents(metinler)

print("Oluşturulan embedding sayısı:", len(vektorler))

# Yeni vektörler hazırlandıktan sonra
# aynı belgeye ait eski kayıtları temizle.
if eski_ids:
    db.delete(ids=eski_ids)

# Hazır vektörleri ve metinleri Chroma'ya ekle.
db._collection.add(
    ids=ids,
    documents=metinler,
    embeddings=vektorler,
    metadatas=[belge.metadata for belge in belgeler],
)


# 8. SONUCU KONTROL ET

son_kayit = db._collection.count()

kaydedilen = db.get(
    where={"source": SOURCE_NAME}
)

print("\n=== CHROMA KAYIT SONUCU ===")
print("Önceki toplam kayıt:", onceki_kayit)
print("Kaydedilen yeni parça:", len(belgeler))
print("Belgeye ait mevcut kayıt:", len(kaydedilen["ids"]))
print("Chroma toplam kayıt:", son_kayit)

if len(kaydedilen["ids"]) == len(belgeler):
    print("SONUÇ: Kayıt başarılı.")
else:
    print("UYARI: Kayıt sayısı beklenen değerle uyuşmuyor.")
