"""5. gün / Lab 1: TXT + kaynak + birim metadata."""
import json
from langchain_core.documents import Document
from config import ROOT, DATA_DIR

def load_documents():
    metadata = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        meta = {"source": path.name, **metadata[path.name]}
        docs.append(Document(page_content=content, metadata=meta))
    return docs

if __name__ == "__main__":
    docs = load_documents()
    print("Belge sayısı:", len(docs))
    for d in docs:
        print(d.metadata["source"], "|", d.metadata["birim"], "|", len(d.page_content), "karakter")