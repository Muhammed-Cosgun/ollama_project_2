"""5. gün / Lab 2: chunking + overlap + metadata koruma."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from d5_01_documents import load_documents

def make_chunks(docs, chunk_size=50, chunk_overlap=10):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    counts = {}
    for doc in chunks:
        source = doc.metadata["source"]
        index = counts.get(source, 0)
        doc.metadata["chunk_id"] = f"{source}:{index:03d}"
        counts[source] = index + 1
    return chunks

if __name__ == "__main__":
    pieces = make_chunks(load_documents())
    print("Chunk sayısı:", len(pieces))
    for x in pieces[:5]:
        print(x.metadata["chunk_id"], "->", repr(x.page_content[:110]))
