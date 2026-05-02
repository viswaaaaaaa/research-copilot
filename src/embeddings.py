from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.ingestion import load_and_chunk_pdf
import os

# Where ChromaDB will save data on your Mac
CHROMA_PATH = "data/chroma_db"

def create_vector_store(pdf_path: str):
    """Embed chunks and store in ChromaDB"""

    # Step 1 — Load and chunk the PDF
    chunks = load_and_chunk_pdf(pdf_path)

    # Step 2 — Load embedding model (runs locally, no API needed)
    print("\n🧠 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"  # small, fast, free, local
    )
    print("✅ Embedding model loaded")

    # Step 3 — Embed chunks + store in ChromaDB
    print("\n📦 Storing chunks in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ {len(chunks)} chunks stored in ChromaDB")

    return vector_store

def load_vector_store():
    """Load existing ChromaDB (so you don't re-embed every time)"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vector_store

def search(query: str, k: int = 3):
    """Search ChromaDB for most relevant chunks"""

    print(f"\n🔍 Searching for: '{query}'")
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    print(f"\n--- Top {k} results ---")
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} | Score: {round(score, 3)} | Page: {doc.metadata['page']}")
        print(doc.page_content[:300])
        print("...")

    return results

if __name__ == "__main__":
    pdf_path = "data/papers/sample.pdf"

    # First run — embed and store
    if not os.path.exists(CHROMA_PATH):
        print("🆕 First time — creating vector store...")
        create_vector_store(pdf_path)
    else:
        print("✅ Vector store already exists — loading...")

    # Test a search query
    search("What is the main contribution of this paper?")