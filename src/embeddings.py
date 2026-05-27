from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever

from src.ingestion import load_and_chunk_pdf

import os

# Where ChromaDB will save data
CHROMA_PATH = "data/chroma_db"


def create_vector_store(pdf_path: str):
    """Embed chunks and store in ChromaDB"""

    # Step 1 — Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)

    # Step 2 — Load embedding model
    print("\n🧠 Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("✅ Embedding model loaded")

    # Step 3 — Store in ChromaDB
    print("\n📦 Storing chunks in ChromaDB...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ {len(chunks)} chunks stored in ChromaDB")

    return vector_store


def load_vector_store():
    """Load existing ChromaDB"""

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vector_store


def load_hybrid_retriever(pdf_path: str):
    """
    Hybrid Retrieval:
    BM25 + Dense Vector Search
    """

    print("\n⚡ Loading hybrid retriever...")

    # Load chunks for BM25
    chunks = load_and_chunk_pdf(pdf_path)

    # Load vector DB
    vector_store = load_vector_store()

    # VECTOR RETRIEVER
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    # BM25 RETRIEVER
    bm25_retriever = BM25Retriever.from_documents(chunks)

    bm25_retriever.k = 2

    print("✅ Hybrid retriever ready")

    return bm25_retriever, vector_retriever


def search(query: str, pdf_path: str):
    """
    Hybrid Search:
    BM25 + Vector Retrieval
    """

    print(f"\n🔍 Searching for: '{query}'")

    # Load retrievers
    bm25_retriever, vector_retriever = load_hybrid_retriever(
        pdf_path
    )

    # BM25 keyword retrieval
    bm25_results = bm25_retriever.invoke(query)

    # Dense vector retrieval
    vector_results = vector_retriever.invoke(query)

    # Combine results
    combined_results = bm25_results + vector_results

    # Remove duplicate chunks
    seen = set()

    results = []

    for doc in combined_results:

        content = doc.page_content

        if content not in seen:

            seen.add(content)

            results.append(doc)

    print(f"\n--- Top Results ---")

    for i, doc in enumerate(results):

        page = doc.metadata.get("page", "Unknown")

        print(f"\nResult {i+1} | Page: {page}")

        print(doc.page_content[:300])

        print("...")

    return results


if __name__ == "__main__":

    pdf_path = "data/papers/sample.pdf"

    # First run — create embeddings
    if not os.path.exists(CHROMA_PATH):

        print("🆕 First time — creating vector store...")

        create_vector_store(pdf_path)

    else:

        print("✅ Vector store already exists — loading...")

    # Test hybrid search
    search(
        "What is the main contribution of this paper?",
        pdf_path
    )
