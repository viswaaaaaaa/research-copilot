from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from src.memory import get_memory_context, update_memory

CHROMA_PATH = "data/chroma_db"


def build_rag_chain():
    print("🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("📦 Loading vector store...")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10
        }
    )

    print("🤖 Loading Llama 3.2...")
    llm = OllamaLLM(model="llama3.2")

    print("✅ System ready!\n")
    return llm, retriever


def ask(question, llm, retriever):
    print(f"\n❓ Question: {question}")
    print("⏳ Thinking...\n")

    # Retrieve docs
    docs = retriever.invoke(question)

    # ✅ Filter out license/watermark chunks
    docs = [
        d for d in docs
        if "authorized licensed use" not in d.page_content.lower()
        and len(d.page_content.strip()) > 80
    ]

    # Fallback if all filtered out
    if not docs:
        print("⚠️  No clean chunks found.")
        return "I don't have enough information in the provided context."

    # Limit to 2 chunks for M1 8GB RAM safety
    docs = docs[:2]

    # Build context — slightly more content per chunk
    context = "\n\n".join([
        f"[Page {doc.metadata['page']}]\n{doc.page_content[:250]}"
        for doc in docs
    ])

    # Get memory context
    memory_context = get_memory_context()

    # Clean focused prompt
    prompt = f"""You are a research paper assistant.
Answer ONLY using the paper content below. Be specific and direct.
Do NOT use any prior knowledge — only what is in the context.
If the answer is not in the context, say: "This specific information is not in the provided sections."
{f"User research background: {memory_context}" if memory_context else ""}

Paper content:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    print("💬 Answer:\n")
    print(response)

    print("\n📎 Sources used:")
    for i, doc in enumerate(docs):
        print(f"  [{i+1}] Page {doc.metadata['page']} — {doc.page_content[:80]}...")

    # Save to memory
    update_memory(question, response)

    return response


if __name__ == "__main__":
    llm, retriever = build_rag_chain()

    questions = [
        "Who are the authors of this paper?",
        "What is the main contribution of this paper?",
        "What problem does data correlation cause in differential privacy?",
        "Explain the CR-FS method proposed in the paper"
    ]

    for q in questions:
        print("\n" + "=" * 60)
        ask(q, llm, retriever)
        print()