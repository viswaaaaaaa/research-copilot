from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

from sentence_transformers import CrossEncoder

from src.memory import get_memory_context, update_memory

import json
import os

from datetime import datetime

CHROMA_PATH = "data/chroma_db"

# -----------------------------------
# Cross-Encoder Reranker
# -----------------------------------

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# -----------------------------------
# Observability Logging
# -----------------------------------

def log_query(question, score, answer, pages):
    """
    Log every query for observability
    """

    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question": question,
        "best_rerank_score": round(float(score), 3),
        "pages_used": pages,
        "answer_length": len(answer),
        "answered": (
            "not enough information"
            not in answer.lower()
        )
    }

    os.makedirs("data", exist_ok=True)

    with open("data/query_log.jsonl", "a") as f:

        f.write(json.dumps(log) + "\n")


# -----------------------------------
# Guardrails
# -----------------------------------

def check_guardrails(question: str) -> tuple:
    """
    Basic prompt injection + validation
    """

    injection_patterns = [
        "ignore previous instructions",
        "forget everything",
        "you are now",
        "act as",
        "jailbreak",
        "system prompt",
        "override"
    ]

    for pattern in injection_patterns:

        if pattern in question.lower():

            return False, (
                "Potential prompt injection detected"
            )

    # Too short
    if len(question.strip()) < 5:

        return False, "Question too short"

    return True, "OK"


# -----------------------------------
# Build RAG System
# -----------------------------------

def build_rag_chain():

    print("🧠 Loading embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("📦 Loading vector store...")

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Broad retrieval for reranking
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20
        }
    )

    print("🤖 Loading Llama 3.2...")

    llm = OllamaLLM(
        model="llama3.2"
    )

    print("✅ System ready!\n")

    return llm, retriever


# -----------------------------------
# Reranking
# -----------------------------------

def rerank(query, docs, top_k=3):
    """
    Cross-encoder reranking
    """

    if not docs:
        return [], []

    # Query-document pairs
    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    # Predict scores
    scores = reranker.predict(pairs)

    # Sort descending
    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    top_scores = [
        score for score, _ in ranked[:top_k]
    ]

    top_docs = [
        doc for _, doc in ranked[:top_k]
    ]

    return top_docs, top_scores


# -----------------------------------
# Main Ask Function
# -----------------------------------

def ask(question, llm, retriever):

    # -----------------------------------
    # Guardrails
    # -----------------------------------

    is_safe, reason = check_guardrails(question)

    if not is_safe:

        print(f"🚫 Guardrail triggered: {reason}")

        return (
            f"I can't process this request: "
            f"{reason}"
        )

    print(f"\n❓ Question: {question}")

    print("⏳ Thinking...\n")

    # -----------------------------------
    # Retrieval
    # -----------------------------------

    docs = retriever.invoke(question)

    # Remove noisy chunks
    docs = [
        d for d in docs
        if "authorized licensed use"
        not in d.page_content.lower()
        and len(d.page_content.strip()) > 80
    ]

    if not docs:

        print("⚠️ No clean chunks found.")

        return (
            "I don't have enough information "
            "in the provided context."
        )

    # -----------------------------------
    # Reranking
    # -----------------------------------

    docs, scores = rerank(
        question,
        docs,
        top_k=3
    )

    # -----------------------------------
    # Deduplicate
    # -----------------------------------

    unique_docs = []

    unique_scores = []

    seen = set()

    for doc, score in zip(docs, scores):

        content = doc.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_docs.append(doc)

            unique_scores.append(score)

    docs = unique_docs

    scores = unique_scores

    # -----------------------------------
    # Confidence Bands
    # -----------------------------------

    best_score = max(scores)

    if best_score > 5:

        confidence = "high"

    elif best_score > 0:

        confidence = "medium"

    elif best_score > -10:

        confidence = "low"

    else:

        confidence = "very_low"

    print(
        f"📊 Best rerank score: "
        f"{best_score:.3f}"
    )

    print(
        f"📈 Confidence level: "
        f"{confidence}"
    )

    # Reject only extremely weak retrieval
    if confidence == "very_low":

        print(
            "⚠️ Retrieval confidence too low"
        )

        return (
            "I couldn't find sufficiently "
            "relevant information in the paper."
        )

    # -----------------------------------
    # Build Context
    # -----------------------------------

    context = "\n\n".join([
        f"[Page {doc.metadata['page']}]\n"
        f"{doc.page_content}"
        for doc in docs
    ])

    # -----------------------------------
    # Memory Context
    # -----------------------------------

    memory_context = get_memory_context()

    # -----------------------------------
    # Prompt
    # -----------------------------------

    prompt = f"""
You are a research paper assistant.

Answer ONLY using the provided paper content.

Rules:
- Be accurate and concise
- Do NOT invent information
- Do NOT use outside knowledge
- If uncertain, clearly say so
- Prefer direct factual answers

Current retrieval confidence:
{confidence}

{f"User research background: {memory_context}" if memory_context else ""}

Paper content:
{context}

Question:
{question}

Answer:
"""

    # -----------------------------------
    # LLM Generation
    # -----------------------------------

    response = llm.invoke(prompt)

    print("💬 Answer:\n")

    print(response)

    # -----------------------------------
    # Sources
    # -----------------------------------

    print("\n📎 Sources used:")

    for i, (doc, score) in enumerate(
        zip(docs, scores)
    ):

        print(
            f"  [{i+1}] "
            f"Score: {score:.3f} | "
            f"Page {doc.metadata['page']}"
        )

        print(
            f"      {doc.page_content[:100]}..."
        )

    # -----------------------------------
    # Observability Logging
    # -----------------------------------

    log_query(
        question=question,
        score=best_score,
        answer=response,
        pages=[
            doc.metadata["page"]
            for doc in docs
        ]
    )

    # -----------------------------------
    # Memory Update
    # -----------------------------------

    update_memory(question, response)

    return response


# -----------------------------------
# Main
# -----------------------------------

if __name__ == "__main__":

    llm, retriever = build_rag_chain()

    questions = [
        "Who are the authors of this paper?",
        "What is the main contribution of this paper?",
        "What problem does data correlation cause in differential privacy?",
        "Explain the CR-FS method proposed in the paper",
        "What datasets were used in experiments?",
        "What are the limitations of the proposed method?"
    ]

    for q in questions:

        print("\n" + "=" * 60)

        ask(q, llm, retriever)

        print()
