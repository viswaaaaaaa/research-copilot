from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.tools.web_search import web_search
from src.tools.arxiv_fetch import arxiv_search
from src.memory import get_memory_context, update_memory

CHROMA_PATH = "data/chroma_db"
CONFIDENCE_THRESHOLD = 1.2


def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    llm = OllamaLLM(model="llama3.2")

    return vector_store, retriever, llm


def check_confidence(vector_store, query):
    results = vector_store.similarity_search_with_score(query, k=1)

    if not results:
        return 999, []

    score = results[0][1]
    docs = [results[0][0]]

    return score, docs


def answer_from_docs(query, llm, docs):
    context = "\n\n".join([doc.page_content[:200] for doc in docs])

    memory_context = get_memory_context()

    prompt = f"""
{memory_context}

You are a research assistant.

Answer in 2-3 sentences only.
Use ONLY the context.
Be concise.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)


def answer_from_web(query, llm):
    paper_keywords = ["paper", "research", "arxiv"]

    if any(k in query.lower() for k in paper_keywords):
        results = arxiv_search(query)["results"]
    else:
        results = web_search(query)["results"]

    context = ""
    for r in results:
        context += f"{r.get('title','')}\n{r.get('content', r.get('summary',''))}\n\n"

    prompt = f"""
You are a research assistant.

Answer clearly using the information below.

{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)


# ✅ THIS IS WHAT UI NEEDS
def smart_ask(query: str):
    print(f"\n❓ {query}")

    vector_store, retriever, llm = load_components()

    score, docs = check_confidence(vector_store, query)

    print(f"Confidence score: {score}")

    if score < CONFIDENCE_THRESHOLD:
        print("Using local RAG")
        answer = answer_from_docs(query, llm, docs)
    else:
        print("Using web/arxiv")
        answer = answer_from_web(query, llm)

    update_memory(query, answer)

    return answer