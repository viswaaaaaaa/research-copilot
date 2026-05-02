import streamlit as st
import sys
import os
import warnings

# ✅ Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports
from src.ingestion import load_and_chunk_pdf
from src.embeddings import create_vector_store, load_vector_store
from src.retriever import build_rag_chain, ask
from src.memory import get_memory_context, load_memory
from src.agent import smart_ask
from src.tools.web_search import web_search


# ─── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Research Co-pilot",
    page_icon="🔬",
    layout="wide"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; color: #3C3489; }
.sub-header { font-size: 0.95rem; color: #888; margin-bottom: 1rem; }
.source-box {
    background: #f8f9ff;
    border-left: 3px solid #3C3489;
    padding: 8px;
    border-radius: 4px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "ready" not in st.session_state:
    st.session_state.ready = False


# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 Research Co-pilot")

    # Upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        pdf_path = f"data/papers/{uploaded_file.name}"
        os.makedirs("data/papers", exist_ok=True)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ✅ Fix 1 — error handling on PDF processing
        if st.button("Process PDF"):
            with st.spinner("Processing..."):
                try:
                    create_vector_store(pdf_path)
                    st.session_state.llm, st.session_state.retriever = build_rag_chain()
                    st.session_state.ready = True
                    st.success(f"✅ Ready! {uploaded_file.name} processed.")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    # ✅ Fix 2 — error handling on load existing
    if st.button("Load Existing Papers"):
        try:
            st.session_state.llm, st.session_state.retriever = build_rag_chain()
            st.session_state.ready = True
            st.success("✅ Loaded!")
        except Exception as e:
            st.error("No papers found. Please upload one first.")

    # Memory display
    st.markdown("### 🧠 Memory")
    memory = load_memory()
    if memory["user_domain"]:
        st.write(", ".join(memory["user_domain"]))
    else:
        st.caption("No memory yet — start asking questions!")

    # Mode selection
    mode = st.radio(
        "Mode",
        ["🧠 Smart", "📄 Local", "🌐 Web"],
        index=1
    )

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ─── Header ───────────────────────────────────────────────
st.markdown("## 🔬 Research Co-pilot")
st.markdown("Ask questions about your papers · searches web when needed · remembers your domain")

if not st.session_state.ready:
    st.info("👈 Upload a PDF or load existing papers from the sidebar to get started")


# ─── Chat History ─────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.write(s)


# ─── Chat Input ───────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your research paper..."):

    if not st.session_state.ready:
        st.warning("⚠️ Please upload and process a PDF first!")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("⏳ Thinking..."):

            try:
                # 🧠 SMART MODE
                if "Smart" in mode:
                    try:
                        answer = smart_ask(prompt)
                        sources = ["Smart routing used (local + web + arxiv)"]
                    except Exception:
                        # Fallback to local if smart fails
                        answer = ask(prompt, st.session_state.llm, st.session_state.retriever)
                        sources = ["Local fallback (web tools unavailable)"]

                # 🌐 WEB MODE
                elif "Web" in mode:
                    try:
                        results = web_search(prompt)
                        if results["results"]:
                            answer = "\n\n".join([
                                f"**{r['title']}**\n{r['content'][:200]}"
                                for r in results["results"]
                            ])
                            sources = [r["url"] for r in results["results"]]
                        else:
                            answer = "No web results found. Try a different query."
                            sources = []
                    except Exception as e:
                        answer = f"Web search failed: {e}"
                        sources = []

                # 📄 LOCAL MODE
                else:
                    answer = ask(
                        prompt,
                        st.session_state.llm,
                        st.session_state.retriever
                    )
                    sources = ["Relevant sections from uploaded PDF"]

                st.markdown(answer)

                # Memory context display
                mem_ctx = get_memory_context()
                if mem_ctx:
                    with st.expander("🧠 Memory used"):
                        st.caption(mem_ctx)

                # Save response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")