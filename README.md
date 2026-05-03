# 🔬 AI Research Co-pilot

> An intelligent research assistant that ingests papers, answers questions 
> with citations, searches the web when needed, and learns your research 
> domain over time.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![Ollama](https://img.shields.io/badge/Ollama-Llama3.2-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What it does

Upload any research paper (PDF) and have a conversation with it:

- **Cited answers** — every response shows which page it came from  
- **Smart routing** — low confidence → automatically searches web or Arxiv  
- **Persistent memory** — remembers your research domain across sessions  
- **Custom tools** — live web search + Arxiv paper fetcher (MCP-inspired)

---

## 🚀 Why this project matters

Traditional RAG systems fail when local context is insufficient.  
This system improves reliability by:

- Dynamically switching between local knowledge and live data  
- Reducing hallucinations through strict context grounding  
- Personalizing responses using persistent user memory  

This makes it closer to production-grade AI systems used in real-world applications.

---

## ⚡ Try it in 30 seconds

```bash
git clone https://github.com/viswaaaaaaa/research-copilot.git
cd research-copilot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
cp .env.example .env  # add your Tavily API key
streamlit run ui/app.py

###  Architecture
```markdown

#🏗 Architecture
User Question
     │
     ▼
Confidence Check (retrieval quality / similarity)
     │
     ├── High confidence → RAG Pipeline (local docs)
     │        ├── Embed query → ChromaDB search
     │        ├── Retrieve top chunks (MMR)
     │        ├── Inject memory context
     │        └── Llama 3.2 synthesizes cited answer
     │
     └── Low confidence → Live Tools
              ├── Web Search (Tavily)
              └── Arxiv Paper Fetcher
