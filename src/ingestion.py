from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re

def clean_text(text: str) -> str:
    """Remove noise patterns common in IEEE papers"""
    # Remove IEEE license lines
    text = re.sub(r'Authorized licensed use.*?Restrictions apply\.', '', text, flags=re.DOTALL)
    text = re.sub(r'Downloaded on.*?UTC from IEEE Xplore\.', '', text, flags=re.DOTALL)
    text = re.sub(r'1551-3203.*?more information\.', '', text, flags=re.DOTALL)
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def load_and_chunk_pdf(pdf_path: str):
    """Load a PDF and split it into chunks"""

    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    # Clean each page
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # Filter out pages that are mostly noise after cleaning
    filtered = []
    for doc in documents:
        text = doc.page_content.strip()

        # Skip nearly empty pages after cleaning
        if len(text) < 150:
            print(f"  ⏭ Skipping empty/noisy page {doc.metadata['page']}")
            continue

        # Skip last page if it looks like references only
        if doc.metadata["page"] >= len(documents) - 1:
            if text.count("[") > 8:
                print(f"  ⏭ Skipping reference page {doc.metadata['page']}")
                continue

        filtered.append(doc)

    print(f"✅ Kept {len(filtered)}/{len(documents)} pages after filtering")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_documents(filtered)

    # Final chunk-level cleaning
    clean_chunks = []
    for chunk in chunks:
        text = chunk.page_content.strip()

        # Skip chunks with too much noise
        if "authorized licensed" in text.lower():
            continue
        if "ieee xplore" in text.lower():
            continue
        if len(text) < 80:
            continue

        clean_chunks.append(chunk)

    print(f"✅ Split into {len(clean_chunks)} clean chunks")

    # Preview
    print("\n--- PREVIEW: First 3 chunks ---")
    for i, chunk in enumerate(clean_chunks[:3]):
        print(f"\nChunk {i+1} (page {chunk.metadata['page']}):")
        print(chunk.page_content[:200])
        print("...")

    return clean_chunks


if __name__ == "__main__":
    pdf_path = "data/papers/sample.pdf"

    if not os.path.exists(pdf_path):
        print("⚠️  No PDF found. Add any PDF to data/papers/ folder")
    else:
        chunks = load_and_chunk_pdf(pdf_path)
        print(f"\n🎉 Success! {len(chunks)} chunks ready for embedding")