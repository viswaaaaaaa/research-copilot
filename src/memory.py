import json
import os
from datetime import datetime

MEMORY_FILE = "data/memory.json"

def load_memory() -> dict:
    """Load memory from local JSON file"""
    if not os.path.exists(MEMORY_FILE):
        return {
            "user_domain": [],
            "topics_discussed": [],
            "sessions": []
        }
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory: dict):
    """Save memory to local JSON file"""
    os.makedirs("data", exist_ok=True)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def update_memory(question: str, answer: str):
    """
    After each Q&A — extract topic and save to memory
    So next session the agent knows your research domain
    """
    memory = load_memory()

    # Add this topic to memory
    memory["topics_discussed"].append({
        "question": question,
        "summary": answer[:150],  # save short summary
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    # Keep only last 10 topics (avoid memory file getting too big)
    memory["topics_discussed"] = memory["topics_discussed"][-10:]

    # Extract domain keywords from question
    keywords = extract_keywords(question)
    for kw in keywords:
        if kw not in memory["user_domain"]:
            memory["user_domain"].append(kw)

    # Keep only last 20 keywords
    memory["user_domain"] = memory["user_domain"][-20:]

    save_memory(memory)
    print(f"  💾 Memory updated — domain: {memory['user_domain'][-3:]}")


def extract_keywords(text: str) -> list:
    """Simple keyword extractor — no API needed"""
    # AI/ML domain keywords to watch for
    domain_words = [
        "privacy", "differential", "machine learning", "deep learning",
        "neural network", "transformer", "RAG", "retrieval", "embedding",
        "classification", "clustering", "federated", "security", "correlation",
        "feature selection", "nlp", "computer vision", "reinforcement"
    ]
    found = []
    text_lower = text.lower()
    for word in domain_words:
        if word in text_lower:
            found.append(word)
    return found


def get_memory_context() -> str:
    """
    Build a context string from memory to inject into prompts
    Tells the LLM what the user has been researching
    """
    memory = load_memory()

    if not memory["topics_discussed"] and not memory["user_domain"]:
        return ""  # No memory yet

    context = "USER RESEARCH CONTEXT (from past sessions):\n"

    if memory["user_domain"]:
        context += f"- Research domain: {', '.join(memory['user_domain'])}\n"

    if memory["topics_discussed"]:
        context += "- Recently asked about:\n"
        for topic in memory["topics_discussed"][-3:]:  # last 3 only
            context += f"  • {topic['question'][:80]}\n"

    return context


def show_memory():
    """Print current memory — useful for debugging"""
    memory = load_memory()
    print("\n🧠 Current Memory:")
    print(f"  Domain keywords: {memory['user_domain']}")
    print(f"  Topics discussed: {len(memory['topics_discussed'])}")
    if memory["topics_discussed"]:
        print("  Last 3 questions:")
        for t in memory["topics_discussed"][-3:]:
            print(f"    • [{t['timestamp']}] {t['question'][:60]}")


if __name__ == "__main__":
    # Test memory
    print("Testing memory system...")

    update_memory(
        "What is correlated differential privacy?",
        "Correlated differential privacy extends sensitivity to capture data correlation..."
    )

    update_memory(
        "What problem does data correlation cause?",
        "Data correlation causes more privacy leakage than expected..."
    )

    show_memory()

    print("\n📋 Memory context that gets injected into prompts:")
    print(get_memory_context())