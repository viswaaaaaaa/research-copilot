from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

def web_search(query: str, max_results: int = 3) -> dict:
    """
    MCP Tool 1: Search the web for current information
    Used when ChromaDB doesn't have enough context
    """
    print(f"🌐 Web searching: '{query}'")
    
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results
    )
    
    # Format results cleanly
    results = []
    for r in response["results"]:
        results.append({
            "title": r["title"],
            "url": r["url"],
            "content": r["content"]
        })
        print(f"  ✅ Found: {r['title']}")
    
    return {
        "query": query,
        "results": results,
        "source": "web_search"
    }


if __name__ == "__main__":
    # Test it
    result = web_search("latest RAG techniques 2025")
    print("\n--- Results ---")
    for r in result["results"]:
        print(f"\n📰 {r['title']}")
        print(f"🔗 {r['url']}")
        print(f"📝 {r['content'][:200]}...")