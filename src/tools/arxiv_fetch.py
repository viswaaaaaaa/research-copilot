import arxiv
from dotenv import load_dotenv

load_dotenv()


def arxiv_search(query: str, max_results: int = 3) -> dict:
    """
    MCP Tool 2: Search and fetch papers from Arxiv
    Used when user asks about recent research papers
    """
    print(f"📚 Arxiv searching: '{query}'")

    try:
        client = arxiv.Client()

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []

        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [str(a) for a in paper.authors[:3]],
                "summary": paper.summary[:500],
                "url": paper.entry_id,
                "published": str(paper.published.date())
            })
            print(f"  ✅ Found: {paper.title[:60]}...")

        # ✅ Handle empty results safely
        if not results:
            print("  ⚠️ No Arxiv results found — falling back to web search")
            from src.tools.web_search import web_search
            fallback = web_search(query)
            fallback["source"] = "web_fallback"
            return fallback

        return {
            "query": query,
            "results": results,
            "source": "arxiv"
        }

    except Exception as e:
        print(f"  ⚠️ Arxiv unavailable: {e} — falling back to web search")

        # ✅ Fallback to web search
        from src.tools.web_search import web_search
        fallback = web_search(query)
        fallback["source"] = "web_fallback"

        return fallback


if __name__ == "__main__":
    # Test it
    result = arxiv_search("retrieval augmented generation 2024")

    print("\n--- Results ---")

    for r in result["results"]:
        print(f"\n📄 {r.get('title', 'No title')}")
        print(f"👥 {', '.join(r.get('authors', []))}")
        print(f"📅 {r.get('published', 'N/A')}")
        print(f"📝 {r.get('summary', r.get('content', ''))[:200]}...")