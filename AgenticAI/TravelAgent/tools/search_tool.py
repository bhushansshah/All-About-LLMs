# tools/search_tool.py
"""
Search tools for the travel agent using LangChain's @tool decorator.
Updated to follow 2025 LangChain best practices.
"""
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def search_travel_info(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the web for travel-related information using SerpAPI.
    
    Use this tool to find information about destinations, activities,
    accommodations, restaurants, and attractions. The tool returns
    relevant web search results with titles, links, and snippets.
    
    Args:
        query: The search query (e.g., "best restaurants in Tokyo", "things to do in Paris")
        num_results: Maximum number of results to return (default: 5)
    
    Returns:
        A list of dictionaries containing search results with keys:
        - title: The title of the result
        - link: URL to the source
        - snippet: Brief description/excerpt
    """
    import os
    from utils.config import get_env
    
    # Try to import serpapi with fallback handling
    try:
        from serpapi import GoogleSearch
    except ImportError:
        try:
            from serpapi.google_search import GoogleSearch
        except ImportError:
            logger.error("SerpAPI package not installed. Install with: pip install google-search-results")
            return [{"error": "SerpAPI package not installed"}]
    
    api_key = get_env("SERPAPI_KEY")
    if not api_key:
        logger.error("SERPAPI_KEY not found in environment")
        return [{"error": "Search API key not configured"}]
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }
    
    try:
        search = GoogleSearch(params)
        resp = search.get_dict()
        organic = resp.get("organic_results", []) or resp.get("organic", [])
        
        results = []
        for r in organic[:num_results]:
            results.append({
                "title": r.get("title", ""),
                "link": r.get("link") or r.get("url", ""),
                "snippet": r.get("snippet", "") or r.get("snippet_highlighted_words", "") or "",
            })
        
        logger.info(f"Search completed for '{query}': {len(results)} results")
        return results
        
    except Exception as e:
        logger.exception(f"Search failed for query: {query}")
        return [{"error": f"Search failed: {str(e)}"}]


@tool
def generate_search_queries(destination: str, num_days: int, focus_areas: Optional[List[str]] = None) -> List[str]:
    """
    Generate optimized search queries for travel research.
    
    This tool creates targeted search queries to gather comprehensive
    information about a destination. Use this before searching to get
    better, more focused results.
    
    Args:
        destination: The travel destination (e.g., "Tokyo", "Paris, France")
        num_days: Number of days for the trip
        focus_areas: Optional list of specific interests (e.g., ["food", "history", "nightlife"])
    
    Returns:
        A list of 3-5 optimized search queries
    """
    queries = [
        f"best things to do in {destination}",
        f"top neighborhoods to visit in {destination}",
        f"where to stay in {destination} {num_days} days",
    ]
    
    if focus_areas:
        for area in focus_areas[:2]:  # Limit to 2 additional queries
            queries.append(f"best {area} in {destination}")
    else:
        queries.append(f"best restaurants in {destination}")
        queries.append(f"{destination} travel itinerary {num_days} days")
    
    return queries[:5]  # Maximum 5 queries
