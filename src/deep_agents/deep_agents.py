from deepagents import create_deep_agent
import os
from dotenv import load_dotenv
from tavily import TavilyClient

from src.constants.constants import MODEL, SYSTEM_PROMPT
from src.rag.rag import rag_search

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTS_MD = os.path.normpath(os.path.join(_HERE, "../../AGENTS.md"))
_SKILLS_DIR = os.path.normpath(os.path.join(_HERE, "../skills/"))


def web_search(
    query: str,
    max_results: int = 5,
    include_raw_content: bool = False,
):
    """Search the web for the given query using Tavily to retrieve up-to-date information.

    Use this tool ONLY after rag_search has been called and the retrieved context
    is clearly insufficient, incomplete, or missing critical depth. Do not call this
    tool speculatively. Use the gap identified from rag_search results as the query.
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_client.search(
        query, max_results=max_results, include_raw_content=include_raw_content
    )

agent = create_deep_agent(
    model=MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=[web_search, rag_search],
    memory=[_AGENTS_MD],
    skills=[_SKILLS_DIR],
)
