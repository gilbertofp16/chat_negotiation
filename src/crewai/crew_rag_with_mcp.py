import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from crewai import LLM, Agent, Crew, Task
from src.mcp_servers.mcp_client import mcp_search_sync, web_context_from_results
from src.retriever.get_retriever import (
    get_chroma_retriever,  # CrewAI’s LLM wrapper (LiteLLM underneath)
)

# --------------------------------------------------------------------------------------
# Config / setup
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()


# --------------------------------------------------------------------------------------
# LLM & Embeddings
# --------------------------------------------------------------------------------------
def make_llm() -> LLM:
    """CrewAI LLM wrapper using Gemini via LiteLLM."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")
    return LLM(model="gemini/gemini-1.5-pro-latest", api_key=api_key, temperature=0.2)


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004")


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
def create_agent(role: str, goal: str, backstory: str, llm: LLM, verbose: bool = True) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )


# --------------------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------------------
def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        page_number = doc.metadata.get("page", "N/A")
        parts.append(f"--- Document {i+1} (Page: {page_number}) ---\n{doc.page_content}\n")
    return "".join(parts).strip()


def retrieve_sync(query: str, retriever: VectorStoreRetriever) -> str:
    docs = retriever.invoke(query)
    return format_docs(docs) if docs else ""


def _truncate(s: str, max_chars: int = 12000) -> str:
    if s and len(s) > max_chars:
        return s[: max_chars - 1000] + "\n\n[... truncated ...]\n" + s[-1000:]
    return s


async def crew_answer_with_pdf(question: str) -> Dict[str, Any]:
    llm = make_llm()
    coach = create_agent(
        role="Master Negotiation Coach",
        goal="Answer questions only with supporting evidence from provided context.",
        backstory="You are a negotiation coach who cross-checks retrieved book context with reputable web sources.",
        llm=llm,
    )

    # Load your persisted Chroma (not in-memory here)
    retriever = get_chroma_retriever(k=8)

    # 1) Chroma context
    doc_context = retrieve_sync(question, retriever)
    if not doc_context:
        doc_context = "[No results found from Chroma.]"
    doc_context = _truncate(doc_context)

    # 2) MCP web search context focused on Black Swan Group
    query = f'{question} site:blackswanltd.com OR "Black Swan Group" negotiation'
    web_items = mcp_search_sync(query, max_results=6)
    web_context = web_context_from_results(web_items) or "[No web results.]"
    web_context = _truncate(web_context)

    task = Task(
        description=(
            "Using ONLY the information below, answer the user's question and explicitly state "
            "whether the web context aligns with the Chroma context. If there are conflicts, list them.\n\n"
            f"Question: {question}\n\n"
            f"Chroma context:\n{doc_context}\n\n"
            f"Web context:\n{web_context}\n"
        ),
        expected_output=(
            "A concise answer with (1) citations to page numbers from Chroma text when present, "
            "(2) a short alignment report: agrees / partially agrees / conflicts, with bullet points."
        ),
        agent=coach,
    )

    crew = Crew(agents=[coach], tasks=[task], verbose=True)

    try:
        result = crew.kickoff()
        return {"text": str(result)}
    except Exception as e:
        logging.error(f"Crew execution error: {e}")
        return {"text": f"Error: {e}"}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- CrewAI PDF RAG (ephemeral) ---")
    question = "What is BATNA?"
    pdf = "/home/gilberto/code/chat_negotiation/data/source/negotiation_book.pdf"  # <- change to your PDF

    try:
        out = asyncio.run(crew_answer_with_pdf(question, pdf))
        print("\n--- Answer ---")
        print(out.get("text", "No answer"))
        print("----------------")
    except Exception as e:
        logging.error(f"Execution error: {e}")

    print("\n--- Done ---")
