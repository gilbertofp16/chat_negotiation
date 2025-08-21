import asyncio
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langfuse.callback import CallbackHandler

from crewai import LLM, Agent, Crew, Task
from src.config import REASONING_MODEL

# Import the underlying client functions, not the LangChain tool wrapper
from src.mcp_servers.mcp_client import bsw_context_from_results, mcp_browse_sync
from src.retriever.get_retriever import get_chroma_retriever

# --------------------------------------------------------------------------------------
# Config / setup
# --------------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()


# --------------------------------------------------------------------------------------
# CrewAI Tool Definition
# --------------------------------------------------------------------------------------
@tool("Browse BSW Tool")
def browse_bsw(topic: str) -> str:
    """
    Searches the web for information on Black Swan negotiation techniques to sanity-check or enrich an answer.
    Input should be a specific negotiation topic (e.g., 'mirroring', 'calibrated questions').
    """
    return bsw_context_from_results(mcp_browse_sync(topic))


# --------------------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------------------
def make_llm() -> LLM:
    """CrewAI LLM wrapper using Gemini via LiteLLM."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")

    return LLM(
        model=f"gemini/{REASONING_MODEL}",
        api_key=api_key,
        temperature=0.2,
    )


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
def create_agent(role: str, goal: str, backstory: str, llm: LLM, tools: list = None, verbose: bool = True) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
        tools=tools or [],
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


# --------------------------------------------------------------------------------------
# Crew run
# --------------------------------------------------------------------------------------
async def crew_answer(question: str) -> Dict[str, Any]:
    """
    Answers a question using a CrewAI crew with an agent that can use the BSW tool.
    """
    llm = make_llm()

    coach = create_agent(
        role="Master Negotiation Coach",
        goal=(
            "Your primary goal is to answer questions based on the provided book context. "
            "You can use the `Browse BSW Tool` to sanity-check or enrich your answers about specific "
            "Black Swan negotiation techniques, but the book is your main source of truth."
        ),
        backstory="You are a world-class negotiation coach who trusts the provided book but verifies with targeted web searches.",
        llm=llm,
        tools=[browse_bsw],  # Pass the CrewAI-native tool
    )

    # Retrieve the book context first.
    retriever = get_chroma_retriever(k=8)
    doc_context = retrieve_sync(question, retriever)
    if not doc_context:
        doc_context = "[No results found from the book.]"
    doc_context = _truncate(doc_context)

    # The task description now includes the retrieved context and instructs the agent on how to behave.
    task = Task(
        description=(
            "Answer the user's question based on the [Retrieved Context from Book] provided below. "
            "If you need to verify a specific Black Swan technique, use your `Browse BSW Tool`. "
            "Always cite page numbers from the book context where possible.\n\n"
            f"Question: {question}\n\n"
            f"[Retrieved Context from Book]:\n{doc_context}\n"
        ),
        expected_output=(
            "A clear, practical answer grounded in the book context. If the web tool was used, "
            "mention it and cite the sources."
        ),
        agent=coach,
    )

    # Add the Langfuse callback handler for observability
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    handler = (
        CallbackHandler(public_key=langfuse_public_key, secret_key=langfuse_secret_key)
        if (langfuse_public_key and langfuse_secret_key)
        else None
    )

    crew = Crew(agents=[coach], tasks=[task], verbose=True)

    try:
        # Pass the handler to the kickoff method
        result = crew.kickoff(callbacks=[handler] if handler else None)
        return {"text": str(result)}
    except Exception as e:
        logging.error(f"Crew execution error: {e}", exc_info=True)
        return {"text": f"Error: {e}"}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- CrewAI PDF RAG (Agentic) ---")
    question = "What is BATNA?"

    try:
        out = asyncio.run(crew_answer(question))
        print("\n--- Answer ---")
        print(out.get("text", "No answer"))
        print("----------------")
    except Exception as e:
        logging.error(f"Execution error: {e}", exc_info=True)

    print("\n--- Done ---")
