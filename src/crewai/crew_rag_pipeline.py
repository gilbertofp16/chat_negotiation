import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from crewai import Agent, Crew, Task, LLM  # CrewAI’s LLM wrapper (LiteLLM underneath)

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
# Build a retriever from a PDF (in-memory)
# --------------------------------------------------------------------------------------
def build_pdf_retriever(
    pdf_path: str,
    k: int = 8,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    collection_name: str = "pdf_ephemeral",
) -> VectorStoreRetriever:
    """
    Load a PDF, chunk it, embed with text-embedding-004, and create a Chroma retriever.
    Everything is ephemeral (in-memory) unless you pass persist_directory to Chroma.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Load pages with page metadata preserved (PyPDFLoader sets metadata['page'])
    docs = PyPDFLoader(pdf_path).load()

    # Chunk for semantic retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)

    # Embed and index (in-memory Chroma)
    embeddings = make_embeddings()
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        collection_name=collection_name,
        # omit persist_directory for in-memory; add it if you want to persist
        # persist_directory="path/to/chroma_dir"
    )

    # Return a retriever
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})


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


# --------------------------------------------------------------------------------------
# Crew run (no tools)
# --------------------------------------------------------------------------------------
async def crew_answer_with_pdf(question: str, pdf_path: str) -> Dict[str, Any]:
    # Create LLM + agent
    llm = make_llm()
    coach = create_agent(
        role="Master Negotiation Coach",
        goal="Answer questions only with supporting evidence from provided context.",
        backstory="You are a negotiation coach who only cites from retrieved context and includes page numbers.",
        llm=llm,
    )

    # Build an in-memory retriever from the PDF
    retriever = build_pdf_retriever(pdf_path=pdf_path, k=8)

    # Retrieve top chunks and pass them into the Task
    retrieved_context = retrieve_sync(question, retriever)
    if not retrieved_context:
        retrieved_context = "[No results found. Consider re-chunking, increasing k, or verifying the PDF content.]"
    retrieved_context = _truncate(retrieved_context)

    task = Task(
        description=(
            "Answer the user's question using only the retrieved context provided below. "
            "Cite page numbers present in the context. If no evidence is found, say so.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{retrieved_context}\n"
        ),
        expected_output="A clear, well structured answer that cites page numbers from the context.",
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
