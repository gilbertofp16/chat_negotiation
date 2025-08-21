from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_DB_PATH, DEFAULT_RETRIEVER_K, EMBEDDING_MODEL_NAME


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Creates an embeddings client.

    See also:
        src.config
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def get_chroma_retriever(k: int = DEFAULT_RETRIEVER_K) -> BaseRetriever:
    """Creates a Chroma retriever backed by the configured persistence path.

    Args:
        k: Number of documents to retrieve.

    Returns:
        A retriever instance.

    See also:
        src.config
    """
    path = Path(CHROMA_DB_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Chroma persistence path not found: {CHROMA_DB_PATH}")
    embeddings = make_embeddings()
    vectorstore = Chroma(persist_directory=str(path), embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": int(k)})
