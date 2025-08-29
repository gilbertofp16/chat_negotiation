# src/retriever/get_retriever.py
import logging
import os
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma  # <-- same wrapper as indexing
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_DB_PATH, DEFAULT_RETRIEVER_K, EMBEDDING_MODEL_NAME


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def get_chroma_retriever(
    k: int = DEFAULT_RETRIEVER_K,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> BaseRetriever:
    final_persist_directory = persist_directory or CHROMA_DB_PATH
    final_collection_name = collection_name or os.getenv(
        "CHROMA_COLLECTION_NAME", "pdf_ephemeral"
    )

    if not final_collection_name:
        raise ValueError("Chroma collection name must be set.")

    path = Path(final_persist_directory)
    if not path.exists():
        raise FileNotFoundError(
            f"Chroma persistence path not found: {persist_directory}"
        )

    embeddings = make_embeddings()  # used for query-time embedding
    vectorstore = Chroma(
        persist_directory=str(path),
        collection_name=final_collection_name,
        embedding_function=embeddings,
    )

    # optional sanity check
    try:
        count = vectorstore._collection.count()  # type: ignore[attr-defined]
    except Exception:
        count = None
    if count == 0:
        raise RuntimeError(
            f"Chroma collection '{final_collection_name}' at '{path}' is empty. "
            f"Reindex the PDF using the same collection name and embedding model '{EMBEDDING_MODEL_NAME}'."
        )
    logging.info(
        f"Loaded Chroma collection '{final_collection_name}' at '{path}', vectors={count}"
    )

    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": int(k)}
    )
