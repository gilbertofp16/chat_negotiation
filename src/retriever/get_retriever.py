import logging
import os

import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_DB_PATH, DEFAULT_RETRIEVER_K, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


async def get_chroma_retriever(k: int = DEFAULT_RETRIEVER_K) -> Chroma.as_retriever:
    """
    Initializes Chroma vector store and returns a retriever.

    Args:
        k: Number of documents to retrieve.

    Returns:
        A Chroma retriever instance.
    """
    try:
        # Corrected parameter name from model_name to model
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        logging.error(f"Error initializing Chroma retriever: {e}")
        raise
