import logging
import os

import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Configuration constants for retriever
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "data/chroma")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")
DEFAULT_RETRIEVER_K = int(os.environ.get("DEFAULT_RETRIEVER_K", 3))


def get_chroma_retriever(k: int = DEFAULT_RETRIEVER_K) -> Chroma.as_retriever:
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
