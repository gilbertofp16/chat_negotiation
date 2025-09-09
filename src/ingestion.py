import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns Google Generative AI Embeddings using the model from config.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def ingest_pdf(
    pdf_path: str,
    persist_directory: str = CHROMA_DB_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    collection_name: str = "pdf_ephemeral",
):
    """
    Load a PDF, chunk it, embed it, and store it in a persistent Chroma vector store.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF not found: {pdf_path}")
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logging.info(f"Loading PDF from {pdf_path}")
    docs = PyPDFLoader(pdf_path).load()

    logging.info("Splitting documents into chunks")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)

    logging.info("Creating embeddings and ingesting into Chroma")
    embeddings = make_embeddings()
    vectorstore = Chroma.from_documents(
        splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    logging.info(f"Persisting vector store to {persist_directory}")
    vectorstore.persist()
    logging.info("Ingestion complete.")
