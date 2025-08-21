import logging
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, PDF_SOURCE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns Google Generative AI Embeddings using the model from config.
    """
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)


def build_pdf_retriever(
    pdf_path: str,
    persist_directory: str,
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
        splits, embedding=embeddings, collection_name=collection_name, persist_directory=persist_directory
    )

    # Return a retriever
    vectorstore.persist()


if __name__ == "__main__":
    logging.info("--- Starting PDF Ingestion Pipeline ---")
    try:
        build_pdf_retriever(
            pdf_path=PDF_SOURCE_PATH,
            persist_directory=CHROMA_DB_PATH,
        )
        logging.info("--- PDF Ingestion Pipeline Completed Successfully ---")
    except Exception as e:
        logging.error(f"An error occurred during the ingestion pipeline: {e}", exc_info=True)
        logging.info("--- PDF Ingestion Pipeline Failed ---")
