import logging
import os

import chromadb
import litellm
import pytesseract
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, PDF_SOURCE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_pdf_text_with_ocr(pdf_path: str) -> str:
    """
    Extracts text from a PDF file, performing OCR if necessary.
    """
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file not found at: {pdf_path}")
        return ""

    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                logging.warning(f"No text extracted from page {page_num + 1}. Attempting OCR.")
                try:
                    # Note: PdfReader doesn't give direct page images; for real OCR you'd rasterize the page.
                    # Placeholder logic – may need pdf2image if OCR is required.
                    page_text = pytesseract.image_to_string(page.to_image())
                except Exception as e:
                    logging.error(f"OCR failed for page {page_num + 1}: {e}")
            text += page_text
    except Exception as e:
        logging.error(f"Failed to read PDF: {e}")
    return text


def get_text_chunks(text: str) -> list[str]:
    """
    Splits a long text into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(text)


def ingest_book_pipeline():
    """
    Main function to run the data ingestion process.
    """

    logging.info("Starting data ingestion process...")

    # 1. Extract text from PDF with OCR
    raw_text = get_pdf_text_with_ocr(PDF_SOURCE_PATH)
    if not raw_text:
        logging.error("No text extracted from the PDF. Aborting ingestion.")
        return

    # 2. Split text into chunks
    text_chunks = get_text_chunks(raw_text)
    logging.info(f"Split text into {len(text_chunks)} chunks.")

    # 3. Generate embeddings and store in ChromaDB
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name="negotiation_book")

        # Split text_chunks into batches of 100 for embedding
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i : i + batch_size]
            response = litellm.embedding(model=EMBEDDING_MODEL, input=batch)
            all_embeddings.extend([item["embedding"] for item in response.data])

        # Add embeddings to ChromaDB
        collection.add(
            embeddings=all_embeddings,
            documents=text_chunks,
            ids=[f"id_{i}" for i in range(len(text_chunks))],
        )
        logging.info("Successfully ingested and indexed the book into ChromaDB.")
    except Exception as e:
        logging.error(f"Failed to generate embeddings or store in ChromaDB: {e}")


if __name__ == "__main__":
    ingest_book_pipeline()
