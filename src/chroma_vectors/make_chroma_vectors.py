import logging
import argparse
from src.ingestion import ingest_pdf
from src.config import PDF_SOURCE_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Command-line interface for the PDF ingestion pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into the Chroma vector store."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default=PDF_SOURCE_PATH,
        help=f"Path to the PDF file to ingest. Defaults to the value in config: {PDF_SOURCE_PATH}",
    )
    args = parser.parse_args()

    logging.info(f"--- Starting PDF Ingestion Pipeline for {args.filepath} ---")
    try:
        ingest_pdf(pdf_path=args.filepath)
        logging.info(
            f"--- PDF Ingestion for {args.filepath} Completed Successfully ---"
        )
    except Exception as e:
        logging.error(
            f"An error occurred during the ingestion pipeline: {e}", exc_info=True
        )
        logging.info("--- PDF Ingestion Pipeline Failed ---")


if __name__ == "__main__":
    main()
