import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Model Names ---
EMBEDDING_MODEL = "gemini/text-embedding-004"
REASONING_MODEL = "gemini/gemini-1.5-pro-latest"
REASONING_MODEL_NAME = "models/gemini-1.5-pro-latest"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- Data Paths ---
PDF_SOURCE_PATH = "data/source/negotiation_book.pdf"
CHROMA_DB_PATH = "data/chroma"

# --- Observability ---
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Active Prompts ---
# Define which prompt versions to use for different components
ACTIVE_PROMPT_LANGCHAIN_COACH = os.getenv(
    "ACTIVE_PROMPT_LANGCHAIN_COACH", "langchain/negotiation_coach"
)
ACTIVE_PROMPT_LANGCHAIN_COACH_WITH_MCP = os.getenv(
    "ACTIVE_PROMPT_LANGCHAIN_COACH", "langchain/negotiation_coach_with_mcp"
)


ACTIVE_PROMPT_CREWAI_COACH = os.getenv(
    "ACTIVE_PROMPT_CREWAI_COACH", "langchain/negotiation_coach"
)  # Defaulting to the same for now, can be changed later

DEFAULT_RETRIEVER_K = 3
