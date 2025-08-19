import logging
import os

import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langfuse.callback import CallbackHandler

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "data/chroma")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "models/embedding-001")
RAG_MODEL_NAME = os.environ.get("RAG_MODEL_NAME", "models/gemini-1.5-pro-latest")
DEFAULT_RETRIEVER_K = int(os.environ.get("DEFAULT_RETRIEVER_K", 3))

PROMPT_PATH = os.environ.get("PROMPT_FILE", "prompts/langchain/negotiation_coach.yaml")

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")


# -----------------------------------------------------------------------------
# Prompt Loader
# -----------------------------------------------------------------------------
def load_prompt_template(prompt_file_path: str) -> str:
    """
    Loads a system prompt template from a YAML file.
    YAML structure: { template: "..." }
    """
    try:
        with open(prompt_file_path, "r") as f:
            config = yaml.safe_load(f)
        template = config.get("template", "").strip()
        if not template:
            logging.warning(f"No template found in {prompt_file_path}")
        return template
    except FileNotFoundError:
        logging.error(f"Prompt file not found: {prompt_file_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"YAML parse error in {prompt_file_path}: {e}")
        raise


# -----------------------------------------------------------------------------
# Retriever Utils
# -----------------------------------------------------------------------------
def get_chroma_retriever(k: int = DEFAULT_RETRIEVER_K) -> BaseRetriever:
    """
    Initializes Chroma vector store and returns a retriever.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logging.error(f"Error initializing Chroma retriever: {e}")
        raise


# -----------------------------------------------------------------------------
# RAG Chain Factory
# -----------------------------------------------------------------------------
def create_rag_chain(
    retriever: BaseRetriever,
    system_prompt: str,
    model_name: str = RAG_MODEL_NAME,
    langfuse_handler: CallbackHandler = None,
):
    """
    Builds a LangChain RAG chain.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt + "\n\nContext:\n{context}"), ("human", "{question}")]
    )

    llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True)

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm

    if langfuse_handler:
        rag_chain = rag_chain.with_config({"callbacks": [langfuse_handler]})

    return rag_chain


# -----------------------------------------------------------------------------
# Pipeline Entrypoint
# -----------------------------------------------------------------------------
def get_answer(question: str, retriever_k: int = DEFAULT_RETRIEVER_K, model_name: str = None):
    """
    Executes the RAG pipeline end-to-end and returns an answer.
    """
    # Model selection
    if model_name is None:
        model_name = os.environ.get("RAG_MODEL_NAME", RAG_MODEL_NAME)

    # Retriever
    retriever = get_chroma_retriever(k=retriever_k)

    # Langfuse handler
    handler = None
    if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
        handler = CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
        )

    # Prompt
    system_prompt = load_prompt_template(PROMPT_PATH)

    # RAG chain
    rag_chain = create_rag_chain(retriever, system_prompt, model_name, handler)

    # Invoke
    try:
        response = rag_chain.invoke(question)
        text = getattr(response, "content", response)
        return {"text": text}
    except Exception as e:
        logging.error(f"Error during RAG execution: {e}")
        return {"text": "An error occurred while processing your request."}


# -----------------------------------------------------------------------------
# Local Test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    q = "What is no oriented question? Give an example."
    logging.info(f"Testing RAG pipeline with: {q}")
    ans = get_answer(q)
    print("\n--- RAG Pipeline Answer ---")
    print(ans.get("text", "No answer"))
    print("---------------------------")
