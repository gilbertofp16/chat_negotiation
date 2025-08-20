import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler

# Import retriever function
from src.retriever.get_retriever import get_chroma_retriever

# Import configuration loading functions
from utils.load_config import load_llm_configurations, load_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# Configuration constants
# CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, DEFAULT_RETRIEVER_K are now in src/retriever/get_retriever.py
RAG_MODEL_NAME = os.environ.get("RAG_MODEL_NAME", "models/gemini-1.5-pro-latest")  # Model for RAG pipeline


def create_rag_chain(
    retriever,
    system_prompt_template: str,
    model_name: str = RAG_MODEL_NAME,
    llm_params: dict = None,
    langfuse_handler: CallbackHandler = None,
):
    """
    Creates the RAG chain using LangChain.

    Args:
        retriever: The retriever object.
        system_prompt_template: The system prompt template string.
        model_name: The name of the LLM model to use.
        llm_params: Dictionary of LLM parameters (e.g., temperature).
        langfuse_handler: Optional Langfuse callback handler.

    Returns:
        A LangChain runnable chain.
    """
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt_template), ("human", "{question}")])

    llm = ChatGoogleGenerativeAI(
        model=model_name, convert_system_message_to_human=True, **llm_params if llm_params else {}
    )

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm

    if langfuse_handler:
        rag_chain = rag_chain.with_config({"callbacks": [langfuse_handler]})

    return rag_chain


def get_answer(
    question: str, retriever_k: int = 3, model_name: str = None
):  # Default retriever_k to 3 here, as it's now local to get_chroma_retriever
    """
    Orchestrates the RAG pipeline to get an answer.

    Args:
        question: The user's question.
        retriever_k: Number of documents to retrieve for context.
        model_name: The LLM model to use for generating the answer. If None,
                    it will try to get it from environment variables or use a default.

    Returns:
        A dictionary containing the answer text.
    """
    if model_name is None:
        model_name = os.environ.get("RAG_MODEL_NAME", RAG_MODEL_NAME)

    retriever = get_chroma_retriever(k=retriever_k)

    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    handler = None
    if langfuse_public_key and langfuse_secret_key:
        handler = CallbackHandler(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
        )

    prompt_file = "prompts/langchain/negotiation_coach.yaml"
    system_prompt_template_content = load_prompt_template(prompt_file)

    llm_configurations = load_llm_configurations()

    rag_chain = create_rag_chain(retriever, system_prompt_template_content, model_name, llm_configurations, handler)

    try:
        response = rag_chain.invoke(question)
        return {"text": response.content}
    except Exception as e:
        logging.error(f"Error during RAG pipeline execution: {e}")
        return {"text": "An error occurred while processing your request."}


if __name__ == "__main__":
    sample_question = "What is BATNA?"
    print(f"Testing RAG pipeline with question: '{sample_question}'")

    try:
        answer = get_answer(sample_question)
        print("\n--- RAG Pipeline Answer ---")
        print(answer.get("text", "No answer found."))
        print("---------------------------")
    except Exception as e:
        print(f"\nError during test execution: {e}")

# Example of how to use the retriever tool (if needed separately)
# def get_retriever_tool(retriever):
#     """Creates a retriever tool for agents."""
#     return create_retriever_tool(
#         retriever,
#         "negotiation_book_retriever",
#         "Searches and returns passages from the negotiation book."
#     )
