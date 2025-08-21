import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler

from src.config import REASONING_MODEL_NAME

# Import retriever function
from src.mcp_servers.mcp_client import mcp_search_sync, web_context_from_results
from src.retriever.get_retriever import get_chroma_retriever

# Import configuration loading functions
from utils.load_config import load_llm_configurations, load_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()


def create_rag_chain(
    retriever,
    system_prompt_template: str,
    model_name: str = REASONING_MODEL_NAME,
    llm_params: dict = None,
    langfuse_handler: CallbackHandler = None,
    web_context: str = "",  # <-- NEW
):
    """
    Creates the RAG chain using LangChain, now with optional web_context.
    """
    # Merge web and doc contexts into the system prompt
    sys_prompt = (
        system_prompt_template
        + "\n\n[Chroma context]\n{context}\n\n[Web context]\n"
        + (web_context or "[none]")
        + "\n\nInstruction: Answer the question. Then state whether web context aligns with Chroma "
        "(agrees / partial / conflicts) and list any discrepancies."
    )

    prompt_template = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "{question}")])

    llm = ChatGoogleGenerativeAI(model=model_name, convert_system_message_to_human=True, **(llm_params or {}))

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt_template | llm

    if langfuse_handler:
        rag_chain = rag_chain.with_config({"callbacks": [langfuse_handler]})

    return rag_chain


async def get_answer(question: str, retriever_k: int = 3, model_name: str = None):
    if model_name is None:
        model_name = os.environ.get("REASONING_MODEL_NAME", REASONING_MODEL_NAME)

    retriever = get_chroma_retriever(k=retriever_k)

    # Build focused web query via MCP and format as context
    web_q = f'{question} site:blackswanltd.com OR "Black Swan Group" negotiation'
    web_items = mcp_search_sync(web_q, max_results=6)
    web_ctx = web_context_from_results(web_items)

    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    handler = (
        CallbackHandler(public_key=langfuse_public_key, secret_key=langfuse_secret_key)
        if (langfuse_public_key and langfuse_secret_key)
        else None
    )

    prompt_file = "prompts/langchain/negotiation_coach.yaml"
    system_prompt_template_content = load_prompt_template(prompt_file)
    llm_configurations = load_llm_configurations()

    rag_chain = create_rag_chain(
        retriever,
        system_prompt_template_content,
        model_name,
        llm_configurations,
        handler,
        web_context=web_ctx,
    )

    try:
        response = await rag_chain.ainvoke(question)
        return {"text": response.content}
    except Exception as e:
        logging.error(f"Error during RAG pipeline execution: {e}")
        return {"text": "An error occurred while processing your request."}


if __name__ == "__main__":
    sample_question = "What is BATNA?"
    print(f"Testing RAG pipeline with question: '{sample_question}'")

    try:
        answer = asyncio.run(get_answer(sample_question))
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
