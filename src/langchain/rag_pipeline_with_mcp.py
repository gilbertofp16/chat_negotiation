import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from langchain.agents import AgentExecutor, create_tool_calling_agent
from src.config import ACTIVE_PROMPT_LANGCHAIN_COACH_WITH_MCP, REASONING_MODEL_NAME
from src.retriever.get_retriever import get_chroma_retriever
from src.tools.bsw_tool import get_bsw_tool
from utils.load_config import load_llm_configurations, load_prompt_template

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables
load_dotenv()


async def create_agent_executor(
    system_prompt_template: str,
    model_name: str = REASONING_MODEL_NAME,
    llm_params: dict = None,
    langfuse_handler: CallbackHandler = None,
):
    """
    Creates a LangChain agent that can use the BSW tool and a retriever.
    """
    # The agent needs access to both the retriever for book knowledge
    # and the BSW tool for external web searches.
    retriever = get_chroma_retriever()
    bsw_tool = await get_bsw_tool()
    tools = [bsw_tool]

    # The prompt is crucial. It instructs the agent on its role, tells it what tools
    # it has, and provides placeholders for the question, context, and agent scratchpad.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt_template
                + "\n\n[Retrieved Context from Book]\n{context}\n\n",
            ),
            ("human", "{question}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name, convert_system_message_to_human=True, **(llm_params or {})
    )

    # Create the tool-calling agent. This binds the LLM, tools, and prompt together.
    agent = create_tool_calling_agent(llm, tools, prompt_template)

    # The AgentExecutor is the runtime for the agent. It takes the agent's decisions
    # and executes the corresponding tools.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    if langfuse_handler:
        agent_executor = agent_executor.with_config({"callbacks": [langfuse_handler]})

    return agent_executor, retriever


async def get_answer(question: str, model_name: str = None):
    """
    Gets an answer from the agentic RAG pipeline.
    """
    if model_name is None:
        model_name = os.environ.get("REASONING_MODEL_NAME", REASONING_MODEL_NAME)

    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    langfuse = None
    handler = None
    trace = None

    if langfuse_public_key and langfuse_secret_key:
        langfuse = Langfuse(
            public_key=langfuse_public_key, secret_key=langfuse_secret_key
        )
        trace = langfuse.trace(
            name="RAG-Pipeline",
            input={"question": question},
            metadata={"model_name": model_name},
        )
        handler = CallbackHandler(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            # This will link the agent's trace to the same session
            trace_name="Agent-Execution",
            session_id=trace.id,
        )

    system_prompt_template_content = load_prompt_template(
        f"prompts/{ACTIVE_PROMPT_LANGCHAIN_COACH_WITH_MCP}.yaml"
    )
    llm_configurations = load_llm_configurations()

    # Create the agent executor and the retriever
    agent_executor, retriever = await create_agent_executor(
        system_prompt_template_content,
        model_name,
        llm_configurations,
        handler,
    )

    # 1. Retrieve context from the book first. This is always the primary source.
    if trace:
        retrieval_span = trace.span(
            name="Retriever-Chroma",
            input={"question": question},
            metadata={"retriever_type": "chroma"},
        )

    retrieved_docs = retriever.invoke(question)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    if trace and "retrieval_span" in locals():
        documents_for_logging = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in retrieved_docs
        ]
        retrieval_span.end(
            output={
                "documents": documents_for_logging,
                "context_string": context,
            }
        )

    try:
        # 2. Invoke the agent. The agent receives the question and the retrieved book
        #    context. It can then choose to use the `browse_bsw` tool if it deems it
        #    necessary to sanity-check or enrich its answer.
        response = await agent_executor.ainvoke(
            {"question": question, "context": context}
        )
        output = {"text": response.get("output", "No answer found.")}
        if trace:
            trace.update(output=output)
        return output
    except Exception as e:
        logging.error(f"Error during RAG pipeline execution: {e}", exc_info=True)
        output = {"text": "An error occurred while processing your request."}
        if trace:
            trace.update(output=output, level="ERROR", status_message=str(e))
        return output


if __name__ == "__main__":
    sample_question = "I need no oriented question to ask to do a interview on Friday? but the person is HR from Australia needs to be Australian English maybe to british English?"
    print(f"Testing RAG pipeline with question: '{sample_question}'")

    try:
        answer = asyncio.run(get_answer(sample_question))
        print("\n--- RAG Pipeline Answer ---")
        print(answer.get("text", "No answer found."))
        print("---------------------------")
    except Exception as e:
        print(f"\nError during test execution: {e}")
