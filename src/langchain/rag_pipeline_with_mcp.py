import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler

from langchain.agents import AgentExecutor, create_tool_calling_agent
from src.config import ACTIVE_PROMPT_LANGCHAIN_COACH_WITH_MCP, REASONING_MODEL_NAME
from src.observability import initialize_langfuse
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
    retriever = get_chroma_retriever()
    bsw_tool = await get_bsw_tool()
    tools = [bsw_tool]

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

    agent = create_tool_calling_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    if langfuse_handler:
        agent_executor = agent_executor.with_config({"callbacks": [langfuse_handler]})

    return agent_executor, retriever


async def get_answer(question: str, model_name: str = None):
    if model_name is None:
        model_name = os.environ.get("REASONING_MODEL_NAME", REASONING_MODEL_NAME)

    trace, lc_root_span, handler = initialize_langfuse(question, model_name)

    system_prompt_template_content = load_prompt_template(
        f"prompts/{ACTIVE_PROMPT_LANGCHAIN_COACH_WITH_MCP}.yaml"
    )
    llm_configurations = load_llm_configurations()

    agent_executor, retriever = await create_agent_executor(
        system_prompt_template_content,
        model_name,
        llm_configurations,
        handler,  # LC tool/LLM spans go under lc_root_span
    )

    # Child span for retrieval UNDER the LC root (keeps everything in one tree)
    retrieval_span = None
    if lc_root_span:
        retrieval_span = lc_root_span.span(
            name="Retriever-Chroma",
            input={"question": question},
            metadata={"retriever_type": "chroma"},
        )

    retrieved_docs = (
        get_chroma_retriever().invoke(question)
        if "retriever" not in locals()
        else retriever.invoke(question)
    )
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    if retrieval_span:
        documents_for_logging = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in retrieved_docs
        ]
        retrieval_span.end(
            output={"documents": documents_for_logging, "context_string": context}
        )

    try:
        # Agent + tool calls will appear as child spans via CallbackHandler
        response = await agent_executor.ainvoke(
            {"question": question, "context": context}
        )
        output = {"text": response.get("output", "No answer found.")}

        if lc_root_span:
            lc_root_span.end(output={"agent_output": output})
        if trace:
            trace.update(output=output)

        return output
    except Exception as e:
        logging.error(f"Error during RAG pipeline execution: {e}", exc_info=True)
        output = {"text": "An error occurred while processing your request."}

        if lc_root_span:
            lc_root_span.end(
                output={"agent_output": output},
                level="ERROR",
                status_message=str(e),
            )
        if trace:
            trace.update(output=output, level="ERROR", status_message=str(e))

        return output
