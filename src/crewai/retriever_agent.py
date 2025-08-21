import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from crewai import Agent

# Import the existing retriever function
from src.retriever.get_retriever import get_chroma_retriever

# Import the configuration loading function
from utils.load_config import load_agent_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# --- Configuration Loading ---
# Removed the local load_agent_config function as it's now imported from utils.load_config


# --- Factory Functions ---
def make_llm() -> ChatGoogleGenerativeAI:
    """Factory function to create and return the LLM instance."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", convert_system_message_to_human=True)
        return llm
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        raise


def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Factory function to create and return the Embeddings instance."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize Embeddings: {e}")
        raise


# Removed make_retriever as it's now imported from src.retriever.get_retriever


# --- Agent Definition ---
def create_retriever_agent(
    agent_config: Dict[str, Any],
    default_agent_properties: Dict[str, Any],
    llm: ChatGoogleGenerativeAI,
    retriever: Optional[Chroma],
) -> Agent:
    """Creates and returns the Retriever Agent instance."""
    if not retriever:
        error_message = "Retriever is not initialized. Cannot create Retriever Agent."
        logging.error(error_message)
        raise RuntimeError(error_message)

    # Get primary config values, falling back to defaults
    role = agent_config.get("role")
    goal = agent_config.get("goal")
    backstory = agent_config.get("backstory")
    verbose = agent_config.get("verbose")
    allow_delegation = agent_config.get("allow_delegation")

    # Use defaults if primary config values are missing
    if role is None:
        role = default_agent_properties.get("role")
    if goal is None:
        goal = default_agent_properties.get("goal")
    if backstory is None:
        backstory = default_agent_properties.get("backstory")
    if verbose is None:
        verbose = default_agent_properties.get("verbose")
    if allow_delegation is None:
        allow_delegation = default_agent_properties.get("allow_delegation")

    # Fail fast if essential properties are still missing after checking defaults
    if role is None or goal is None or backstory is None:
        missing_props = []
        if role is None:
            missing_props.append("role")
        if goal is None:
            missing_props.append("goal")
        if backstory is None:
            missing_props.append("backstory")
        error_message = f"Missing essential agent properties in configuration: {', '.join(missing_props)}"
        logging.error(error_message)
        raise RuntimeError(error_message)

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,  # verbose and allow_delegation can be False/None if not provided
        allow_delegation=allow_delegation,
        llm=llm,
    )


# --- Helper Functions ---
def format_docs(docs: List[Document]) -> str:
    """Formats a list of Document objects into a single string, including page numbers if available."""
    formatted_text = ""
    for i, doc in enumerate(docs):
        page_number = doc.metadata.get("page", "N/A")
        formatted_text += f"--- Document {i+1} (Page: {page_number}) ---\n"
        formatted_text += doc.page_content + "\n\n"
    return formatted_text.strip()


async def get_retriever_excerpts(query: str, retriever: Optional[Chroma]) -> str:
    """Retrieves and formats excerpts using the provided retriever."""
    if retriever is None:
        raise RuntimeError("Retriever is not initialized.")
    try:
        docs = retriever.invoke(query)
        return format_docs(docs)
    except Exception as e:
        logging.error(f"An error occurred during retrieval for query '{query}': {e}")
        raise RuntimeError(f"Retrieval failed: {e}") from e


# --- Initialization Function ---
def initialize_retriever_agent_components() -> (
    tuple[Optional[Agent], Optional[ChatGoogleGenerativeAI], Optional[GoogleGenerativeAIEmbeddings], Optional[Chroma]]
):
    """Initializes and returns the retriever agent and its dependencies."""
    try:
        # Use the imported load_agent_config function
        config_data = load_agent_config()
        agent_config = config_data.get("retriever_agent", {})
        default_agent_properties = config_data.get("default_agent_properties", {})

        if not agent_config and not default_agent_properties:  # Check if any config was loaded
            logging.error("Failed to load agent configuration. Cannot initialize components.")
            return None, None, None, None

        llm_instance = make_llm()
        embeddings_instance = make_embeddings()

        # Use the imported get_chroma_retriever function and await it
        retriever_instance = asyncio.run(get_chroma_retriever())

        # create_retriever_agent will now raise an error if retriever_instance is None or essential config is missing
        retriever_agent_instance = create_retriever_agent(
            agent_config=agent_config,
            default_agent_properties=default_agent_properties,  # Pass defaults
            llm=llm_instance,
            retriever=retriever_instance,
        )
        return retriever_agent_instance, llm_instance, embeddings_instance, retriever_instance

    except Exception as e:
        logging.error(f"An error occurred during component initialization: {e}")
        # Return None for all components if initialization fails
        return None, None, None, None


# --- Main Execution Block (Demonstrating a simplified RAG pipeline test) ---
if __name__ == "__main__":
    print("--- Demonstrating Simplified RAG Pipeline Test ---")
    print("This test simulates the retriever part of a RAG pipeline, similar to how Langchain tests might use it.")
    print("In a full RAG pipeline, the retrieved content would be passed to an LLM to generate the final answer.")

    try:
        # Initialize components and agent
        retriever_agent_instance, llm_instance, embeddings_instance, retriever_instance = (
            initialize_retriever_agent_components()
        )

        if retriever_agent_instance is None:
            print("Failed to initialize retriever agent components. Exiting test.")
            exit()

        print("\n--- Retriever Agent Instance Created ---")
        print(f"Agent Role: {retriever_agent_instance.role}")
        print(f"Agent Goal: {retriever_agent_instance.goal}")
        backstory_preview = (
            retriever_agent_instance.backstory[:70] + "..." if retriever_agent_instance.backstory else "N/A"
        )
        print(f"Agent Backstory: {backstory_preview}")

        # Test the get_retriever_excerpts function, simulating a RAG pipeline step
        if retriever_instance:
            print("\n--- Testing Retrieval Functionality ---")
            sample_question = "What is BATNA?"  # Using the user's example question
            print(f"Testing RAG pipeline with question: '{sample_question}'")

            try:
                # Execute the async function to get retriever excerpts
                retrieved_content = asyncio.run(get_retriever_excerpts(sample_question, retriever_instance))

                # Simulate the output format that a RAG pipeline might return
                print("\n--- Retrieved Context for RAG Pipeline ---")
                print(retrieved_content)
                print("------------------------------------------")

            except Exception as e:
                print(f"\nError during test execution: {e}")
        else:
            print("\nRetriever not initialized. Cannot perform retrieval test.")

    except Exception as e:
        # This catch block handles errors during the initialization phase itself
        logging.error(f"An error occurred during the test setup: {e}")

    print("\n--- Simplified RAG Pipeline Test Demonstration Complete ---")
