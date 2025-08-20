import streamlit as st
import os
import asyncio  # Import asyncio
from dotenv import load_dotenv

# Import RAG pipeline functions
from src.langchain.rag_pipeline import get_answer

# Import constants from config for UI elements
from src.config import REASONING_MODEL_NAME, DEFAULT_RETRIEVER_K

# Load environment variables (dotenv is loaded in config.py, so this might be redundant if config.py is the sole entry point)
# load_dotenv() # Removed as config.py handles it.


def run_negotiation_coach_app():
    """
    Sets up and runs the Streamlit UI for the negotiation coach.
    """
    st.title("Negotiation Coach")
    st.write("Welcome to your personal negotiation coach!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- RAG Pipeline Integration ---

    # Model Selection
    # Get the default model name from config
    configured_default_model = REASONING_MODEL_NAME

    # Define available models, ensuring the configured default is included
    common_models = ["models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest"]
    available_models = list(dict.fromkeys([configured_default_model] + common_models))  # Ensure unique and order

    # Determine the model to pre-select in the dropdown
    # Use the configured default model directly
    selected_model = st.selectbox(
        "Choose your LLM model:",
        available_models,
        index=available_models.index(configured_default_model) if configured_default_model in available_models else 0,
    )

    # Retriever K value (optional, can be set via env var or UI)
    # For simplicity, using default for now, but could add a number input
    retriever_k = DEFAULT_RETRIEVER_K  # This will be passed to get_answer

    # User input for question using chat input
    user_question = st.chat_input("Ask a question about negotiation:")

    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Generating answer..."):
            try:
                # Call the RAG pipeline using asyncio.run()
                answer = asyncio.run(
                    get_answer(question=user_question, retriever_k=retriever_k, model_name=selected_model)
                )
                response_text = answer.get("text", "Could not retrieve an answer.")

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                # Display assistant message
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Add error message to chat history as well
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})


if __name__ == "__main__":
    run_negotiation_coach_app()
