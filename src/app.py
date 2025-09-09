import streamlit as st
import os
import asyncio
import tempfile
from dotenv import load_dotenv

# Use the MCP-enabled RAG pipeline and direct ingestion
from src.langchain.rag_pipeline_with_mcp import get_answer
from src.retriever.get_retriever import get_chroma_retriever
from src.ingestion import ingest_pdf
from src.config import REASONING_MODEL_NAME

# Load environment variables
load_dotenv()


@st.cache_resource
def cached_retriever():
    """Caches the Chroma retriever resource."""
    return get_chroma_retriever()


def run_negotiation_coach_app():
    """
    Sets up and runs the Streamlit UI for the negotiation coach.
    """
    st.title("Negotiation Coach")
    st.write("Welcome to your personal negotiation coach!")

    # Initialize session state for tracking processed files
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None

    # --- Sidebar for Settings and Uploads ---
    with st.sidebar:
        st.header("Settings")

        # Model Selection
        configured_default_model = REASONING_MODEL_NAME
        common_models = [
            "models/gemini-1.5-pro-latest",
            "models/gemini-1.5-flash-latest",
            "models/gemini-2.5-pro-latest",
        ]
        available_models = list(
            dict.fromkeys([configured_default_model] + common_models)
        )
        selected_model = st.selectbox(
            "Choose your LLM model:",
            available_models,
            index=available_models.index(configured_default_model)
            if configured_default_model in available_models
            else 0,
        )

        st.header("Upload Knowledge Source")
        uploaded_file = st.file_uploader(
            "Upload a PDF to update the knowledge base", type="pdf"
        )

        if (
            uploaded_file is not None
            and uploaded_file.file_id != st.session_state.processed_file_id
        ):
            with st.spinner("Processing and ingesting the document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    tmp_filepath = tmpfile.name

                # Call the ingestion function directly
                ingest_pdf(pdf_path=tmp_filepath)

                # Clean up the temporary file
                os.remove(tmp_filepath)

                # Update the session state with the new file ID
                st.session_state.processed_file_id = uploaded_file.file_id

                # Clear the retriever cache to force a reload with the new data
                st.cache_resource.clear()

                # Rerun the app to reflect the changes and avoid re-processing
                st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_question = st.chat_input("Ask a question about negotiation:")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Generating answer..."):
            try:
                # Ensure retriever is loaded (it's cached)
                cached_retriever()

                # Call the RAG pipeline
                answer = asyncio.run(
                    get_answer(question=user_question, model_name=selected_model)
                )
                response_text = answer.get("text", "Could not retrieve an answer.")

                # TODO: Add citation display logic if sources are in the answer
                # For example:
                # sources = answer.get("sources", [])
                # if sources:
                #     response_text += "\n\n**Sources:**\n"
                #     for source in sources:
                #         response_text += f"- Page {source.get('page_number')}\n"

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"An error occurred: {e}"}
                )


if __name__ == "__main__":
    run_negotiation_coach_app()
