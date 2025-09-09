import os
import uuid
from typing import Optional, Tuple

from langfuse import Langfuse
from langfuse.client import StatefulTraceClient, StatefulSpanClient
from langfuse.callback import CallbackHandler


def initialize_langfuse(
    question: str, model_name: str
) -> Tuple[
    Optional[StatefulTraceClient],
    Optional[StatefulSpanClient],
    Optional[CallbackHandler],
]:
    """
    Initializes Langfuse tracing if credentials are provided in environment variables.

    Args:
        question (str): The user's question to be included in the trace.
        model_name (str): The name of the model being used.

    Returns:
        A tuple containing the trace, root span, and handler, or (None, None, None)
        if Langfuse is not configured.
    """
    langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not (langfuse_public_key and langfuse_secret_key):
        return None, None, None

    langfuse = Langfuse(public_key=langfuse_public_key, secret_key=langfuse_secret_key)
    run_session_id = str(uuid.uuid4())

    trace = langfuse.trace(
        name="Agent-Execution",
        input={"question": question},
        metadata={"model_name": model_name},
        session_id=run_session_id,
    )

    lc_root_span = trace.span(
        name="Agent-Execution (LC Root)",
        input={"question": question},
        metadata={"source": "langchain"},
    )

    handler = trace.get_langchain_handler()

    return trace, lc_root_span, handler
