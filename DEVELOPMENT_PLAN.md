# DEVELOPMENT_PLAN.md

## Overview

Goal: build a local proof of concept negotiation coach chat application that uses an open source book as a knowledge source. The app uses LangChain, runs a Streamlit chat UI, indexes the book into Chroma for retrieval, calls Gemini as the model, adds a tiny MCP browse tool focused on Black Swan negotiation techniques, and sends traces to Langfuse for observability. This is for learning, not for production deployment.

---

## Constraints and assumptions

- Package and env management: **Poetry**. Containerization: **Docker**. Do **not** use pip anywhere.
- Runs locally only. No cloud deployment, staging, or production targets.
- Primary model family: Gemini. Baseline `models/gemini-1.5-pro-latest` for reasoning, `models/text-embedding-004` for embeddings. Flash variants allowed for speed tests.
- Vector store: Chroma with local persistence (`data/chroma`).
- Source: one open licensed **PDF**. If it is a scan, we **must OCR** before indexing.
- Observability: Langfuse (cloud or local) for traces.
- MCP: minimal single purpose tool to browse for “Black Swan negotiation techniques” to sanity check answers, optional in normal flow.
- Cline is used to run and iterate. It does not own state.

---

## Architecture summary

- **Data path:** PDF ingestion → OCR if needed → text chunking → embeddings → Chroma (persisted).
- **Query path (LangChain):** Chroma retriever → RAG prompt → Gemini chat → answer with page citations.
- **Browse tool:** Tiny MCP server exposing one safe “browse/search” action for Black Swan techniques only.
- **UI:** Streamlit chat with session memory, optional local history persistence.
- **Observability:** Langfuse traces for the LangChain path.

---

## Requirement 1. Project scaffolding and environment (Completed)

**Tasks**
1. Initialize repo; create Poetry project; add Dockerfile and `.env.example`.
2. Define dependency groups: core, embeddings, PDF processing/OCR, vector store, UI, observability.
3. Centralize configuration (env loader) for API keys and paths.
4. Provide Make targets (or Poetry scripts): `ingest`, `run`, `trace-check`, `clean`.
5. Pre-commit with basic lint, fmt, and secret scan.

---

## Requirement 2. Data ingestion and indexing of the book (with OCR) (Completed)

**Tasks**
1. **File validation**: confirm license and that pages are extractable.
2. **OCR step**: if the PDF is scanned or contains images:
   - Run OCR locally (Tesseract or equivalent) to produce a text layer.
   - Store an OCR manifest: source hash, OCR tool and version, language, confidence summary.
3. **Parsing**: extract page-wise text with page numbers in metadata.
4. **Chunking**: apply recursive character splitting; define `chunk_size` and `chunk_overlap`.
5. **Embeddings**: generate embeddings with `models/text-embedding-004`.
6. **Indexing**: write chunks and embeddings to Chroma, persist locally.
7. **Index manifest**: record parameters (splitter settings, model versions, collection name, created-at).

---

## Requirement 3. Vector store management with Chroma (Completed)

**Tasks**
1. Define a stable collection name and persist directory (`data/chroma`).
2. Init routine: open collection if present, otherwise instruct user to run ingestion.
3. Documented backup and restore of `data/chroma` for local use.

---

## Requirement 4. Prompt Template Administration

**Tasks**
1.  Create `prompts/langchain/` directory.
2.  Create initial prompt files (e.g., `prompts/langchain/negotiation_coach.yaml`) with name, role, template, and metadata.
3.  Develop `src/prompt_loader.py` utility to load prompts, apply templating, and return framework-specific prompt objects.
4.  Integrate prompt loading into the LangChain component.
5.  Configure active prompts via a setting (e.g., in `.env` or `config.py`).
6.  Add observability hooks to include prompt IDs in Langfuse traces.

---

## Requirement 5. RAG pipeline with LangChain

**Tasks**
1. Author a system prompt: “Negotiation coach, grounded on the book, cite page numbers.”
2. Configure Chroma retriever (`k`, MMR optional).
3. Compose a simple “stuff” combine chain for responses.
4. Allow model selection (e.g., `models/gemini-1.5-pro` vs `models/gemini-1.5-flash-latest`) via UI setting.
5. Add Langfuse callback to capture traces and metadata (model, k, timings).

---

## Requirement 6. Tiny MCP browse tool (Black Swan sanity checks)

**Tasks**
1.  [x] Implement a **minimal MCP server** over stdio with a single action: `browse_bsw(topic)` that constrains queries to the Black Swan negotiation domain.
2.  [x] Create a LangChain `Tool` wrapper around the MCP client to make it available to agents.
3.  [x] Integrate the tool into the LangChain agent executor, allowing the LLM to decide when to call it.
4.  [x] Update the LangChain system prompt to instruct the agent on the proper, secondary use of the tool.
5.  [x] Join the Langfuse logs for the retriever and the agent execution into a single, unified trace.


---

## Requirement 7. Streamlit chat UI with memory and caching

**Tasks**
1. Chat layout rendering history and latest answer with page citations.
2. Upload widget to load a new PDF and trigger ingestion.
3. Session memory for chat; optional simple local persistence for history.
4. Settings panel: switch model.
5. Cache stable resources (Chroma client, embeddings config).

---

## Requirement 8. Observability with Langfuse

**Tasks**
1. Initialize Langfuse early and verify connection.
2. Attach callback/middleware in the LangChain path.
3. Standardize trace metadata: session id, path, model, k, timing.
4. Create a saved view in Langfuse for “RAG runs.”

---

## Requirement 9. Deployment and operations (Local only)

**Tasks**
1. **Local POC only**: run via Poetry or Docker.
2. Provide `docker compose` or Make targets for one-command bring-up.
3. Confirm Chroma persistence under a mounted local volume.
4. Document local health checks and restart steps.

---

## Requirement 10. Unit Testing

**Tasks**
1.  Implement unit tests for all major components.
2.  Ensure test coverage for `src/prompt_loader.py`.
3.  Ensure test coverage for `src/observability.py`.
4.  Ensure test coverage for `src/retriever/get_retriever.py`.
5.  Ensure test coverage for `utils/load_config.py`.
6.  Exclude `crewai` components from unit testing as per requirements.
7.  Integrate a test runner (e.g., pytest) into the project.
8.  Add a `test` script to `pyproject.toml` or `Makefile`.

---

## QA CHECKLIST

- [ ] All user instructions followed
- [ ] All requirements implemented and tested
- [ ] No critical code smell warnings
- [ ] Code follows project conventions and standards
- [ ] Documentation is updated and accurate if needed
- [ ] Security considerations addressed
- [ ] Performance requirements met
- [ ] Integration points verified
- [ ] Deployment readiness confirmed

---

## Success criteria

- Can ingest the target PDF (including OCR when needed) and persist to Chroma locally.
- Streamlit chat answers grounded in the book with correct page citations for at least 9 out of 10 seed questions.
- The **LangChain** path functions using Gemini and returns properly formatted outputs.
- Tiny MCP browse tool works and clearly indicates when it was used, returning a short summary and 1–2 sources about Black Swan techniques.
- Langfuse shows complete traces for the LangChain path with key metadata (model, k, timings).
- Entire system runs locally via Poetry or Docker with a single command.
