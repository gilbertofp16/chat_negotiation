# DEVELOPMENT_PLAN.md

## Overview

Goal: build a local proof of concept negotiation coach chat application that uses an open source book as a knowledge source. The app supports two orchestration paths, LangChain and CrewAI, runs a Streamlit chat UI, indexes the book into Chroma for retrieval, calls Gemini via **LiteLLM** as the unified LLM gateway, adds a tiny MCP browse tool focused on Black Swan negotiation techniques, and sends traces to Langfuse for observability. This is for learning, not for head to head comparison or production deployment.

---

## Constraints and assumptions

- Package and env management: **Poetry**. Containerization: **Docker**. Do **not** use pip anywhere.
- Runs locally only. No cloud deployment, staging, or production targets.
- Primary model family: Gemini, routed through **LiteLLM**. Baseline `gemini/gemini-2.5-pro` for reasoning, `gemini/text-embedding-004` for embeddings. Flash variants allowed for speed tests.
- Vector store: Chroma with local persistence (`data/chroma`).
- Source: one open licensed **PDF**. If it is a scan, we **must OCR** before indexing.
- Observability: Langfuse (cloud or local) for traces.
- MCP: minimal single purpose tool to browse for “Black Swan negotiation techniques” to sanity check answers, optional in normal flow.
- Cline is used to run and iterate. It does not own state.
- Follow a Cookiecutter style project layout with src, tests, pyproject.toml, README.md, and docs.

---

## Architecture summary

- **Data path:** PDF ingestion → OCR if needed → text chunking → embeddings → Chroma (persisted).
- **Query path A (LangChain):** Chroma retriever → RAG prompt → Gemini via LiteLLM → answer with page citations.
- **Query path B (CrewAI):** Retriever agent → Coach agent → Gemini via LiteLLM → answer with page citations.
- **Browse tool:** Tiny MCP server exposing one safe “browse/search” action for Black Swan techniques only.
- **UI:** Streamlit chat with session memory, optional local history persistence.
- **Observability:** Langfuse traces for both paths.
- **Model routing:** LiteLLM unifies all LLM calls, so model switching is consistent across LangChain and CrewAI.

---

## Requirement 1. Project scaffolding and environment

**Tasks**
1. Initialize repo; create Poetry project; add Dockerfile and `.env.example`.
2. Define dependency groups: core, embeddings, PDF processing/OCR, vector store, UI, observability.
3. Add **LiteLLM** dependency for unified LLM routing.
4. Centralize configuration (env loader) for API keys and paths.
5. Provide Make targets (or Poetry scripts): `ingest`, `run`, `trace-check`, `clean`.
6. Pre-commit with basic lint, fmt, and secret scan.

**Dependencies**  
None; this unlocks everything else.

**Approaches**  
- Poetry for locking and reproducibility.  
- Docker for consistent local runtime.  
- LiteLLM for unified LLM abstraction.

**Trade-offs**  
- LiteLLM adds a layer but simplifies switching between Gemini models and provides built-in observability hooks.

**Testing strategy**  
- Fresh-clone install test.  
- Container build succeeds and runs `streamlit` entry point.  
- Sanity test call through LiteLLM to Gemini.

---

## Requirement 2. Data ingestion and indexing of the book (with OCR)

*(unchanged except embeddings also go through LiteLLM)*

**Change**  
- Use LiteLLM embeddings API (`gemini/text-embedding-004`) instead of direct client.

---

## Requirement 3. Vector store management with Chroma

*(unchanged)*

---

## Requirement 4. RAG pipeline with LangChain

**Change**  
- LangChain uses LiteLLM integration (`ChatLiteLLM`) instead of `ChatGoogleGenerativeAI`.  
- Model names follow LiteLLM convention (`gemini/gemini-2.5-pro`, `gemini/gemini-1.5-flash`).

**Rationale**  
- Keeps model selection unified across LangChain and CrewAI.  
- Easy to swap or experiment with OpenAI, Anthropic, or other providers in the future.

---

## Requirement 5. Agentic variant with CrewAI (for learning, not comparison)

**Change**  
- CrewAI agents call Gemini exclusively through LiteLLM.  
- Crew definition specifies `llm="gemini/gemini-2.5-pro"` or other LiteLLM names.

**Rationale**  
- Consistency with LangChain path.  
- Enables toggling models without modifying CrewAI logic.

---

## Requirement 6. Tiny MCP browse tool (Black Swan sanity checks)

*(unchanged)*

---

## Requirement 7. Streamlit chat UI with memory and caching

**Change**  
- Model selection dropdown populated from LiteLLM models config (e.g., `gemini/gemini-2.5-pro`, `gemini/gemini-1.5-flash-8b`).  
- Calls to both LangChain and CrewAI pass model choice through LiteLLM.

---

## Requirement 8. Observability with Langfuse

**Change**  
- Use LiteLLM’s Langfuse integration to trace all LLM calls automatically.  
- Still attach LangChain and CrewAI callbacks, but LLM spans come directly from LiteLLM.

**Rationale**  
- Single source of truth for LLM metrics.  
- Easier trace alignment across orchestration paths.

---

## Requirement 9. Deployment and operations (Local only)

*(unchanged)*

---

## Eliminated requirements

- Evaluation and A/B comparison removed.  
- Security, privacy, and compliance removed.

---

## Implementation approach with rationale

- Use LiteLLM as the **common gateway** for Gemini and embeddings, reducing duplicate setup across LangChain and CrewAI.  
- Build ingestion pipeline with OCR, chunking, embeddings (via LiteLLM), and Chroma.  
- Implement LangChain path first, then CrewAI, both routed through LiteLLM.  
- Keep MCP minimal and constrained.  
- Keep everything local, reproducible with Docker and Poetry.

---

## Testing strategy outline

- **Unit**: chunking, OCR manifest, LiteLLM config loader.  
- **Integration**: ingestion produces embeddings via LiteLLM and persists to Chroma.  
- **E2E**: both LangChain and CrewAI call Gemini through LiteLLM and return cited answers.  
- **Observability**: Langfuse shows traces automatically from LiteLLM.  
- **MCP sanity**: browse returns Black Swan sources only.  
- **Persistence**: restart app and confirm Chroma and chat state.

---

## Success criteria

- Embeddings created with LiteLLM (`gemini/text-embedding-004`).  
- Both **LangChain** and **CrewAI** paths call Gemini via LiteLLM and return correct citations.  
- Streamlit UI lets user toggle between models (from LiteLLM config).  
- Langfuse shows LLM traces for both paths, sourced from LiteLLM.  
- Entire system runs locally via Poetry or Docker.

---

## Integration points with existing code

- Config module provides LiteLLM settings (model names, API key).  
- Retriever factory unchanged.  
- Answer functions (`answer`, `crew_answer`) both route model calls via LiteLLM.  
- Observability unified: LiteLLM → Langfuse, plus callbacks from LangChain/CrewAI.

---

## Dependency map (updated)

1. **Req 1** → Scaffolding and LiteLLM config.  
2. **Req 2** → Ingestion + OCR → embeddings via LiteLLM → Chroma.  
3. **Req 3** → Vector store init/retriever factory.  
4. **Req 4** → LangChain RAG via LiteLLM.  
5. **Req 5** → CrewAI path via LiteLLM.  
6. **Req 6** → Tiny MCP browse.  
7. **Req 7** → Streamlit UI (uses LiteLLM config).  
8. **Req 8** → Observability via LiteLLM → Langfuse.  
9. **Req 9** → Local run and ops.
