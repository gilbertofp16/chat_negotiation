# DEVELOPMENT_PLAN.md

## Overview

Goal: build a local proof of concept negotiation coach chat application that uses an open source book as a knowledge source. The app supports two orchestration paths, LangChain and CrewAI, runs a Streamlit chat UI, indexes the book into Chroma for retrieval, calls Gemini as the model, adds a tiny MCP browse tool focused on Black Swan negotiation techniques, and sends traces to Langfuse for observability. This is for learning, not for head to head comparison or production deployment.

---

## Constraints and assumptions

- Package and env management: **Poetry**. Containerization: **Docker**. Do **not** use pip anywhere.
- Runs locally only. No cloud deployment, staging, or production targets.
- Primary model family: Gemini. Baseline `models/gemini-1.5-pro-latest` for reasoning, `models/text-embedding-004` for embeddings. Flash variants allowed for speed tests.
- Vector store: Chroma with local persistence (`data/chroma`).
- Source: one open licensed **PDF**. If it is a scan, we **must OCR** before indexing.
- Observability: Langfuse (cloud or local) for traces.
- MCP: minimal single purpose tool to browse for “Black Swan negotiation techniques” to sanity check answers, optional in normal flow.
-Cline is used to run and iterate. It does not own state.

---

## Architecture summary

- **Data path:** PDF ingestion → OCR if needed → text chunking → embeddings → Chroma (persisted).
- **Query path A (LangChain):** Chroma retriever → RAG prompt → Gemini chat → answer with page citations.
- **Query path B (CrewAI):** Retriever agent → Coach agent → Gemini chat → answer with page citations.
- **Browse tool:** Tiny MCP server exposing one safe “browse/search” action for Black Swan techniques only.
- **UI:** Streamlit chat with session memory, optional local history persistence.
- **Observability:** Langfuse traces for both paths.

---

## Requirement 1. Project scaffolding and environment (Completed)

**Tasks**
1. Initialize repo; create Poetry project; add Dockerfile and `.env.example`.
2. Define dependency groups: core, embeddings, PDF processing/OCR, vector store, UI, observability.
3. Centralize configuration (env loader) for API keys and paths.
4. Provide Make targets (or Poetry scripts): `ingest`, `run`, `trace-check`, `clean`.
5. Pre-commit with basic lint, fmt, and secret scan.

**Dependencies**  
None; this unlocks everything else.

**Approaches**  
- Poetry for locking and reproducibility.  
- Docker for consistent local runtime.  

**Trade-offs**  
- Poetry learning curve vs reproducibility.  
- Docker build time vs environment parity.

**Testing strategy**  
- Fresh-clone install test.  
- Container build succeeds and runs `streamlit` entry point.

**Integration**  
- One config module imported by UI, ingestion, LangChain, CrewAI, MCP, and Langfuse.

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

**Dependencies**  
- Requirement 1 complete; Google key available.

**Approaches**  
- OCR only when needed to reduce processing time.  
- Optional token-aware splitting in a later pass if retrieval quality is low.

**Trade-offs**  
- Larger chunks lower retrieval calls but risk dilution of relevance.  
- Smaller chunks improve precision but increase token usage.  
- OCR increases preprocessing time but is mandatory for scans.

**Testing strategy**  
- Spot-check OCR quality on random pages.  
- Ensure every chunk maps back to a page.  
- Retrieval smoke test: seed 5–10 questions and confirm top-k includes the right pages.

**Integration**  
- Provide an ingestion entry point callable from Streamlit to re-index a newly uploaded PDF.

---

## Requirement 3. Vector store management with Chroma (Completed)

**Tasks**
1. Define a stable collection name and persist directory (`data/chroma`).
2. Init routine: open collection if present, otherwise instruct user to run ingestion.
3. Documented backup and restore of `data/chroma` for local use.

**Dependencies**  
- Requirement 2 produces at least one collection.

**Approaches**  
- One collection per book (current scope).  
- Future-ready: allow multiple collections; out of scope for this POC.

**Trade-offs**  
- Single collection is simpler now; multi-collection adds complexity with little benefit for a single source.

**Testing strategy**  
- Restart-persistence test.  
- Measure baseline top-k retrieval latency.

**Integration**  
- Expose a retriever factory used by both LangChain and CrewAI.

---

## Requirement 4. Prompt Template Administration

**Tasks**
1.  Create `prompts/` directory with subdirectories for `langchain/` and `crewai/`.
2.  Create initial prompt files (e.g., `prompts/langchain/negotiation_coach.yaml`) with name, role, template, and metadata.
3.  Develop `src/prompt_loader.py` utility to load prompts, apply templating, and return framework-specific prompt objects.
4.  Integrate prompt loading into LangChain and CrewAI components.
5.  Configure active prompts via a setting (e.g., in `.env` or `config.py`).
6.  Add observability hooks to include prompt IDs in Langfuse traces.

**Dependencies**
-   Influences Requirements 5 (LangChain RAG) and 6 (CrewAI).

**Approach**
-   YAML/Jinja2 files for prompts, Python utility for loading, Git for versioning.

**Testing Strategy**
-   Unit test `prompt_loader`.
-   Smoke test prompt loading and configuration.
-   Regression tests with seed queries to detect prompt drift.

**Success Criteria**
-   Prompts are editable without touching Python code.
-   Both CrewAI and LangChain paths read from the same central registry.
-   Prompt versions are traceable in Git and visible in Langfuse metadata.
-   Easy experimentation by switching active prompts in one place.

---

## Requirement 5. RAG pipeline with LangChain

**Tasks**
1. Author a system prompt: “Negotiation coach, grounded on the book, cite page numbers.”  
2. Configure Chroma retriever (`k`, MMR optional).  
3. Compose a simple “stuff” combine chain for responses.  
4. Allow model selection (e.g., `models/gemini-2.5-pro` vs `models/gemini-1.5-flash-latest`) via UI setting.  
5. Add Langfuse callback to capture traces and metadata (model, k, timings).

**Dependencies**  
- Requirements 3 and 4 done; Langfuse configured.

**Approaches**  
- Start with “stuff.” If long answers degrade, evaluate “map-reduce.”

**Trade-offs**  
- “Stuff” is fast and simple; map-reduce scales but adds latency and complexity.

**Testing strategy**  
- Golden Q&A set covering core themes (e.g., BATNA, labeling, calibrated questions).  
- Verify citation formatting and page numbers.  
- Confirm traces appear in Langfuse.

**Integration**  
- Export a single `answer()` with `(question) -> {text, citations}` for the UI.

---

## Requirement 6. Agentic variant with CrewAI (for learning, not comparison)

This requirement implements the same RAG pipeline as Requirement 5, but using the CrewAI framework to explore an agent-based approach. The goal is to produce the same outcome—a coached answer based on the book—but orchestrated by a crew of agents instead of a LangChain chain.

**Tasks**
1.  Use an agent to perform the retrieval step from Chroma, returning concise excerpts annotated with page numbers.
2.  Define a **Coach agent** that synthesizes an answer strictly from retrieved excerpts and cites pages.
3.  Provide a crew composition helper that mirrors model selection options used in LangChain.
4.  Enable tracing so CrewAI runs appear in Langfuse.

**Dependencies**  
- Requirements 3 and 5 ready; model access available.

**Approaches**  
- Use LiteLLM for model routing to Gemini (or CrewAI’s native Gemini support if stable in your version).  
- Keep tool surface minimal: only the retriever (MCP browse remains separate).

**Trade-offs**  
- LiteLLM simplifies swapping models but adds a layer.  
- Native client reduces layers but may be less flexible.

**Testing strategy**  
- Reuse the same golden Q&A set as Requirement 5.  
- Sanity check latency and output formatting.  
- Trace visibility in Langfuse.

**Integration**  
- Export `crew_answer()` with the same signature as `answer()` so UI can toggle orchestration.

---

## Requirement 7. Tiny MCP browse tool (Black Swan sanity checks)

**Tasks**
1. Implement a **minimal MCP server** with a single action: `browse_bsw(topic|query)` that:  
   - Restricts queries to the Black Swan negotiation domain (short allowlist).  
   - Returns a normalized summary and top sources with titles and URLs.  
2. Register MCP in Cline and locally.  
3. Prompt policy: model may call MCP only to **sanity check** and enrich advice; book content remains the primary source.  
4. UI: show when browse was used and list 1–2 sources.

**Dependencies**  
- Base app operational; Cline available.

**Approaches**  
- Start with a simple HTTP fetch plus very light parsing; optional “search” endpoint if needed.  
- Strict allowlist to keep scope narrow.

**Trade-offs**  
- Existing MCP servers are faster to adopt but broader than needed.  
- Custom tiny server gives tight control and clarity for learning.

**Testing strategy**  
- Manual test with queries like “calibrated questions” and “accusation audit.”  
- Check outputs are bounded, sanitized, and clearly marked as external.

**Integration**  
- Single adapter so both LangChain and CrewAI can invoke MCP through a common interface.

---

## Requirement 8. Streamlit chat UI with memory and caching

**Tasks**
1. Chat layout rendering history and latest answer with page citations.  
2. Upload widget to load a new PDF and trigger ingestion.  
3. Session memory for chat; optional simple local persistence for history.  
4. Settings panel: switch LangChain vs CrewAI and choose model.  
5. Cache stable resources (Chroma client, embeddings config).

**Dependencies**  
- Requirements 5 and 6 provide answer functions; Requirement 2 exposes ingestion callable.

**Approaches**  
- Per-session memory by default; simple file-based persistence optional.  

**Trade-offs**  
- Persisted history is convenient but adds file locking considerations in Docker.  
- Session memory is simplest for this local POC.

**Testing strategy**  
- Smoke tests for chat flow, ingestion, toggles, and citation rendering.  
- Refresh behavior with cached resources.

**Integration**  
- UI calls either `answer()` or `crew_answer()` based on the toggle; same response shape.

---

## Requirement 9. Observability with Langfuse

**Tasks**
1. Initialize Langfuse early and verify connection.  
2. Attach callback/middleware in both paths.  
3. Standardize trace metadata: session id, path (“langchain” or “crewai”), model, k, timing.  
4. Create a saved view in Langfuse for “RAG runs.”

**Dependencies**  
- At least one end-to-end query path working.

**Approaches**  
- LangChain native callback handler; CrewAI via recommended integration or OTel bridge.

**Trade-offs**  
- Native handlers are easy; OTel offers flexibility at cost of setup.

**Testing strategy**  
- One end-to-end run shows a proper tree (retrieval → LLM) with timings.  
- Verify sensitive content is not logged beyond necessity.

**Integration**  
- Include trace IDs in local logs for fast navigation during dev.

---

## Requirement 10. Deployment and operations (Local only)

**Tasks**
1. **Local POC only**: run via Poetry or Docker.  
2. Provide `docker compose` or Make targets for one-command bring-up.  
3. Confirm Chroma persistence under a mounted local volume.  
4. Document local health checks and restart steps.

**Dependencies**  
- App is stable and ingest works.

**Approaches**  
- Single-container approach for UI and app; Chroma embedded.  

**Trade-offs**  
- None for local POC; simplicity is the priority.

**Testing strategy**  
- Cold start and smoke query.  
- Restart and persistence verification.

**Integration**  
- Show version string and active model/path in UI footer.

---

## Eliminated requirements (per user direction)

- Prior **Evaluation and A/B comparison** requirement removed.  
- Prior **Security, privacy, and compliance** requirement removed.  

---

## Implementation approach with rationale

- Build ingestion and vector store first to establish the knowledge base.  
- Implement LangChain RAG as the baseline path for clarity.  
- Add CrewAI path to learn agentic orchestration, **not** to formally compare.  
- Integrate Langfuse early for visibility into retrieval and LLM behavior.  
- Add a **tiny, constrained** MCP browse tool specifically for Black Swan concepts to sanity check outputs without distracting from book grounding.  
- Keep the Streamlit UI simple with a path toggle and model selector to learn how each choice behaves locally.  
- Keep everything local in Docker or Poetry for repeatability.

---

## Testing strategy outline

- **Unit**: chunking parameters, citation formatter, config loader.  
- **Integration**: ingestion produces a persisted Chroma collection; retriever returns expected passages on seed questions.  
- **E2E**: ask 5–10 core negotiation questions; verify answers cite correct page ranges; confirm Langfuse traces.  
- **OCR QA**: random-page OCR quality check and fallbacks for low-confidence pages.  
- **MCP sanity**: browse call restricted to Black Swan topics returns bounded summaries and sources.  
- **Persistence**: restart app and confirm Chroma and optional chat history survive.  

---

## Success criteria

- Can ingest the target PDF (including OCR when needed) and persist to Chroma locally.  
- Streamlit chat answers grounded in the book with correct page citations for at least 9 out of 10 seed questions.  
- Both **LangChain** and **CrewAI** paths function using Gemini and return similarly formatted outputs.  
- Tiny MCP browse tool works and clearly indicates when it was used, returning a short summary and 1–2 sources about Black Swan techniques.  
- Langfuse shows complete traces for both paths with key metadata (model, k, timings).  
- Entire system runs locally via Poetry or Docker with a single command.

---

## Integration points with existing code

- Unified config module for env and settings.  
- Retriever factory returning a Chroma-based retriever.  
- Two answer functions (`answer`, `crew_answer`) with the same signature and response shape.  
- Shared tracing helper that attaches the proper handler for the active path.  
- Optional local history persistence interface that the UI can toggle without code changes.

---

## Dependency map (updated)

1. **Req 1** → Scaffolding and config.  
2. **Req 2** → Ingestion + OCR → produces indexed collection.  
3. **Req 3** → Vector store init/retriever factory.  
4. **Req 4** → Prompt Template Administration.
5. **Req 5** → LangChain RAG (depends on 3, 4).  
6. **Req 6** → CrewAI path (depends on 3, 5).  
7. **Req 7** → Tiny MCP browse (depends on core app; optional).  
8. **Req 8** → Streamlit UI (depends on 5 and 6; calls ingestion).  
9. **Req 9** → Observability with Langfuse (depends on any working path).  
10. **Req 10** → Deployment and operations (Local only) (depends on stable app).
