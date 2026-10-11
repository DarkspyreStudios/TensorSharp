# Multimodal knowledge system: research and design rationale

Research accessed **2026-10-10** (America/Los_Angeles). This is a design reference,
not a benchmark report or a claim that TensorSharp implements every capability
described below. The starting discussion was [the shared knowledge-base
discussion](https://chatgpt.com/share/6aca8594-5ce4-83e8-b268-baf0a14a5cab).
The shared page is not a primary technical specification; the project-owned
documentation and source linked here provide the implementation guidance.

## Lessons from the referenced projects

| Project and primary source | Observed pattern | TensorSharp / TensorAgent design implication |
| --- | --- | --- |
| [Open WebUI knowledge tools](https://docs.openwebui.com/features/workspace/knowledge/) | Separate collection discovery, file-name lookup, exact content search, semantic retrieval, and bounded file reads. Attached knowledge scopes agent tools. Directory synchronization compares content hashes. | Expose small, composable knowledge tools. Discover a collection, search within it, then read the cited chunk or document. Do not insert the entire library into every prompt. Scope must be enforced by application code. |
| [LlamaIndex nodes](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_nodes/) and [auto-merging retriever](https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/) | Chunks retain document metadata and relationships. Auto-merging can replace related retrieved children with their parent context. | Keep stable source and chunk identities, chunk order, section paths, and source locations. Allow bounded adjacent or parent-context expansion without conflating the retrieval chunk with the source document. |
| [RAGFlow](https://github.com/infiniflow/ragflow) | Parsing quality, format-dependent chunking, visible references, and combined retrieval are first-class parts of the system. | Extraction diagnostics and citations belong in the ingestion/retrieval contract. A file that uploaded successfully is not necessarily a file whose contents were successfully indexed. |
| [Onyx connectors](https://docs.onyx.app/overview/core_features/connectors) and [project overview](https://github.com/onyx-dot-app/onyx) | Ingestion carries metadata and permissions into a searchable internal representation. Document indexing is a substantial service, separate from a lightweight chat interface. | Share a knowledge service across hosts and clients. Apply collection/document scope before search, listing, reads, and mutation. Local collection selection alone is not a multi-user authorization system. |
| [Dify retrieval service](https://github.com/langgenius/dify/blob/main/api/core/rag/datasource/retrieval_service.py) | Retrieval supports bounded results, document/metadata filtering, optional reranking, score thresholds, and duplicate removal using provider/document identity. | Make search options explicit, deduplicate by stable identity, and provide component scores/citations so retrieval behavior can be tested independently of generated answers. |
| [Docling document model](https://docling-project.github.io/docling/concepts/docling_document/) | One representation preserves text, tables, pictures, hierarchy, reading order, provenance, and layout where available. | Extractors should return ordered typed content segments with optional page, row, slide, bounding-box, and time provenance. Markdown/plain text is a searchable representation, not a replacement for the original asset. |
| [MinerU](https://github.com/opendatalab/MinerU) and [output contract](https://github.com/opendatalab/MinerU/blob/master/docs/en/reference/output_files.md) | Parsing results support structured content and stable page/block locations. Different parsing tiers trade speed and extraction quality. | Preserve source locators and record the extractor/version used. Select expensive OCR/layout work explicitly and report unavailable capabilities. Avoid promising equivalent quality from a basic text extractor. |
| [MarkItDown](https://github.com/microsoft/markitdown) | Format converters normalize documents to Markdown, with optional dependencies and plugins for additional capabilities. Image descriptions and richer audio/video processing depend on configured services/models. | Use an extractor interface and explicit capability detection. Keep the dependency-light .NET path usable; add native or external adapters without coupling the knowledge store to a Python service or requiring network access. |
| [VideoRAG](https://github.com/HKUDS/VideoRAG) | The architecture combines visual and audio evidence, temporal context, and graph-oriented indexing for long videos. | Represent video evidence as time-bounded transcript/frame segments linked to the original video. Transcript import alone does not establish visual understanding or long-video quality parity. |
| [LightRAG](https://github.com/HKUDS/LightRAG) and [lifecycle operations](https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md) | Incremental insertion/deletion must maintain derived chunks, vectors, and graph provenance. Indexing and querying must use a compatible embedding model; changing it requires re-embedding. | Store embedding identity alongside vectors and reject incompatible spaces even when dimensions match. Replace/delete all derived artifacts together; graph layers, if added, must preserve evidence ownership. |
| [Graphiti](https://github.com/getzep/graphiti) | Facts have temporal validity and source episodes. Retrieval combines semantic, lexical, and graph information. | Preserve creation/update time and source lineage now. A future temporal-memory layer should retain contradictory/superseded evidence explicitly; a flat document index must not claim to implement temporal reasoning. |
| [Pathway document store](https://pathway.com/developers/api-docs/pathway-xpacks-llm/document_store) and [connector semantics](https://pathway.com/developers/user-guide/connect/connectors-overview/) | Parsing, splitting, metadata, and retrieval are separate stages. Streaming changes include additions and retractions, and a replacement is a deletion plus addition within one update. | Stable identity, content hashing, atomic replacement, and removal of obsolete chunks are prerequisites for future synchronization. Do not append duplicate vectors when a source is updated. |
| [Khoj query filters](https://docs.khoj.dev/miscellaneous/query-filters/) | Natural-language search can be constrained by file, terms, and dates. | Provide structured filters alongside query text. A bounded exact read is useful after semantic discovery; identifiers and code should remain searchable without an embedding model. |
| [Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/) and [hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/) | Vector spaces specify dimensions and metrics; named vectors distinguish representations. Hybrid retrieval fuses separate candidate rankings. | Persist a vector-space descriptor including model identity, dimension, and normalization/metric. Keep lexical and vector scores separate, and combine ranks rather than directly adding incompatible score scales. |
| [LanceDB hybrid search](https://docs.lancedb.com/search/hybrid-search) | Full-text and vector retrieval can be fused with reciprocal rank fusion; explicit limits and prefilters control the candidate set. | Use bounded candidate pools, deterministic tie breaking, and scope filtering before ranking. An embedded exact-search baseline can serve small local libraries while preserving a future indexed-storage boundary. |

These are independently implemented architectural patterns. No source code from
these projects is copied, and this research does not introduce them as runtime
dependencies. Product documentation and default-branch source may change after
the access date; no upstream benchmark numbers are used as TensorSharp results.

## Proposed system boundaries

1. **Source assets and ingestion.** A collection contains stable document records.
   Retain original bytes, content hash, display name, media type, ingestion status,
   and extractor diagnostics. An extractor returns ordered segments and provenance;
   chunking and embedding are independent stages. Limits apply to bytes read,
   decompressed content, segment count, chunk size, and provider work. Failed or
   cancelled extraction must not publish a partially replaced searchable document.
2. **Persistent knowledge index.** Publish document metadata, chunks, lexical
   information, and optional vectors together. Reopening the service must preserve
   search results. Duplicate input must not multiply chunks. Reindexing a document
   must remove obsolete chunks, and deletion must remove the source and derived
   records without affecting unrelated collections.
3. **Retrieval.** A lexical path works without a loaded embedding model. An optional
   embedding provider supplies vectors in one explicitly identified space. Rank
   fusion combines independently retrieved lexical and dense candidates; scores
   are relevance signals, not probabilities. Enforce collection/document filters,
   result count, and returned-text budgets before returning evidence to the caller.
4. **Inference and agent integration.** Users attach selected collections to a
   conversation or agent run. Search returns short excerpts and stable citations;
   read tools expand only the relevant evidence. Retrieved text is untrusted source
   material, never an instruction to execute tools or change scope. Existing chat
   without selected knowledge should continue to work.
5. **TensorAgent management.** The library surface should expose collection creation
   and selection, upload, processing status, retrieval preview, source/chunk reading,
   reindexing, and deletion. It should state whether a file is fully extracted,
   represented by supplied text, or awaiting an extractor/model.

The .NET service owns the data contract and lifecycle. Native inference remains a
provider behind that contract. OCR, transcription, captioning, and joint multimodal
embeddings are different capabilities: accepting a media file is not evidence
that all four ran successfully. Supplied captions/transcripts are useful evidence,
but must remain distinguishable from model-extracted observations.

## Identity, scope, and performance decisions

- **Identity:** source identity, content identity, chunk identity, and embedding
  identity are separate. A rename need not alter content; identical text in two
  user-authorized collections must retain independent ownership. A parser change
  can require reindexing even if source bytes did not change.
- **Scope:** selected collection IDs provide retrieval scope. Hosted multi-user
  deployments additionally need authenticated ownership/ACL checks; an arbitrary
  client-supplied user ID or collection ID is not authorization. Every read and
  mutation needs the same policy as search.
- **Bounds:** cap ingestion and response sizes, use cancellation, and batch
  embeddings. Progressive search/read reduces prompt usage, but it does not
  justify unbounded internal scans or returning every chunk.
- **Storage:** a portable local store and exact vector search are a reasonable
  measurable baseline. They do not imply million-document scalability. Keep store
  and provider boundaries suitable for an inverted index, ANN backend, or server
  implementation when corpus measurements justify one.
- **Compatibility:** vector dimensionality is necessary but insufficient.
  Identical dimensions from different models or preprocessing settings do not
  produce a shared semantic space. Incompatible vectors must not be silently
  compared, and reindexing must have an explicit outcome.
- **Multimodal evidence:** preserve modality and source position. Time ranges
  identify audio/video moments; page/slide/row ranges identify document locations.
  A generated caption is a derived interpretation and should not be presented as
  verbatim source text.

## Validation required for implementation claims

| Area | Required evidence |
| --- | --- |
| End-to-end lifecycle | Create collection → upload → extract/index → search → read citation → reopen store → reindex → delete; verify unrelated content remains available. |
| Grounding | Known queries resolve to expected document and source location; returned citations/read results agree with stored source text. |
| Scope | Search, listing, direct reads, and mutation cannot escape the allowed collection set; empty selection semantics are explicit. |
| Multimodal ingestion | Separate coverage for text/structured documents, images, audio, and video. Fake extractors validate orchestration only; model-backed extraction requires actual media/model runs. |
| Vector compatibility | Reject wrong dimension, non-finite values, and mismatched model identity; cover lexical fallback and reindexing after a model change. |
| Resource limits | Reject oversized inputs; cover cancellation, malformed containers, decompression bounds, empty documents, interrupted persistence, and repeated updates. |
| Search quality | Report recall or expected-source hit rate on a fixed labeled corpus, including lexical identifiers, paraphrases, multilingual text, and negatives where supported. |
| Performance | Record corpus/chunk count, text volume, embedding dimensions/provider, hardware/runtime, warm-up, ingestion rate, search p50/p95, and memory. Separate extraction/model cost from index lookup cost. |

Benchmarks using generated text, supplied transcripts, or deterministic fake
embeddings measure the exercised storage/retrieval/orchestration path. They do not
establish OCR accuracy, speech recognition quality, visual/video understanding,
real-model semantic recall, generation quality, or mobile-device performance.
Unavailable model/device scenarios must be reported as unvalidated, not passing.
Graph construction, temporal reasoning, external connector synchronization, and
distributed indexing need their own implementation and evaluation before being
advertised.

Reusable validation programs belong in `eng/`; required fixtures belong in the
test project. Generated measurements and logs belong in ignored
`docs/validation/` or `artifacts/`. This managed knowledge feature does not require
ggml changes. Any subsequent native work must record the unchanged upstream
dependency revision and actual device/model coverage.
