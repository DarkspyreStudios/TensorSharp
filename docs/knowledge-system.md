# Personal multimodal knowledge

TensorSharp.Knowledge stores original files and searchable, cited representations
of their contents. TensorAgent adds collection management, uploads, search, and
per-chat selection. The same store and retrieval contracts serve the standalone
TensorSharp server and agent tools.

The default is local BM25 text retrieval. Optional GGUF embeddings add semantic
retrieval; compatible loaded media models can derive text from images, audio, and
sampled video frames. No models are downloaded automatically. See
[research and design rationale](knowledge-system-research.md) for the primary
open-source references that informed these independently implemented patterns.

## Using the library in TensorAgent

1. Open **Menu → Knowledge**, name a collection, and select **Create**.
2. Browse that collection and choose **Upload documents or media**. The library
   upload is separate from a one-turn chat attachment. It retains the original
   after the upload cache or chat is cleared.
3. Read each document's extraction status and warnings. A batch can succeed for
   some files and fail for others; both outcomes are shown.
4. Search the collection to inspect excerpts and source locations. **Read
   surrounding context** expands a result within a bounded window. **Original
   source** retrieves the exact current source version.
5. Check the collections to use in this chat. Browsing a collection does not
   select it for inference. Selected collection chips appear above the composer;
   up to 32 collections can be selected. A new chat starts with no selection, and
   reopening a saved chat restores its selection.
6. Ask a question or start agent work. Relevant evidence is retrieved before the
   model responds. When a model supports tools, it can discover sources, refine
   its search, expand evidence, and open originals for analysis.

Assistant turns show **Retrieved sources** links derived from host-supplied
source identities. These links survive saved-chat restoration and stream replay;
they do not depend on the model reproducing a URL correctly. They identify the
evidence retrieved for the turn, not a guarantee that every statement in the
answer is supported. Source text and titles in the library are rendered as text.

At the end of a response, the host can also repair one narrow citation mistake:
a filename used where a document ID belongs in a local source URL. It substitutes
the ID only when that filename uniquely identifies a source already retrieved
for this request, in the URL's matching collection and version, and that version
is still current. A repair emits a complete-answer SSE `replace` frame followed
by `knowledgeCitationRepairs` with the repair count. Unknown or unretrieved links,
ambiguous filenames, other origins/collections, and stale versions remain
unchanged. This normalization neither retrieves new evidence nor proves generated
claims; the separate **Retrieved sources** links remain the authoritative source
identities supplied by the host.

Documents load in pages of 100 with **Load more documents**. Refresh reloads the
current catalog. **Reindex** processes the retained original again using the
currently available extractors and embedding provider. This is useful after
loading a media model or changing an embedding model. Removing a document or
collection requires a second inline confirmation; removing a selected collection
also removes it from the current chat's selection.

## Extraction and provenance

| Input | Searchable representation and location | Coverage |
| --- | --- | --- |
| Text, Markdown, code, logs, XML, YAML and related text formats | Text segments; line range and Markdown section when available | Strict text decoding with BOM detection; binary data is rejected |
| CSV/TSV | Header/value records, including quoted multiline fields; record row | Searchable records, not a database query engine |
| JSON arrays, JSONL/NDJSON | Array item or line records; row and JSON item label | Bounded parsing; other JSON values remain text |
| HTML | Text with script/style/noscript removed | No script execution, rendering, or remote resource loading |
| SRT/VTT | Subtitle cues with start/end times | Supplied subtitles only |
| PDF | Selectable text with page number | Partial: no scanned-page OCR, image interpretation, or layout coordinates |
| DOCX | Paragraphs, tables, referenced notes | Partial: no rendered pagination, embedded images, or page layout |
| PPTX | Slide text in presentation order | Partial: no image/chart interpretation, speaker notes, or rendered layout |
| XLSX | Worksheet rows, cell coordinates, cached values | Partial: no formula recalculation or chart/image interpretation |
| Images | Generated description and visible-text transcription using a loaded vision model/projector | Partial: model output can omit or misread details; no native image embeddings |
| Audio | Generated transcript using a loaded compatible audio tower or an explicitly configured transcription companion | Partial: no diarization, word timestamps, or non-speech event extraction |
| Video | Up to eight described frames, with sample timestamps | Partial: no soundtrack indexing or coverage of events between samples; variable-frame-rate times are approximate |
| Unsupported formats or unavailable media capabilities | Original file and searchable filename metadata | Metadata only; content has not been understood |

`Ready` means the selected extractor produced content without a declared coverage
warning. `Partial` means content was extracted with limitations. `MetadataOnly`
means only the filename is indexed; its metadata chunk must not be mistaken for
extracted content. Document records include extractor identity, warnings, content
hash, version, and embedding identity where applicable. Generated descriptions
and transcripts always remain marked partial.

The media adapter uses TensorSharp's existing model scheduler. Images are limited
to 16,777,216 pixels. Local audio accepts PCM WAV up to 120 seconds; other formats
need a compatible `TS_JEV_TRANSCRIPTION_URL` companion or conversion before
reindexing. The companion has its own format and duration limits. Video requires
an installed compatible codec and vision model, with limits of ten minutes,
36,000 frames, and 16,777,216 pixels per frame. Each local media generation has a
three-minute deadline, a 768-token output budget, and a 16,000-character cap.
File extensions alone do not establish codec or model support.

## Store lifecycle and retrieval

Each document is an atomic durable unit containing metadata, derived chunks, and
optional vectors. Original bytes are immutable SHA-256-addressed objects;
document manifests reference them. A single-writer lock protects the store.
Startup loads validated manifests, rebuilds the in-memory inverted index, and
cleans abandoned staging files and unreferenced originals. This is an embedded
personal store, not a distributed or multi-writer service.

Within a collection, the filename identifies an imported document. Uploading the
same filename and identical bytes is idempotent. Changed bytes replace that
document's derived content and advance its version. Reindexing also advances the
version. Failed extraction or embedding leaves the previous committed document
available. Deletion removes derived records and any original no longer referenced
by another document. Only the current version is retained: old citation reads
fail as stale instead of silently returning different content.

Chunks retain their source segment's location, modality, document ID, and
version. References have the form
`kb://{collectionId}/{documentId}@{version}/{chunkIndex}`. A location may describe
the whole segment containing a chunk rather than precise coordinates for each
character. Adjacent reads expand within the same document and version.

Collection filtering occurs before ranking. BM25 uses an inverted index and
Unicode-aware terms, including CJK n-grams. When embeddings are configured,
normalized exact cosine search considers only vectors with matching model
identity and dimensions. Reciprocal rank fusion combines the independent lexical
and dense rankings with deterministic tie breaking. Search scores are ranking
signals, not probabilities. Dense search scans compatible vectors in the selected
scope; there is no approximate nearest-neighbor index.

The GGUF provider owns an encoder separately from the chat model. Its vector-space
identity includes the weights' SHA-256, query/document prefixes, and aggregation
version. Input longer than the encoder's token window is split and combined by a
normalized, token-count-weighted mean. Changing model or prefixes does not mix
old and new vector spaces: incompatible documents retain lexical search and need
reindexing for semantic retrieval.

Default store limits are 64 MiB per file, 2 GiB of logical source storage, 100
collections, 10,000 documents, 2,000,000 extracted characters per document,
10,000 segments/chunks per document, and 100,000 total chunks. Chunk size is 1,200
characters with 120 characters of overlap. Search is bounded to 50 results,
32,000 returned text characters, and 4,000 query characters. SDK hosts can supply
`KnowledgeOptions` to set appropriate limits. Transient ingestion files and
derived indexes require additional disk and memory beyond source bytes.

## Inference and agent behavior

Automatic retrieval and knowledge-tool integration currently apply to the WebUI
`POST /api/chat` path used by TensorAgent and the standalone TensorSharp WebUI
adapter. OpenAI-compatible `/v1` endpoints and the Ollama adapter do not
automatically consume `knowledgeCollections` or add these knowledge tools. Other
applications can use the core SDK or `/api/knowledge` API and explicitly compose
retrieved evidence into their own inference requests.

`knowledgeCollections` in a WebUI chat request explicitly grants a retrieval scope
for that request. Missing or empty selection supplies no knowledge. Invalid or deleted
selections are rejected. The host retrieves up to six excerpts for the current
question, so tool-free models also receive relevant evidence. The host adds fixed
trust/citation guidance and presents source contents as untrusted reference data
in the current user message. Extracted instructions cannot change the allowed
collection set.

For models with tool support, the host provides:

| Tool | Behavior |
| --- | --- |
| `knowledge_list` | Discover selected sources in pages of up to 20 records |
| `knowledge_search` | Retrieve bounded, cited evidence within the selected scope |
| `knowledge_read` | Expand an exact reference with zero to two neighboring chunks on either side |
| `knowledge_open` | Retrieve the original's versioned source link and, when code tools have a private workspace, copy the complete original there for analysis |

The tools cannot modify the library or enlarge their scope. Subagents inherit the
same scope; original-file copies go into the invoking agent's private workspace.
This allows dataset calculations on complete originals rather than assuming that
the top retrieved rows represent a whole dataset. Retrieval response budgets
include serialized UTF-8 metadata and text, with an explicit truncation signal.

Fresh retrieval is rebuilt for each turn. Automatically inserted excerpts are not
stored as chat history. Source metadata is stored for citation display, but it
does not make an unselected collection accessible. A prior assistant answer is
still part of ordinary conversation history and may itself repeat source facts.

## Host setup and HTTP API

TensorAgent creates its library under `AgentPaths.DataRoot/knowledge` and exposes
management through its existing authenticated loopback server. The standalone
`TensorSharp.Server.Host` enables the library when `TS_KNOWLEDGE_DIRECTORY` names
a store directory. Both use the same store implementation. Collection IDs are
scope identifiers, not access credentials: a remote deployment must put all
management, search, source-download, and chat endpoints behind the application's
authentication and authorization boundary. The standalone library does not add a
multi-user permission model.

Optional embedding configuration, set before starting the host:

```sh
export TS_KNOWLEDGE_EMBEDDING_MODEL=/absolute/path/to/embedding-model.gguf
export TS_KNOWLEDGE_EMBEDDING_BACKEND=CPU
export TS_KNOWLEDGE_QUERY_PREFIX=''
export TS_KNOWLEDGE_DOCUMENT_PREFIX=''
```

Choose a supported sentence-embedding GGUF and use the prefixes required by that
model. Omit the embedding model variable for lexical retrieval. Media processing
uses the currently loaded compatible model; upload does not choose or download
one. A configured transcription companion is an external service and receives
the recording as part of the user's explicitly configured workflow.

Both hosts expose these routes below `/api/knowledge`:

| Method and path | Request / response |
| --- | --- |
| `GET /collections` | `{ collections: [...] }` |
| `POST /collections` | `{ name, description? }` → `{ collection }` |
| `DELETE /collections/{collection}` | Delete the collection and its sources |
| `GET /collections/{collection}/documents?offset=0&limit=100` | `{ documents, total, nextOffset }`; default 100, maximum 200 per page |
| `POST /collections/{collection}/documents` | Multipart upload → `{ documents }`; TensorAgent also returns per-file `errors` |
| `DELETE /collections/{collection}/documents/{document}` | Delete one source |
| `POST /collections/{collection}/documents/{document}/reindex` | `{ document }` with a new version |
| `GET /collections/{collection}/documents/{document}/source?version=1` | Original bytes as an attachment; stale version returns 409 |
| `POST /search` | `{ query, collectionIds, limit?, maxCharacters? }` → `{ results }` |
| `POST /read` | `{ reference, collectionIds, adjacentChunks?, maxCharacters? }` → `{ results }` |

TensorAgent accepts up to 16 files and 128 MiB per upload batch; per-file limits
still apply. The standalone endpoint accepts one file per request and validates
the multipart envelope before committing it. Source responses are downloads with
`nosniff` and `no-store`, preventing uploaded HTML from being served as active UI.
Ordinary argument/format errors use 400, missing sources use 404, stale references
use 409, and exceeded quotas use 413. Chat requests use `knowledgeCollections`;
search/read bodies use `collectionIds`.

## Embedding in a .NET application

The core project has no dependency on TensorAgent or a running inference model:

```csharp
using TensorSharp.Knowledge;

var options = new KnowledgeOptions();
using var store = new KnowledgeStore(
    "/absolute/path/to/library", new BuiltInKnowledgeExtractor(options), options: options);
var collection = await store.CreateCollectionAsync("Project research");
await using var input = File.OpenRead("notes.md");
var document = await store.ImportAsync(collection.Id, "notes.md", input, "text/markdown");
var hits = await store.SearchAsync("launch requirements", new KnowledgeSearchOptions
{
    CollectionIds = new[] { collection.Id }, Limit = 6, MaxCharacters = 8000
});
if (hits.Count > 0)
{
    var context = store.Read(hits[0].Reference, maxCharacters: 8000,
        adjacentChunks: 1, collectionIds: new[] { collection.Id });
}
```

Custom extractors implement `IKnowledgeExtractor` and return ordered
`KnowledgeSegment` values with provenance, status, warnings, and extractor ID.
Custom embedding providers implement `IKnowledgeEmbeddingProvider` with a stable
model identity, dimensions, and batched query/document embeddings. The core SDK's
`Read` method allows an omitted scope for trusted local callers; HTTP and agent
adapters always supply explicit scope. Applications are responsible for their
own caller authorization.

## Validation and current boundaries

Reusable tests live in `TensorSharp.Knowledge.Tests`, the knowledge tests in
`InferenceWeb.Tests`, and `TensorAgent/tests/TensorAgent.Tests`. They exercise
format extraction, provenance, durable lifecycle, stale references, model-space
compatibility, bounds, request scope, tool behavior, transport, saved-chat
metadata, and executable UI workflows. Optional live-model tests require an
explicitly configured model and must be reported as not run when unavailable.

The [knowledge benchmark](../eng/validation/KnowledgeBench/README.md) measures
deterministic corpus ingestion, warm retrieval, restart, memory, and a small
authored retrieval fixture. Optional real embedding runs are reported separately
from lexical runs. The [end-to-end probe](../eng/validation/knowledge/http_probe.py)
exercises the real host. The [full-CSV agent probe](../eng/validation/knowledge/agent_probe.py)
checks a real model opening the complete uploaded CSV with `knowledge_open`,
executing Python aggregation, and returning the expected result. It records the
actual tool trace and does not accept an unsupported model success claim as
passing. This is one synthetic calculation, not a general agent benchmark.
Generated logs, screenshots, and actual measurements
belong in ignored `docs/validation/` or `artifacts/`, not this design document.
Tiny media samples and synthetic queries establish only the scenarios exercised;
they do not demonstrate general document-understanding or RAG quality parity with
the researched systems.

Without an embedding provider, retrieval uses lexical matches. A broad request
such as “summarize this collection” can return no excerpts for a model without
tools; arbitrary source chunks are not inserted as a substitute. Ask about
specific terms, configure a supported embedding model, or use a tool-capable model
to list and open the selected sources for a complete review.

This implementation does not provide graph/temporal memory, connector
synchronization, ANN/vector-database scaling, learned reranking, universal OCR,
native joint image/audio/video embeddings, long-video understanding, background
ingestion jobs, or multi-tenant ACLs. The extractor and embedding boundaries allow
future providers without changing original-file ownership or source citations.
