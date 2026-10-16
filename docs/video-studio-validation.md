# Video Studio validation and capability matrix

This records implementation-branch validation on 2026-10-15, not a production
readiness certification. See [design and reference audit](video-studio-design.md)
for source commits, licensing, architecture and dependency boundaries, and the
[user guide](video-studio.md) for operation.

## Environment and dependency revision

- macOS 27.0, Apple Silicon arm64, 48 GiB unified memory.
- .NET SDK 10.0.302 / runtime 10.0.10; Apple MAUI workload 10.0.20.
- Operator-installed Homebrew FFmpeg/ffprobe 9.0.2. The tested build enables GPL,
  version 3 components and libx264; Studio uses available H.264/AAC encoders.
- Upstream `ExternalProjects/ggml`:
  `a05c9922613c6a2230dbb1d0de679519ecafb4f0`, unchanged working tree. No ggml patch
  or native kernel change was introduced for Studio.
- Implementation baseline: TensorSharp `main`
  `6bdc5c7b799fdc63e061b8dbd3ec9a64bb46cf48`; branch `codex/video-studio`.

Generated logs, screenshots, reports, projects and sample exports are ignored
under `artifacts/studio/`. They are not source-controlled fixtures. Reusable
drivers are in `eng/studio/`; small colors/tones are generated deterministically
with FFmpeg. No model downloads or remote services are part of these drivers.

## Recorded validation

| Check | Actual result and scope |
|---|---|
| Shared core tests | 53 passed, 0 failed, 0 skipped. Real FFmpeg output tests include pixels, caption timing, audio RMS/mixing/fades/dissolve, 30000/1001 timing, true variable-rate source trimming, decode, captured revisions, packages and derivatives. Generation lifecycle tests explicitly use doubles; missing-tool/mobile tests prove inference is never called. Startup cleanup and package disk-reserve rejection are covered. |
| Host regression subset | 187 passed, 0 skipped: Studio tools/routes plus existing video turns, host, upload media and loopback range/auth tests. Additional host/localization subset: 88 passed. Model routing tests use doubles. |
| Full TensorAgent test project | At the full-suite checkpoint: 1702 total, 1588 passed, 2 failed, 112 skipped in about 2 min 8 s. Later loopback security cases passed in the focused suite below. The two failures are existing, unchanged bundled-skill description assertions: ResearchSkillTests requires a description longer than 200 characters; MarketDataSkillTests expects “gainers”. Two Studio integration regressions found by the first run were fixed before this result. The full project is not reported as passing. |
| Shared UI tests | 13 passed using a simulated DOM, including bounded restored messages, command contracts, immediate history restoration and all eight locale catalogs. These are not native-device tests. |
| Final focused host/security suite | 200 passed, 0 skipped, including IPv4/IPv6/mapped-loopback, remote peer, untrusted Host, explicit remote opt-in, image-turn routing and localization checks. |
| Server and TensorAgent builds | Server.Host, TensorAgent.Core and the real TensorAgentHost harness build successfully. |
| Mac Catalyst build | Final `net10.0-maccatalyst`, arm64 Debug: exit 0, 64 warnings, 0 errors; build only, no launched MAUI application claim. Final server build: 5 warnings/0 errors; TensorAgent harness: 11 warnings/0 errors. Existing warnings remain visible in build logs. |
| Running WebUX browser | Created project, imported real files, added clips on two video tracks, edited placement/trim/fade/caption, rendered playable MP4 and reopened revision 8. Browser video reached readyState 4, 320×180, 2 s. |
| Running TensorAgent shared page | Real AgentAppHost loopback page: imported/added/trimmed/undid, pointer-dragged 24 frames and undid, rendered playable MP4, reopened saved state after process restart. Linked saved conversation appeared immediately without another send or revision change. Desktop and 390×844 browser viewport screenshots; viewport simulation is not an iPhone test. |
| HTTP vertical slice, both hosts | Existing chat upload → Studio import → multiple tracks → atomic proposal/stale rejection/undo/redo → captured-revision render → range response → complete probe/decode. Server and Agent projects re-rendered after actual host restarts. |
| Native Server Studio generation | Real LTX-2.5 Distilled Q4_K_M, MLX, seed 42, 512×512, 17 frames at 24 fps, audio: completed, managed asset/provenance saved, inserted alongside imported clips, exported 4.708333 s composition. Admission-to-completion about 125.6 s, including identity hashing and cold setup. |
| Native TensorAgent Studio generation | Same actual model/settings through AgentAppHost: completed and exported 4.708333 s composition; about 114.0 s. External outbound networking denied using macOS sandbox-exec; localhost allowed. Hash/provenance, thumbnails, waveform, package round trip, scoped media and CSRF checks passed. |
| Native cancellation | TensorAgent LTX job cancelled after entering denoise: running → cancelling → cancelled, no new asset, following render completed. This validates this short MLX workload, not every model/backend or a mid-generation resume checkpoint. |
| Real conversational planner, both hosts | Gemma-4-E4B-it-Q8_0 on ggml_metal inspected the selected clip and produced a validated atomic trim-to-24-frames + volume-0.25 proposal. Explicit apply, one-step undo, manual movement and stale proposal rejection passed through both real HTTP hosts. No inference doubles. |
| Final restart/export/security smoke | Both original projects containing real generated footage reopened and exported again after the final VFR/recovery/security build, each producing 113 decoded video frames. Both fresh HTTP vertical slices passed. Actual server spoofed-Host and cross-origin requests to `/api/studio` and Studio-bound `/api/chat` returned 403. |

The two real generated clips were 512×512, 17 decoded frames, 0.708333 s, H.264
with audio, 187858 bytes, SHA-256
`57e6af40bd12f7195acf1bfcc258f732b0d0312340104a4dd260dd2a3bcfc561`.
The model file SHA-256 is
`0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0`.
This is a very short functional sample, not a long-duration quality benchmark,
character-consistency evaluation, throughput comparison or VRAM benchmark.

The first actual planner attempt produced nested operation-name JSON and failed
strict validation without modifying the project. Tool documentation and repair
feedback were clarified with an explicit flat command example. The same natural
instruction then passed on both hosts. This records the observed failure and
correction; it is not a claim that every text model reliably follows the schema.
An ordinary browser request initially asked for a clip ID. A trusted, Studio-only
workflow instruction now directs the existing agent to inspect the selected
project first; the browser request “Lower the selected clip volume to 0.5” then
produced a valid proposal without user-supplied IDs or tool names.

The 112 full-suite skips were explicit: 54 need an unstaged embedded CPython
runtime; 16 need network opt-in (nine also need Python); 33 need text/multimodal
model opt-in; six need image/video weights; three are platform-restricted. These
are not passes. The separately recorded real Studio LTX/Gemma runs do not turn
those other skipped scenarios into successful coverage.

Output checking includes full decoding, expected dimensions, at most one-frame
duration/count tolerance, red/blue pixel discrimination and nonzero PCM audio.
Core tests additionally check caption-on/off pixels and relative audio amplitude.
Rendered preview and downloaded export are the same captured-revision output;
there is no separate browser composition engine to claim parity with.

A true variable-rate fixture (four source frames at 0, 0.4, 1.0 and 2.0 s)
initially exposed an incorrect trim: the held frame covering an in-point could
be discarded. Sampling source presentation timestamps onto the project clock
before integer-frame trimming fixed it. The resulting 48-frame export now checks
red/blue/green boundaries and audio onset within one frame. The metadata VFR flag
is still a rate-comparison heuristic, not exhaustive classification of every
container's timestamp behavior.

### Asset deletion follow-up

`StudioHostTests` now includes the same deletion scenario over the actual
ASP.NET Core WebUX routes and authenticated TensorAgent loopback routes. The
focused run passed **32 tests, 0 failed, 0 skipped**. It verifies the default
`asset_in_use` refusal, explicit removal of affected timeline clips, unchanged
linked clips using other assets, stale revisions, cross-project asset rejection,
the required request header, retry idempotency, undo/redo and retained media bytes.
Deleted assets are unavailable through the current asset URL until undo restores
them. Fixtures use seeded PNG metadata with media processes disabled; this test
does not claim real generation, rendering or native-device coverage.
A broader focused run including `LocalizationTests` and `WebUiRoutesTests`
passed **135 tests, 0 failed, 0 skipped** after the shared UI/catalog changes.

```sh
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter FullyQualifiedName~StudioHostTests --nologo -v:q
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter 'FullyQualifiedName~StudioHostTests|FullyQualifiedName~LocalizationTests|FullyQualifiedName~WebUiRoutesTests' --nologo -v:q
```

Logs are ignored at `artifacts/studio/delete-asset-host-tests.log` and
`artifacts/studio/delete-asset-focused-host-tests.log`. The Server project reference
is test-only; TensorAgent's application dependency graph is unchanged.

The deletion follow-up also passed **67 core tests** (14 new cases) and **24
shared UI tests**, with no skips. Core coverage includes durable targeted asset
undo, legacy history, later imports, atomic rollback, missing media, package
exclusion and captured jobs. A real FFmpeg render completes after its captured
source is removed from the current library; the conditioning-generation lifecycle
test uses an explicit adapter double, not real model inference.

Running browser checks on both hosts confirmed the two-clip usage warning,
deletion, Undo and Redo; WebUX also verified Cancel. After restarting both hosts,
the deleted state persisted, Undo restored the asset and both clips, and each
restored project exported a fully decoded 320×180, 48-frame MP4. Screenshots,
sample exports and the token-free report are in
`artifacts/studio/delete-asset-browser/`. These are desktop browser checks against
the actual hosts, not native-device tests.

Final Server.Host and TensorAgentHost builds passed (5 and 11 warnings,
respectively). The additional Mac Catalyst build was **blocked**: the installed
.NET Mac Catalyst 26.5.10315 workload requires Xcode 26.6, while the selected Xcode
is 27.0. This follow-up does not claim a successful native application build or
launch. No toolchain settings were changed to bypass the compatibility check.

```sh
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj --no-restore --logger 'console;verbosity=normal'
node --test eng/validation/tests/studio-ui.test.mjs
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj --nologo -v:q
dotnet build eng/validation/TensorAgentHost/TensorAgentHost.csproj --nologo -v:q
# Attempted; blocked by the Xcode/workload mismatch described above:
dotnet build TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj -f net10.0-maccatalyst -p:TensorSharpAppleTargets=true --nologo -v:q
```

Additional ignored evidence uses the `artifacts/studio/delete-asset-` prefix:
`core-tests.log`, `ui-tests.log`, `server-build.log`, `agent-build.log`, and
`maccatalyst-build.log`.

## Feature and platform status

| Capability | Implemented | Validation / limits |
|---|---|---|
| One shared project/command/render format | Both hosts | Core and HTTP tests; one exclusive service writer per storage root. TensorAgent token authentication; Server Studio loopback by default, explicit protected remote deployment opt-in. Not a multi-user authorization database. |
| Library, storyboard, multiple video/audio tracks | Both shared pages | Browser checks, simulated interaction tests, core tests. Generated and imported assets use the same clips. |
| Placement, drag, exact trim, split, duplicate, remove, snapping/zoom/scrub | Shared page/commands | Core and browser subset; not every touch gesture tested on hardware. Source media survives deletion/undo. |
| Delete library assets | Shared page/command in both hosts | Warns about timeline use; confirmed deletion removes that asset's clips atomically. Both HTTP transports verify revision checks, isolation, undo/redo and retained media. Does not reclaim disk space. |
| Linked edits, volume/fades/dissolve, contain/cover, track mute/visibility | Shared commands/render | Automated command/render tests. Dissolve requires an existing overlap and changes no clip positions. Embedded clip audio travels with its video. |
| Captions | Manual timed editable captions | In-house portable uppercase Basic Latin bitmap glyphs; unsupported Unicode refused. No automatic transcription/word-based cutting adapter. |
| Preview/export | Desktop/server FFmpeg render job; H.264/AAC MP4 | Real macOS rendering and full decode in both hosts. Browser/WebView must support the codec. Live composited preview and proxy editing deferred. |
| Versioned persistence, autosave, undo/redo | Shared core | Atomic writes, revision conflicts, idempotency, reopen/restart, hash relink, package limits. Schema 1 is first format; future schema rejected. No invented historic migration. |
| Agent edits | Existing runtime, scoped tools, reviewed atomic proposals | Structured-tool tests plus actual Gemma planner on both hosts; manual continuation and stale rejection use same command service. TensorAgent uses its saved-conversation recorder; server stores bounded Studio history in the current browser, not a new durable server chat store. |
| Generate alternate takes | Existing native video pipeline | Real LTX text-to-video/audio in both hosts. Original take remains an asset; replacement preserves placement. Side-by-side comparison viewer deferred. |
| Generation admission | Model capabilities plus complete local output pipeline | Missing ffprobe, FFmpeg, H.264/AAC or disabled subprocess platform marks model unavailable with a reason and refuses before adapter validation/inference. |
| Generation/render jobs | Durable bounded queue, cancellation, restart interruption | Doubles for deterministic races; native MLX cancellation separately validated. Interrupted stages retry from the beginning. |
| Import, thumbnails, waveforms | Async probe, bounded cached 320×180 thumbnail and 512 peaks | Real FFmpeg checks. These operations are cancellable request work, not durable queued jobs. |
| Offline operation | No cloud account/upload path | Manual editing without model checked; native TensorAgent run with external networking denied. Installed local tools/models required. |
| Responsive layouts/accessibility/localization | Shared page, semantic controls, keyboard editing, focus trap, localized principal labels | Desktop browser and mobile viewport; advanced/error text retains English fallback. Screen-reader and native touch validation outstanding. |
| Windows/Linux desktop | Same managed implementation and executable discovery | Not built/run on Windows or Linux for this task. Encoder/runtime availability remains platform-specific. |
| iOS/Android | Project-package review/edit UI and capability gating | No desktop subprocess import/export; mobile generation disabled by Studio host policy. No physical device or mobile native-media export validation. |
| Explicit remote host | Uses the host serving the current interface | No automatic remote fallback, remote media upload or paid provider added. A separate TensorAgent remote Studio transport is deferred. |

### Native model inventory

These are all current implementations of `IVideoGenerationModel`. Controls use
the loaded checkpoint's capabilities, not just the family name.

| Family | Exposed Studio operations | Validation in this task |
|---|---|---|
| Wan (2.1 / 2.2 variants) | T2V where permitted; first image for applicable I2V/TI2V checkpoints, 4n+1 temporal grid, variant sampling defaults | Adapter compiled and admission covered; no real Wan Studio inference. |
| MiniMax-H3 partitions | T2V, first/last where partition permits, identity images only for reference partition, available audio output, 17n+5 frames at 24 fps | Adapter compiled; no real H3 inference. H3 model license has separate territory restrictions; no license changes or downloads authorized. |
| LTX 2.5 Distilled | T2V, first/last, temporal image guides; identity images only with installed reference LoRA; optional audio output; 8n+1 frames and fixed distilled recipe | Real T2V/audio at 512×512×17 on MLX in both hosts. I2V, first/last, keyframes and identity references not generated through Studio in this run. |

Audio input, video reference conditioning, arbitrary LoRA selection, native
retake/extension and masked generative edits are not exposed by this Studio
adapter. Existing model-specific APIs remain separate. Spatial/temporal grids,
fixed steps/CFG, reference counts and incompatible options are rejected before
expensive work. Unsupported options are not silently ignored.

## Reproduce

### Chat/Studio generation parity

The generation-quality regression was caused by Studio's fixed 512×512 default
overriding the LTX preset's 1024×1024 output. Both interfaces already called the
same native model and encoder. Studio now resolves its omitted settings from
the same host defaults as chat and exposes them to the generation form. Reused
takes keep their recorded settings; **Use model defaults** upgrades an old take
before regeneration.

Full-resolution validation also exposed excessive LTX decoder memory overlap.
TensorSharp releases denoiser device residency before decoding and releases
completed convolutional residual intermediates earlier. MLX materializes the
activation and residual boundaries so lazy graphs do not retain completed
layers. This preserves the decoder's convolution math and full clip geometry;
it does not introduce temporal tiling or lower the resolution. Numerical checks
include untraced CPU/MLX decoder oracles and denoiser release/reload consistency.

For a real fixed-seed comparison, start a host with the desired model/config and
run the following probe. It imports the same original image into both surfaces,
records the requests and effective settings, and compares every decoded frame.
The report distinguishes media equality from a subjective anatomy review; a
shorter preview does not validate a six-second generation.

```sh
python3 eng/studio/validate_generation_parity.py \
  --url http://127.0.0.1:5002 --image imgs/banner_1.png \
  --prompt-file /path/to/prompt.txt --seed 42 --frames 145 \
  --output artifacts/studio-quality/parity
```

Keep the resulting videos, frame hashes, reports and visual review snapshots in
the ignored output directory. Record the actual backend, ggml revision and any
unavailable devices or model scenarios with the run; passing request/unit tests
alone is not evidence of native generation quality.

### Editor and host checks

```sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
dotnet build eng/validation/TensorAgentHost/TensorAgentHost.csproj
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj --nologo -v:q
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter 'FullyQualifiedName~StudioHostTests|FullyQualifiedName~VideoTurnsTests|FullyQualifiedName~LoopbackServerTests|FullyQualifiedName~LoopbackRangeTests|FullyQualifiedName~MediaRoutesTests|FullyQualifiedName~AgentAppHostTests' --nologo -v:q
node --test eng/validation/tests/studio-ui.test.mjs
dotnet build TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj -f net10.0-maccatalyst -p:TensorSharpAppleTargets=true
```

Start a host as described in the [guide](video-studio.md), then:

```sh
python3 eng/studio/validate.py --url http://127.0.0.1:8187 --output artifacts/studio/server-check
python3 eng/studio/validate.py --connection artifacts/studio-agent/connection.json --output artifacts/studio/agent-check
# Add --generate only with an explicitly loaded real video model.
# After stopping/restarting the host with the same storage:
python3 eng/studio/validate.py --url http://127.0.0.1:8187 --reopen PROJECT_ID --output artifacts/studio/reopen-check
# Explicitly load a tool-capable text model before testing the actual planner:
python3 eng/studio/validate_agent.py --url http://127.0.0.1:8187 --output artifacts/studio/planner-check
python3 eng/studio/validate_cancel.py --connection artifacts/studio-agent/connection.json --project PROJECT_ID --output artifacts/studio/cancel-check
```

The harness needs native library discovery configured, just like the application.
For the recorded MLX run, `TENSORSHARP_MLX_LIBRARY_DIR` pointed to the built
`TensorSharp.Server.Host/bin` containing `libmlxc.dylib` and `libmlx.dylib`. The
first harness attempt without that setting failed model loading and was not
counted as successful inference. The network-denied run prepended:

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'
```

Do not publish harness `connection.json` or raw startup logs: they contain the
loopback authentication credential. Reports and project packages omit it.
