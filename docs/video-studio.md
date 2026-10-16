# Video Studio

Video Studio is a shared editor in **WebUX Chat** and **TensorAgent**. Open
**Video Studio** from the chat interface/menu, or **Open in Studio** beside a
video served by the current host. Media stays on that host. No cloud account or
AI model is required for manual editing.

## Start locally

Use the existing .NET 10 development setup. Desktop media import requires
operator-installed `ffprobe`. Studio generation and rendering additionally require
`ffmpeg` with an available H.264 encoder and AAC. These output capabilities are
checked before native generation can enter the queue. Studio does not download
these tools or install models. Review the
[dependency notes](video-studio-design.md#dependency-and-licensing-decisions)
before redistributing a media-tool build.

```sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
TS_STUDIO_DIRECTORY="$PWD/artifacts/my-studio" \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --host 127.0.0.1 --port 8187
```

Open `http://127.0.0.1:8187`. `TS_STUDIO_DIRECTORY` selects server project/job
storage; otherwise Studio uses its directory beside the server output. Do not
point two active hosts at the same storage directory. Use project packages to
move work between hosts.

Studio server access is local by default, even if other server APIs listen on a
network interface. Both the connection peer and Host header must be loopback.
For a deliberately remote deployment, protect the whole interface/API with an
authenticated proxy first, then set `TS_STUDIO_ALLOW_REMOTE=1` on startup. The
standalone server does not provide authentication or per-user project isolation;
that opt-in only removes Studio's loopback restriction. TensorAgent keeps its
existing local session-token protection.

TensorAgent includes the same editor in its existing MAUI interface. Its
in-process `AgentAppHost` stores Studio projects under its private data root,
in `studio-projects`. For desktop development without launching MAUI, the
existing real loopback-host harness runs that application core:

```sh
dotnet run --project eng/validation/TensorAgentHost -- \
  --root artifacts/studio-agent --skills TensorAgent/skills --port 8188
```

Use the authenticated `entryUrl` in the harness's local `connection.json`.
That file contains a session credential: do not publish it or add it to Git.
The harness exercises the real application core and shared web page; it is not
a physical iOS/Windows-device test.

## Edit a project

1. Open Studio, create a project, and set its dimensions and rational frame
   rate. Import video, still images or audio, or open an existing chat upload.
2. Add assets to video/audio tracks. Library assets remain available when their
   timeline clips are removed. A clip's own audio moves with its picture;
   explicitly linked clips share move/trim/split operations.
3. Select clips in the timeline or shot strip. Drag to move or use the inspector
   for exact frame positions, source in-points and lengths. Split at the cursor,
   duplicate, remove, change fit, adjust volume/fades, and mute/hide tracks.
   Snapping and zoom help with longer sequences; exact controls work without
   dragging and on touch layouts. Split outside active fades, or clear a fade
   first; splitting inside its envelope is refused to preserve its appearance.
   For a dissolve, first overlap two clips on the same video track, with the
   incoming clip starting and ending later. Apply the dissolve to that pair.
   It changes fades without moving either clip: the incoming picture fades in
   over the outgoing picture, while their audio gains crossfade. A third clip
   overlapping that interval or conflicting existing fades must be resolved first.
4. Add/edit timed captions. The initial portable caption renderer supports
   Basic Latin text and renders it uppercase; unsupported characters are refused
   for export. Automatic transcription and word-based cuts are not exposed.
5. Render a preview. This is a playable composition of a **captured project
   revision**, using the same rendering path as export. An edit marks an earlier
   preview stale. Source preview is separately labeled and does not claim to
   show the complete timeline.
6. Export the rendered MP4. Export is a background job; keep editing while it
   runs. The result belongs to the captured revision, not later changes.

Each successful command autosaves atomically. Undo/redo survives reopening.
An atomic multi-command proposal is one undo step. Import and generation results
survive undo. A failed transaction leaves the previous document intact.

### Delete an asset

In either host, open the **Asset library** and click **Delete asset** on the
asset's card. Confirm the dialog, or choose **Cancel** to keep it. If the asset
is used on the timeline, the warning shows how many clips use it. Confirming
removes the library entry and those clips together; other assets and their
clips remain in place. Use **Undo** to restore the asset and its removed clips
in one step; **Redo** removes them again. These edits autosave.

Deletion removes the asset from the current project. Studio retains its managed
media file for undo and renders already captured from earlier revisions, so this
action does not reclaim disk space. Your original imported file is preserved.

Thumbnails are bounded 320×180 derivatives. Waveforms contain 512 peak buckets,
not an entire soundtrack in browser memory. Codec playback still depends on the
browser/WebView; export errors report missing encoders rather than quietly
substituting another output format. Probing and derivative creation are bounded,
asynchronous and cancellable; unlike generation/export they are not durable
queued jobs. Completed thumbnails and waveforms are cached.

Default limits are 32 tracks, 2,000 clips/assets, 5,000 captions and 86,400 timeline
frames (one hour at 24 fps). A render supports at most 120 combined clips and
captions. Media imports and expanded project packages are limited to 2 GiB;
there is a 256 MiB free-space reserve. The most recent 40 edit transactions are
retained for undo. Host options can lower or adjust the configurable limits.

## Generate and refine

Load a native video model using the application's existing model controls (or
the server's existing `--model ... --backend ...` startup arguments). Studio
queries the loaded model. Wan, MiniMax-H3 and LTX 2.5 keep their own frame grids,
sampling recipes and conditioning rules. Unsupported options fail before a job
starts. No hidden model or remote-provider substitution occurs. A loaded model
is marked unavailable in Studio when this host cannot probe, validate and render
its output. Installing a model alone therefore does not enable generation on a
host without the required local media tools or subprocess support. Project
review and manual edits remain available.

Choose a prompt and basic dimensions/frame count; advanced controls expose seed,
steps, guidance, negative prompt where meaningful, and actual conditioning
capabilities. Temporal image guides are distinct from identity references.
Optional native audio depends on the loaded checkpoint and companions.
LoRA selection, native retake and generative extension are not Studio controls.
Ordinary trimming and fades are timeline operations, not generative editing.

New generations inherit the host's chat video defaults, including the configured
output dimensions, frame rate and clip length, resolved to the loaded model's
supported frame grid. With `config/ltx-2.5-distilled.json`, the default output is
1024×1024. `--video-frames 144` becomes 145 frames at 24 fps (about six seconds),
as in chat. LTX's first sampling pass runs at half the output size: lowering the
output to 512×512 also lowers that pass to 256×256 and can distort faces and
hands. Higher resolution requires more memory and generation time.

**New take** retains the selected take's settings. For an older 512×512 take,
choose **Use model defaults** before generating again to apply the current
dimensions and sampling defaults while keeping its prompt and conditioning.
Existing footage is not regenerated automatically. Studio and chat use the same
native generation and encoding path; compare them with the same image, prompt,
seed and settings. This removes Studio's default resolution disadvantage but
does not guarantee perfect anatomy for every model output.

A successful job adds a managed asset with prompt, effective settings, seed,
model identity and output metadata. Add the take to a shot or replace a selected
clip while retaining its placement. Original assets stay in the library for
comparison and undo; a replacement must fit the requested source range.

The existing host currently has **one active model slot**. Conversational editing
requires a text model that supports structured tools; video models are rejected
as planners. Ask the Studio assistant to inspect the selection and propose edits.
Review the proposal, then apply it. Changing the project makes old proposals
stale; regenerate the proposal against the current selection instead of
overwriting newer work.

The assistant can also save bounded shot-generation plans. After planning,
explicitly load the desired video model and run a planned shot. Its requested
settings are validated against that model before execution. Model switching is
not automatic. The project remains editable throughout; generated footage is
added through the same command layer as imported media.

## Portable projects and recovery

Export a `.tsstudio` project package to continue on another installation. Packages
contain project settings, tracks, clips, captions, asset metadata/provenance and
relative media files. They omit conversation history, host paths, credentials,
undo history and job history. Import creates a new project ID. Hash verification,
path restrictions and compressed/expanded size limits apply before publishing
the imported project.

Missing files are reported, not silently discarded. Relinking accepts the same
original bytes, verified by SHA-256, so existing source ranges and captured
renders retain their meaning. A new version of footage should be imported as a
new take. Schema 1 is the first Studio format; unknown future schemas are refused
without rewriting them. There is no historic Studio format to migrate yet.

Generation/export jobs survive panel closure and browser disconnection. Cancelling
a queued job stops it before execution. A running job first becomes **cancelling**
and only becomes **cancelled** after its worker/process cleanup finishes. On
restart, a generation result already committed to the asset library is reconciled
with its job as completed. Other unfinished jobs become **interrupted**. An
explicit retry uses a new request identifier and restarts the operation;
mid-denoising checkpoint resume is not available. Failed jobs retain their error. Free disk space, correct missing media/model
settings, then retry explicitly. The queue is bounded; Studio does not silently
lower quality after out-of-memory or disk errors.

Mobile can review and edit imported project packages. Desktop subprocess-based
probing, thumbnails, waveforms and export are unavailable on iOS/Android. Studio
generation is also disabled there because the required output validation and
rendering pipeline is unavailable; a loaded native model does not bypass this
gate. No desktop-equivalent mobile validation is claimed. There is no automatic remote
fallback. An explicitly configured remote host has its own storage and compute
boundary.

## Reproduce validation

```sh
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj
node --test eng/validation/tests/studio-ui.test.mjs

# Against the server command above; uses real synthetic media and actual FFmpeg.
python3 eng/studio/validate.py --url http://127.0.0.1:8187 \
  --output artifacts/studio-check

# Against the real TensorAgent loopback harness.
python3 eng/studio/validate.py --connection artifacts/studio-agent/connection.json \
  --output artifacts/studio-agent-check

# With a real video model already loaded; no test double or download is used.
python3 eng/studio/validate.py --url http://127.0.0.1:8187 \
  --generate --output artifacts/studio-native-check
```

After stopping/restarting the same host against the same data root, pass
`--reopen <projectId>` to the validation script. It reopens the document and
renders/decodes it again. Generated evidence belongs in ignored `artifacts/` or
`docs/validation/`, never force-added to Git. See
[the validation record](video-studio-validation.md) for actual environments,
results and remaining limitations.
