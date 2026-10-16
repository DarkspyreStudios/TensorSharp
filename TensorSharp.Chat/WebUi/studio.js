// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// One shared editor for WebUX and TensorAgent. The host owns all persistent state.
(function () {
  'use strict';
  const API = '/api/studio';
  const S = { root: null, project: null, projects: [], capabilities: null, selected: null, selectedCaption: null,
    playhead: 0, zoom: 65, snap: true, busy: false, jobs: [], proposal: null, tab: 'clip',
    previewRevision: null, previewJob: null, sourcePreview: false, sourcePreviewAssetId: null, timer: null, epoch: 0, previousFocus: null, generationDrafts: [], deleteDialog: null };
  const ui = {};
  function tr(key, fallback, params) {
    const i18n = window.TensorSharpI18n;
    if (i18n && typeof i18n.t === 'function') return i18n.t(key, fallback, params);
    return String(fallback).replace(/\{(\w+)\}/g, (all, k) => params && Object.prototype.hasOwnProperty.call(params, k) ? String(params[k]) : all);
  }
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function button(text, action, cls) {
    const b = el('button', cls, text); b.type = 'button';
    b.addEventListener('click', () => run(action)); return b;
  }
  function field(parent, label, value, type, options) {
    const l = el('label', 'ts-field', label); const n = el(type === 'textarea' ? 'textarea' : type === 'select' ? 'select' : 'input');
    if (type !== 'textarea' && type !== 'select') n.type = type || 'text';
    if (options) for (const o of options) { const item = el('option', '', o.label); item.value = o.value; item.disabled = !!o.disabled; n.appendChild(item); }
    if (value != null) n.value = String(value); l.appendChild(n); parent.appendChild(l); return n;
  }
  function num(parent, label, value, min, max, step) { const n = field(parent, label, value, 'number'); n.min = min == null ? 0 : min; if (max != null) n.max = max; n.step = step || 1; return n; }
  function checked(parent, label, value, changed) { const l = el('label', 'ts-check'); const n = el('input'); n.type = 'checkbox'; n.checked = value; l.append(n, document.createTextNode(label)); if (changed) n.addEventListener('change', () => run(() => changed(n.checked))); parent.appendChild(l); return n; }
  function row(parent) { const n = el('div', 'ts-row'); parent.appendChild(n); return n; }
  function id() { return window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID().replace(/-/g, '') : Date.now().toString(16) + Math.random().toString(16).slice(2); }
  function fps() { return S.project ? S.project.frameRate.numerator / S.project.frameRate.denominator : 24; }
  function seconds(frames) { return frames / fps(); }
  function frames(sec) { return Math.round(sec * fps()); }
  function fmt(frame) { return seconds(frame).toFixed(2) + 's · ' + frame + 'f'; }
  function clip() { return S.project && S.project.clips.find(c => c.id === S.selected); }
  function asset(assetId) { return S.project && S.project.assets.find(a => a.id === assetId); }
  function mediaUrl(assetId) { return API + '/projects/' + encodeURIComponent(S.project.id) + '/assets/' + encodeURIComponent(assetId); }
  function outputUrl(job) { return API + '/projects/' + encodeURIComponent(job.projectId) + '/jobs/' + encodeURIComponent(job.id) + '/output'; }
  function status(text, error) { if (!ui.status) return; ui.status.textContent = text; ui.status.dataset.error = String(!!error); }
  async function response(res) {
    const text = await res.text(); let value;
    try { value = text ? JSON.parse(text) : {}; } catch (_) { value = { message: text.slice(0, 400) }; }
    if (!res.ok) { const detail = value.message || value.error || value.detail || res.statusText; const e = new Error(typeof detail === 'string' ? detail : detail.message || JSON.stringify(detail)); e.status = res.status; e.code = value.code; throw e; }
    return value;
  }
  async function api(action, payload) {
    return response(await fetch(API, { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: JSON.stringify({ action, payload: payload || {} }) }));
  }
  async function run(action) {
    try { return await action(); }
    catch (e) {
      status(e.message || String(e), true);
      if (S.project && (e.status === 409 || e.code === 'revision_conflict')) {
        try { await refreshProject(); status(tr('studio.conflict', 'The project changed. Your edit was not applied. Review the latest revision and try again.'), true); } catch (_) { /* Keep the original error. */ }
      }
    }
  }
  function setBusy(value) { S.busy = value; ui.save.textContent = value ? tr('studio.saving', 'Saving…') : S.project ? tr('studio.saved', 'Saved · revision {revision}', { revision: S.project.revision }) : tr('studio.local', 'Local project storage'); }
  function receiveProject(value) {
    const p = value.project || value; if (!p.id || !Array.isArray(p.tracks)) throw new Error(tr('studio.invalidProject', 'The host returned an invalid project.')); S.project = p;
    if (S.selected && !p.clips.some(c => c.id === S.selected)) S.selected = null;
    if (S.regenerateAsset && !p.assets.some(a => a.id === S.regenerateAsset)) S.regenerateAsset = null;
    if (S.sourcePreviewAssetId && !p.assets.some(a => a.id === S.sourcePreviewAssetId)) resetPreview();
    return p;
  }
  async function change(commands) {
    if (!S.project || S.busy) return;
    const projectId = S.project.id, expectedRevision = S.project.revision;
    setBusy(true);
    try {
      const p = await api('commands/apply', { projectId, expectedRevision, commandId: id(), commands });
      if (S.project && S.project.id === projectId) { receiveProject(p); renderProject(); status(tr('studio.applied', 'Edit saved. Source assets are preserved.')); return true; }
    } finally { setBusy(false); }
  }
  async function history(kind) {
    if (!S.project || S.busy) return;
    const projectId = S.project.id, expectedRevision = S.project.revision;
    setBusy(true);
    try { const project = await api('commands/' + kind, { projectId, expectedRevision, commandId: id() }); if (S.project && S.project.id === projectId) { receiveProject(project); renderProject(); status(tr('studio.applied', 'Edit saved. Source assets are preserved.')); } }
    finally { setBusy(false); }
  }
  async function listProjects() {
    const value = await api('projects/list'); S.projects = Array.isArray(value) ? value : value.projects || [];
    const selected = S.project && S.project.id; ui.projectSelect.replaceChildren();
    const empty = el('option', '', tr('studio.chooseProject', 'Open a project…')); empty.value = ''; ui.projectSelect.appendChild(empty);
    for (const p of S.projects) { const n = el('option', '', p.name); n.value = p.id; ui.projectSelect.appendChild(n); }
    ui.projectSelect.value = selected || '';
  }
  async function newProject() {
    const name = await projectName();
    if (name == null || !name.trim()) return;
    receiveProject(await api('projects/create', { name: name.trim(), width: 1280, height: 720, fpsNumerator: 24, fpsDenominator: 1 }));
    S.selected = null; S.selectedCaption = null; S.proposal = null; S.generationDrafts = []; S.playhead = 0; resetPreview(); renderProject(); rememberProject(); await listProjects(); await refreshJobs();
  }
  function projectName() {
    return new Promise(resolve => {
      const previous = document.activeElement, backdrop = el('div', 'ts-name-dialog'), form = el('form', 'ts-stack');
      form.setAttribute('role', 'dialog'); form.setAttribute('aria-modal', 'true'); form.setAttribute('aria-label', tr('studio.createProject', 'Create project'));
      form.appendChild(el('h2', '', tr('studio.createProject', 'Create project'))); const input = field(form, tr('studio.namePrompt', 'Project name'), tr('studio.untitled', 'Untitled video')); input.required = true; input.maxLength = 200;
      const finish = value => { backdrop.remove(); if (previous && previous.isConnected) previous.focus(); resolve(value); };
      const actions = row(form); actions.appendChild(button(tr('studio.cancel', 'Cancel'), () => finish(null)));
      const submit = el('button', 'ts-primary', tr('studio.create', 'Create')); submit.type = 'submit'; actions.appendChild(submit);
      form.addEventListener('submit', e => { e.preventDefault(); if (input.value.trim()) finish(input.value.trim()); });
      backdrop.addEventListener('keydown', e => {
        if (e.key === 'Escape') { e.stopPropagation(); e.preventDefault(); finish(null); }
        if (e.key === 'Tab') { e.stopPropagation(); const focusable = [input, ...actions.querySelectorAll('button')]; const first = focusable[0], last = focusable[focusable.length - 1]; if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); } }
      });
      backdrop.appendChild(form); S.root.appendChild(backdrop); input.focus(); input.select();
    });
  }
  function rememberProject() { try { window.localStorage.setItem('tensorsharp.studio.project', S.project.id); } catch (_) { /* Storage can be disabled. */ } }
  async function openProject(projectId) {
    if (!projectId || S.busy) return; ++S.epoch;
    receiveProject(await api('projects/get', { projectId })); S.selected = null; S.selectedCaption = null; S.proposal = null; S.generationDrafts = []; S.playhead = 0;
    resetPreview(); renderProject(); rememberProject(); await refreshJobs(); await loadProposals(); status(tr('studio.opened', 'Project reopened. Every accepted edit is saved by the host.'));
  }
  async function refreshProject() {
    if (!S.project) return; const projectId = S.project.id; const value = await api('projects/get', { projectId });
    if (S.project.id === projectId) { receiveProject(value); renderProject(); }
  }
  function build() {
    if (S.root) return;
    S.root = el('section', 'ts-studio'); S.root.hidden = true; S.root.tabIndex = -1; S.root.setAttribute('role', 'dialog'); S.root.setAttribute('aria-modal', 'true'); S.root.setAttribute('aria-label', tr('studio.title', 'Video Studio'));
    const header = el('div', 'ts-studio-header'); header.appendChild(el('h2', '', tr('studio.title', 'Video Studio')));
    ui.projectSelect = el('select'); ui.projectSelect.setAttribute('aria-label', tr('studio.projects', 'Projects')); ui.projectSelect.addEventListener('change', () => run(() => openProject(ui.projectSelect.value)));
    header.append(ui.projectSelect, button(tr('studio.new', 'New'), newProject));
    header.append(button(tr('studio.reopen', 'Reopen'), refreshProject));
    ui.save = el('span', 'ts-save-state', tr('studio.local', 'Local project storage')); ui.save.setAttribute('aria-live', 'polite'); header.appendChild(ui.save);
    header.appendChild(button(tr('studio.backChat', 'Back to chat'), close)); S.root.appendChild(header);
    ui.status = el('div', 'ts-status', tr('studio.welcomeHint', 'Import media or generate footage, then assemble an editable timeline.')); ui.status.setAttribute('role', 'status'); ui.status.setAttribute('aria-live', 'polite'); S.root.appendChild(ui.status);
    const mobile = el('div', 'ts-mobile-tabs');
    mobile.append(button(tr('studio.library', 'Library'), () => togglePanel('library')), button(tr('studio.properties', 'Properties'), () => { S.tab = 'clip'; renderInspector(); togglePanel('inspector'); }), button(tr('studio.generate', 'Generate'), () => { S.tab = 'generate'; renderInspector(); togglePanel('inspector', true); }), button(tr('studio.assistant', 'Assistant'), () => { S.tab = 'assistant'; renderInspector(); togglePanel('inspector', true); }));
    const hidePanels = button('×', () => { ui.library.dataset.panelHidden = 'true'; ui.inspector.dataset.panelHidden = 'true'; }); hidePanels.setAttribute('aria-label', tr('studio.closePanels', 'Close side panels')); mobile.appendChild(hidePanels);
    S.root.appendChild(mobile);
    ui.workspace = el('div', 'ts-studio-workspace'); S.root.appendChild(ui.workspace);
    ui.library = el('aside', 'ts-studio-section ts-studio-library'); ui.library.dataset.panelHidden = 'true';
    ui.preview = el('section', 'ts-studio-section ts-studio-preview'); ui.inspector = el('aside', 'ts-studio-section ts-studio-inspector'); ui.inspector.dataset.panelHidden = 'true';
    ui.timeline = el('section', 'ts-studio-timeline'); ui.workspace.append(ui.library, ui.preview, ui.inspector, ui.timeline);
    ui.import = el('input'); ui.import.type = 'file'; ui.import.multiple = true; ui.import.accept = 'video/*,audio/*,image/*'; ui.import.hidden = true;
    ui.import.addEventListener('change', () => run(async () => { const files = Array.from(ui.import.files || []); ui.import.value = ''; for (const file of files) await importFile(file); })); S.root.appendChild(ui.import);
    S.root.addEventListener('keydown', keydown); document.body.appendChild(S.root); renderProject();
  }
  function togglePanel(name, force) { const panel = ui[name]; panel.dataset.panelHidden = String(force ? false : panel.dataset.panelHidden !== 'true'); if (name === 'library') ui.inspector.dataset.panelHidden = 'true'; else ui.library.dataset.panelHidden = 'true'; }
  async function open(options) {
    build(); S.previousFocus = document.activeElement; S.root.hidden = false; S.root.focus();
    const tasks = await Promise.allSettled([api('capabilities'), listProjects()]);
    if (tasks[0].status === 'fulfilled') S.capabilities = tasks[0].value;
    else status(tasks[0].reason.message, true);
    if (!S.project) { let previous; try { previous = window.localStorage.getItem('tensorsharp.studio.project'); } catch (_) { }
      if (previous && S.projects.some(p => p.id === previous)) await run(() => openProject(previous)); }
    renderProject();
    if (options && options.url) {
      if (!S.project) await newProject();
      if (S.project) await importChat(options.url);
    }
    if (S.timer) clearTimeout(S.timer); poll();
    ui.projectSelect.focus();
  }
  function close() { if (!S.root) return; if (S.deleteDialog) S.deleteDialog(false); S.root.hidden = true; if (ui.video) ui.video.pause(); if (S.timer) clearTimeout(S.timer); S.timer = null; if (S.previousFocus && S.previousFocus.isConnected) S.previousFocus.focus(); }
  async function poll() { if (!S.root || S.root.hidden) return; try { if (S.project) await refreshJobs(); } catch (e) { status(tr('studio.connection', 'Connection interrupted. Project and jobs remain on the host. {error}', { error: e.message }), true); } finally { if (!S.root.hidden) S.timer = setTimeout(poll, 2000); } }
  function renderProject() {
    if (!S.root) return; setBusy(S.busy);
    if (!S.project) {
      ui.library.replaceChildren(el('h3', '', tr('studio.library', 'Asset library')), el('p', 'ts-empty', tr('studio.libraryEmpty', 'Your imported media and generated takes appear here.')));
      ui.preview.replaceChildren(el('div', 'ts-welcome'));
      const welcome = ui.preview.firstChild; welcome.append(el('h2', '', tr('studio.createTitle', 'Make room for your next story')), el('p', '', tr('studio.createDescription', 'Create a local project, import footage, and build a timeline. Editing works without an AI model.')), button(tr('studio.createProject', 'Create project'), newProject, 'ts-primary'));
      ui.inspector.replaceChildren(); ui.timeline.replaceChildren(); return;
    }
    ui.projectSelect.value = S.project.id;
    renderLibrary(); buildPreview(); renderTimeline(); renderInspector(); updatePlayback();
  }
  async function importFile(file) {
    if (!S.project) return; const projectId = S.project.id; status(tr('studio.importing', 'Importing and probing {name}…', { name: file.name }));
    const value = await response(await fetch(API + '/projects/' + encodeURIComponent(projectId) + '/import?fileName=' + encodeURIComponent(file.name), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
    if (S.project && S.project.id === projectId) { if (value.project || value.tracks) receiveProject(value); else await refreshProject(); renderProject(); status(tr('studio.imported', 'Media imported. Add it to any compatible track.')); }
  }
  async function importChat(url) {
    const safe = localMediaUrl(url); if (!safe) throw new Error(tr('studio.localOnly', 'Studio imports only media served by this host. Remote URLs are not fetched.'));
    const projectId = S.project.id; await api('assets/import-chat', { projectId, url: safe }); if (S.project.id === projectId) await refreshProject(); status(tr('studio.chatImported', 'Chat media is available in this project’s asset library.'));
  }
  function renderLibrary() {
    ui.library.replaceChildren(); const head = el('div', 'ts-section-heading'); const importButton = button(tr('studio.import', 'Import'), () => ui.import.click()); importButton.disabled = S.capabilities && S.capabilities.importAvailable === false;
    importButton.title = importButton.disabled ? (S.capabilities.renderUnavailableReason || tr('studio.importUnsupported', 'Import probing is unavailable on this host.')) : '';
    head.append(el('h3', '', tr('studio.library', 'Asset library')), importButton); ui.library.appendChild(head);
    ui.library.appendChild(el('p', 'ts-hint', tr('studio.assetHint', 'Drag assets to a track, or use Add. Removing a clip preserves its source.')));
    if (!S.project.assets.length) ui.library.appendChild(el('p', 'ts-empty', tr('studio.noAssets', 'Import video, images, or audio to begin.')));
    for (const a of S.project.assets) {
      const card = el('article', 'ts-asset'); card.dataset.missing = String(!!a.missing); card.draggable = true;
      card.addEventListener('dragstart', e => { e.dataTransfer.setData('application/x-tensorsharp-asset', a.id); e.dataTransfer.effectAllowed = 'copy'; });
      if (!a.missing && a.kind !== 'audio' && S.capabilities && S.capabilities.externalProcesses) {
        const thumbnail = el('img', 'ts-thumbnail'); thumbnail.loading = 'lazy'; thumbnail.alt = ''; thumbnail.width = 320; thumbnail.height = 180;
        thumbnail.src = API + '/thumbnail/' + encodeURIComponent(S.project.id) + '/' + encodeURIComponent(a.id);
        thumbnail.addEventListener('error', () => { thumbnail.hidden = true; }); card.appendChild(thumbnail);
      }
      card.append(el('strong', '', a.name), el('p', 'ts-hint', a.kind + ' · ' + (a.durationSeconds ? a.durationSeconds.toFixed(2) + 's' : a.width + '×' + a.height) + (a.generation ? ' · ' + tr('studio.generated', 'Generated') : '')));
      if (a.variableFrameRate) card.appendChild(el('p', 'ts-hint', tr('studio.vfr', 'Variable frame rate: output sampled by timestamps.')));
      if (a.missing) {
        card.appendChild(el('p', 'ts-hint', tr('studio.missing', 'Source media missing. Relink the original file to preserve timeline references.')));
        const relink = el('input'); relink.type = 'file'; relink.accept = 'video/*,audio/*,image/*'; relink.hidden = true;
        const projectId = S.project.id;
        relink.addEventListener('change', () => run(async () => {
          const file = relink.files[0]; if (!file) return;
          const value = await response(await fetch(API + '/projects/' + encodeURIComponent(projectId) + '/assets/' + encodeURIComponent(a.id) + '/relink', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
          if (S.project.id === projectId) { receiveProject(value); renderProject(); status(tr('studio.relinked', 'Original media restored. The host verified its content hash.')); }
        })); card.append(relink, button(tr('studio.relink', 'Relink original file'), () => relink.click()));
      }
      const actions = row(card); const add = button(tr('studio.add', 'Add'), () => insertAsset(a.id)); add.disabled = !!a.missing; actions.append(add, button(tr('studio.source', 'Source'), () => previewSource(a)));
      if (clip() && !a.missing) actions.appendChild(button(tr('studio.useTake', 'Use take'), () => change([{ type: 'replaceAsset', clipId: S.selected, assetId: a.id }])));
      if (a.generation) actions.appendChild(button(tr('studio.regenerate', 'New take'), () => { S.tab = 'generate'; S.regenerateAsset = a.id; renderInspector(); togglePanel('inspector', true); }));
      if (a.hasAudio && !a.missing && S.capabilities && S.capabilities.externalProcesses) actions.appendChild(button(tr('studio.waveform', 'Waveform'), () => showWaveform(card, a)));
      const remove = button(tr('studio.deleteAsset', 'Delete asset'), () => deleteAsset(a.id), 'ts-danger');
      remove.setAttribute('aria-label', tr('studio.deleteAssetNamed', 'Delete asset: {name}', { name: a.name })); actions.appendChild(remove);
      ui.library.appendChild(card);
    }
  }
  function confirmAssetDeletion(a, usageCount) {
    return new Promise(resolve => {
      const previous = document.activeElement, root = S.root, backdrop = el('div', 'ts-name-dialog ts-delete-dialog'), dialog = el('div', 'ts-stack ts-delete-content');
      const title = el('h2', '', tr('studio.deleteAssetTitle', 'Delete “{name}”?', { name: a.name })); title.id = 'ts-delete-title-' + id();
      const warning = el('p', usageCount ? 'ts-delete-warning' : '', usageCount
        ? tr('studio.deleteAssetUsed', 'This asset is used by {count} timeline clip(s). Deleting it will also remove those clips. Other assets and their clips will stay.', { count: usageCount })
        : tr('studio.deleteAssetUnused', 'This asset is not used on the timeline. It will be removed from this project’s library.'));
      warning.id = 'ts-delete-warning-' + id();
      const retention = el('p', 'ts-hint', tr('studio.deleteAssetRetention', 'You can Undo this change. Original files and managed media are retained for Undo and existing jobs; this does not free disk space.')); retention.id = 'ts-delete-retention-' + id();
      dialog.setAttribute('role', 'alertdialog'); dialog.setAttribute('aria-modal', 'true'); dialog.setAttribute('aria-labelledby', title.id); dialog.setAttribute('aria-describedby', warning.id + ' ' + retention.id);
      dialog.append(title, warning, retention);
      const background = Array.from(root.children).map(node => ({ node, inert: node.inert })); for (const item of background) item.node.inert = true;
      let finished = false;
      const finish = value => { if (finished) return; finished = true; backdrop.remove(); for (const item of background) item.node.inert = item.inert; S.deleteDialog = null; if (previous && previous.isConnected) previous.focus(); resolve(value); };
      S.deleteDialog = finish;
      const actions = row(dialog), cancel = button(tr('studio.cancel', 'Cancel'), () => finish(false)), confirm = button(tr('studio.deleteAsset', 'Delete asset'), () => finish(true), 'ts-danger'); actions.append(cancel, confirm);
      backdrop.addEventListener('keydown', e => {
        e.stopPropagation();
        if (e.key === 'Escape') { e.preventDefault(); finish(false); }
        else if (e.key === 'Tab') {
          if (e.shiftKey && document.activeElement === cancel) { e.preventDefault(); confirm.focus(); }
          else if (!e.shiftKey && document.activeElement === confirm) { e.preventDefault(); cancel.focus(); }
          else if (document.activeElement !== cancel && document.activeElement !== confirm) { e.preventDefault(); cancel.focus(); }
        }
      });
      backdrop.appendChild(dialog); root.appendChild(backdrop); cancel.focus();
    });
  }
  async function deleteAsset(assetId) {
    if (!S.project || S.busy || S.deleteDialog) return;
    const a = asset(assetId); if (!a) return;
    const projectId = S.project.id, expectedRevision = S.project.revision, epoch = S.epoch;
    const usageCount = S.project.clips.filter(c => c.assetId === assetId).length;
    if (!await confirmAssetDeletion(a, usageCount)) return;
    // The confirmation authorizes only the project revision and usages the user reviewed.
    if (!S.project || S.project.id !== projectId || S.project.revision !== expectedRevision || S.epoch !== epoch || S.busy)
      throw new Error(tr('studio.deleteAssetChanged', 'The project changed while you reviewed deletion. Nothing was deleted. Review the asset and try again.'));
    const applied = await change([{ type: 'deleteAsset', assetId, removeTimelineClips: usageCount > 0 }]);
    if (applied && S.project && S.project.id === projectId) {
      status(tr('studio.assetDeleted', 'Asset removed from the library. Use Undo to restore it and any removed clips. Source files are preserved.'));
      if (ui.library) { ui.library.tabIndex = -1; ui.library.focus(); }
    }
  }
  async function insertAsset(assetId, trackId, atFrame) {
    const a = asset(assetId); if (!a) return; const kind = a.kind === 'audio' ? 'audio' : 'video'; const commands = [];
    if (!trackId) { const existing = S.project.tracks.find(t => t.kind === kind); if (existing) trackId = existing.id; else { trackId = id(); commands.push({ type: 'addTrack', trackId, kind, name: kind === 'audio' ? tr('studio.audioTrack', 'Audio') : tr('studio.videoTrack', 'Video') }); } }
    const durationFrames = a.kind === 'image' ? frames(5) : Math.max(1, Math.floor(a.durationSeconds * fps()));
    const clipId = id(); commands.push({ type: 'insertClip', clipId, assetId, trackId, startFrame: atFrame == null ? S.playhead : atFrame, durationFrames, sourceStartFrame: 0 });
    await change(commands); selectClip(clipId);
  }
  async function showWaveform(card, a) {
    if (card.querySelector('.ts-waveform')) return;
    const projectId = S.project.id; const value = await response(await fetch(API + '/waveform/' + encodeURIComponent(projectId) + '/' + encodeURIComponent(a.id), { credentials: 'same-origin' }));
    if (!card.isConnected) return; const peaks = (value.peaks || []).slice(0, 512); if (!peaks.length) return;
    const canvas = el('canvas', 'ts-waveform'); canvas.width = 512; canvas.height = 64; canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', tr('studio.waveformLabel', 'Audio amplitude waveform for {name}', { name: a.name }));
    const context = canvas.getContext('2d'); context.strokeStyle = '#79c9ad'; context.lineWidth = 1;
    for (let i = 0; i < peaks.length; i++) { const height = Math.max(1, Math.min(1, Math.abs(Number(peaks[i]) || 0)) * 30); context.beginPath(); context.moveTo(i + .5, 32 - height); context.lineTo(i + .5, 32 + height); context.stroke(); }
    card.appendChild(canvas);
  }
  function resetPreview() { if (ui.video) { ui.video.pause(); ui.video.removeAttribute('src'); ui.video.load(); ui.video.hidden = true; } if (ui.previewMessage) ui.previewMessage.hidden = false; S.previewRevision = null; S.previewJob = null; S.sourcePreview = false; S.sourcePreviewAssetId = null; S.awaitPreviewJob = null; }
  function buildPreview() {
    if (ui.preview.querySelector('.ts-monitor')) { renderShots(); previewNotice(); return; }
    ui.preview.replaceChildren(); const monitor = el('div', 'ts-monitor'); ui.video = el('video'); ui.video.controls = true; ui.video.playsInline = true; ui.video.preload = 'metadata'; ui.video.hidden = true;
    ui.video.addEventListener('timeupdate', () => { if (!S.sourcePreview) { S.playhead = Math.round(ui.video.currentTime * fps()); updatePlayback(); } });
    ui.video.addEventListener('error', () => status(tr('studio.playbackError', 'This browser could not decode the media. Download the rendered output or try a compatible browser.'), true));
    ui.previewMessage = el('p', 'ts-hint', tr('studio.previewEmpty', 'Build a timeline, then render its preview. Preview and export use the same composition.'));
    monitor.append(ui.video, ui.previewMessage); ui.preview.appendChild(monitor);
    const playback = el('div', 'ts-playback'); ui.playhead = el('input'); ui.playhead.type = 'range'; ui.playhead.min = 0; ui.playhead.step = 1; ui.playhead.setAttribute('aria-label', tr('studio.scrub', 'Timeline playhead in frames'));
    ui.playhead.addEventListener('input', () => seek(Number(ui.playhead.value))); ui.time = el('span', 'ts-time'); playback.append(ui.playhead, ui.time); ui.preview.appendChild(playback);
    ui.previewLabel = el('p', 'ts-hint'); ui.preview.appendChild(ui.previewLabel);
    const actions = row(ui.preview); ui.renderButton = button(tr('studio.renderPreview', 'Render preview'), () => renderJob(true), 'ts-primary'); ui.exportButton = button(tr('studio.export', 'Export MP4'), () => renderJob(false)); actions.append(ui.renderButton, ui.exportButton);
    ui.shots = el('div', 'ts-shots'); ui.shots.setAttribute('aria-label', tr('studio.storyboard', 'Storyboard shots')); ui.preview.appendChild(ui.shots); renderShots(); previewNotice();
  }
  function renderAllowed() { return !!(S.capabilities && S.capabilities.renderAvailable); }
  function previewNotice() {
    if (!ui.previewLabel) return;
    ui.previewLabel.textContent = S.sourcePreview ? tr('studio.sourcePreview', 'Source preview · timeline effects are not shown here.') : S.previewRevision == null ? tr('studio.previewNeedRender', 'No composition preview rendered yet.') : tr(S.previewRevision === S.project.revision ? 'studio.previewCurrent' : 'studio.previewStale', S.previewRevision === S.project.revision ? 'Composition preview · revision {revision}' : 'Preview is from revision {revision}. Render again to see your latest edits.', { revision: S.previewRevision });
    const unavailable = (S.capabilities || {}).renderUnavailableReason || tr('studio.renderUnsupported', 'Local rendering is unavailable on this host.');
    if (ui.renderButton) { ui.renderButton.disabled = !renderAllowed() || !S.project.durationFrames; ui.renderButton.title = renderAllowed() ? '' : unavailable; }
    if (ui.exportButton) { ui.exportButton.disabled = !renderAllowed() || !S.project.durationFrames; ui.exportButton.title = renderAllowed() ? '' : unavailable; }
  }
  function previewSource(a) {
    if (a.kind === 'image') { status(tr('studio.imagePreview', 'Add this image to the timeline and render a preview.')); return; }
    S.sourcePreview = true; S.sourcePreviewAssetId = a.id; ui.video.src = mediaUrl(a.id); ui.video.hidden = false; ui.previewMessage.hidden = true; previewNotice();
  }
  function showRender(job) { S.previewJob = job.id; S.previewRevision = job.projectRevision; S.sourcePreview = false; S.sourcePreviewAssetId = null; ui.video.src = outputUrl(job); ui.video.hidden = false; ui.previewMessage.hidden = true; previewNotice(); }
  function renderShots() {
    if (!ui.shots || !ui.shots.isConnected) return; ui.shots.replaceChildren();
    for (const c of S.project.clips.filter(c => { const a = asset(c.assetId); return a && a.kind !== 'audio'; }).sort((a, b) => a.startFrame - b.startFrame)) {
      const a = asset(c.assetId); const b = button(a.name + ' · ' + seconds(c.durationFrames).toFixed(1) + 's', () => selectClip(c.id), 'ts-shot'); b.setAttribute('aria-pressed', String(S.selected === c.id)); ui.shots.appendChild(b);
    }
  }
  function seek(frame) { S.playhead = Math.max(0, Math.min(S.project.durationFrames, Math.round(frame))); if (ui.video && ui.video.src && !S.sourcePreview && Number.isFinite(ui.video.duration)) ui.video.currentTime = Math.min(seconds(S.playhead), ui.video.duration); updatePlayback(); }
  function updatePlayback() {
    if (!S.project || !ui.playhead) return; ui.playhead.max = Math.max(1, S.project.durationFrames); ui.playhead.value = S.playhead; ui.time.textContent = fmt(S.playhead);
    for (const line of ui.timeline.querySelectorAll('.ts-playhead')) line.style.left = (seconds(S.playhead) * S.zoom) + 'px';
  }
  async function renderJob(preview) {
    const job = await api('jobs/render', { projectId: S.project.id, expectedRevision: S.project.revision, requestId: id() });
    if (preview) S.awaitPreviewJob = job.id || (job.job && job.job.id); status(tr('studio.renderQueued', 'Render queued against the current project revision. You can continue editing.')); S.tab = 'jobs'; renderInspector(); await refreshJobs();
  }
  async function refreshJobs() {
    if (!S.project) return; const projectId = S.project.id; const value = await api('jobs/list', { projectId }); if (!S.project || S.project.id !== projectId) return;
    const next = Array.isArray(value) ? value : value.jobs || []; const prior = new Map(S.jobs.map(j => [j.id, j.state])); S.jobs = next;
    const completed = next.some(j => j.state === 'completed' && prior.get(j.id) !== 'completed' && j.kind === 'generate');
    if (completed && !S.busy) await refreshProject();
    if (S.tab === 'jobs') renderInspector();
    const wanted = S.jobs.find(j => j.id === S.awaitPreviewJob && j.state === 'completed');
    if (wanted) { showRender(wanted); S.awaitPreviewJob = null; }
    if (!S.previewJob && !S.sourcePreview) { const latest = S.jobs.find(j => j.kind === 'render' && j.state === 'completed' && j.projectRevision === S.project.revision); if (latest) showRender(latest); }
  }
  function selectClip(clipId) { S.selected = clipId; S.selectedCaption = null; S.tab = 'clip'; renderInspector(); renderTimeline(); renderLibrary(); renderShots(); }
  function snapped(frame, movingId) {
    frame = Math.max(0, Math.round(frame)); if (!S.snap) return frame;
    const targets = [0, S.playhead];
    for (const c of S.project.clips) if (c.id !== movingId) targets.push(c.startFrame, c.startFrame + c.durationFrames);
    const tolerance = Math.max(1, Math.round(8 * fps() / S.zoom)); let best = frame, distance = tolerance + 1;
    for (const target of targets) { const next = Math.abs(target - frame); if (next <= tolerance && next < distance) { best = target; distance = next; } }
    return best;
  }
  function renderTimeline() {
    const oldScroll = ui.timeline.querySelector('.ts-timeline-scroll'); const scrollLeft = oldScroll ? oldScroll.scrollLeft : 0; const scrollTop = oldScroll ? oldScroll.scrollTop : 0;
    ui.timeline.replaceChildren(); const toolbar = el('div', 'ts-timeline-toolbar'); toolbar.append(el('h3', '', tr('studio.timeline', 'Timeline')));
    toolbar.append(button(tr('studio.undo', 'Undo'), () => history('undo')), button(tr('studio.redo', 'Redo'), () => history('redo')),
      button(tr('studio.videoTrackAdd', '+ Video'), () => change([{ type: 'addTrack', name: tr('studio.videoTrack', 'Video'), kind: 'video' }])),
      button(tr('studio.audioTrackAdd', '+ Audio'), () => change([{ type: 'addTrack', name: tr('studio.audioTrack', 'Audio'), kind: 'audio' }])));
    const split = button(tr('studio.split', 'Split'), () => splitSelected()); split.disabled = !clip(); toolbar.appendChild(split);
    const remove = button(tr('studio.remove', 'Remove'), () => removeSelected()); remove.disabled = !clip(); toolbar.appendChild(remove);
    checked(toolbar, tr('studio.snap', 'Snap'), S.snap, v => { S.snap = v; });
    const zoom = el('input'); zoom.type = 'range'; zoom.min = 15; zoom.max = 220; zoom.value = S.zoom; zoom.setAttribute('aria-label', tr('studio.zoom', 'Timeline zoom'));
    zoom.addEventListener('input', () => { S.zoom = Number(zoom.value); renderTimeline(); }); toolbar.appendChild(zoom); ui.timeline.appendChild(toolbar);
    const scroll = el('div', 'ts-timeline-scroll'); scroll.setAttribute('aria-label', tr('studio.timeline', 'Timeline')); ui.timeline.appendChild(scroll);
    const duration = Math.max(10, seconds(S.project.durationFrames) + 4); const width = Math.min(200000, duration * S.zoom);
    const rulerRow = el('div', 'ts-ruler-row'); rulerRow.appendChild(el('div', 'ts-track-label', fps().toFixed(3).replace(/0+$/, '').replace(/\.$/, '') + ' fps'));
    const ruler = el('div', 'ts-ruler'); ruler.style.width = width + 'px'; const tickStep = Math.max(S.zoom < 30 ? 5 : 1, Math.ceil(duration / 1500));
    for (let second = 0; second < duration; second += tickStep) { const tick = el('span', 'ts-tick', second + 's'); tick.style.left = second * S.zoom + 3 + 'px'; ruler.appendChild(tick); }
    ruler.addEventListener('pointerdown', e => { seek(frames((e.clientX - ruler.getBoundingClientRect().left) / S.zoom)); }); rulerRow.appendChild(ruler); scroll.appendChild(rulerRow);
    const addPlayhead = lane => { const play = el('div', 'ts-playhead'); play.style.left = seconds(S.playhead) * S.zoom + 'px'; lane.appendChild(play); };
    addPlayhead(ruler);
    for (let index = 0; index < S.project.tracks.length; index++) {
      const t = S.project.tracks[index], trackRow = el('div', 'ts-track-row'), label = el('div', 'ts-track-label');
      label.appendChild(el('strong', '', t.name));
      const mute = button(t.muted ? tr('studio.unmuteShort', 'M̸') : tr('studio.muteShort', 'M'), () => change([{ type: 'updateTrack', trackId: t.id, muted: !t.muted }])); mute.title = t.muted ? tr('studio.unmute', 'Unmute track') : tr('studio.mute', 'Mute track'); mute.setAttribute('aria-pressed', String(t.muted)); label.appendChild(mute);
      if (t.kind !== 'audio') { const visible = button(t.visible ? '◉' : '○', () => change([{ type: 'updateTrack', trackId: t.id, visible: !t.visible }])); visible.title = tr('studio.visibility', 'Toggle track visibility'); visible.setAttribute('aria-pressed', String(t.visible)); label.appendChild(visible); }
      const up = button('↑', () => change([{ type: 'reorderTrack', trackId: t.id, index: Math.max(0, index - 1) }])); up.title = tr('studio.trackUp', 'Move track earlier in composition'); up.disabled = index === 0; label.appendChild(up); trackRow.appendChild(label);
      const lane = el('div', 'ts-lane'); lane.dataset.trackId = t.id; lane.style.width = width + 'px'; lane.style.backgroundSize = S.zoom + 'px 100%';
      lane.addEventListener('click', e => { if (e.target === lane) seek(frames((e.clientX - lane.getBoundingClientRect().left) / S.zoom)); });
      lane.addEventListener('dragover', e => { if (Array.from(e.dataTransfer.types).includes('application/x-tensorsharp-asset')) e.preventDefault(); });
      lane.addEventListener('drop', e => { const assetId = e.dataTransfer.getData('application/x-tensorsharp-asset'); if (!assetId) return; e.preventDefault(); run(() => insertAsset(assetId, t.id, snapped(frames((e.clientX - lane.getBoundingClientRect().left) / S.zoom)))); });
      for (const c of S.project.clips.filter(c => c.trackId === t.id)) {
        const a = asset(c.assetId); const item = el('button', 'ts-clip', a ? a.name : tr('studio.missingAsset', 'Missing asset')); item.type = 'button'; item.dataset.kind = a ? a.kind : t.kind;
        item.style.left = seconds(c.startFrame) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(c.durationFrames) * S.zoom) + 'px';
        item.setAttribute('aria-pressed', String(S.selected === c.id)); item.title = (a ? a.name : c.id) + '\n' + tr('studio.clipPosition', 'Start {start}; duration {duration}', { start: fmt(c.startFrame), duration: fmt(c.durationFrames) }); item.setAttribute('aria-label', item.title);
        item.appendChild(el('span', '', seconds(c.durationFrames).toFixed(2) + 's')); attachDrag(item, c); lane.appendChild(item);
      }
      addPlayhead(lane); trackRow.appendChild(lane); scroll.appendChild(trackRow);
    }
    if (S.project.captions.length) {
      const captionRow = el('div', 'ts-track-row'); captionRow.appendChild(el('div', 'ts-track-label', tr('studio.captions', 'Captions'))); const lane = el('div', 'ts-lane'); lane.style.width = width + 'px';
      for (const c of S.project.captions) { const item = button(c.text, () => { S.selectedCaption = c.id; S.selected = null; S.tab = 'clip'; renderInspector(); }, 'ts-clip'); item.dataset.kind = 'text'; item.style.left = seconds(c.startFrame) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(c.durationFrames) * S.zoom) + 'px'; lane.appendChild(item); }
      addPlayhead(lane); captionRow.appendChild(lane); scroll.appendChild(captionRow);
    }
    if (!S.project.tracks.length) scroll.appendChild(el('p', 'ts-empty', tr('studio.emptyTimeline', 'Add video or audio tracks, then place media from the library.')));
    scroll.scrollLeft = scrollLeft; scroll.scrollTop = scrollTop;
  }
  function attachDrag(item, c) {
    let drag = null, suppressClick = false;
    item.addEventListener('click', e => { if (suppressClick) { suppressClick = false; e.preventDefault(); return; } selectClip(c.id); });
    item.addEventListener('pointerdown', e => {
      if (e.button !== 0 || S.busy) return; const bounds = item.getBoundingClientRect();
      drag = { pointer: e.pointerId, x: e.clientX, original: c.startFrame, start: c.startFrame, source: c.sourceStartFrame, duration: c.durationFrames,
        edge: e.clientX - bounds.left < 7 ? 'left' : bounds.right - e.clientX < 7 ? 'right' : 'move', changed: false, projectId: S.project.id, revision: S.project.revision };
      item.setPointerCapture(e.pointerId);
    });
    item.addEventListener('pointermove', e => {
      if (!drag || e.pointerId !== drag.pointer || Math.abs(e.clientX - drag.x) < 4 && !drag.changed) return;
      drag.changed = true; const delta = frames((e.clientX - drag.x) / S.zoom);
      if (drag.edge === 'move') { drag.start = snapped(drag.original + delta, c.id); item.style.left = seconds(drag.start) * S.zoom + 'px'; }
      else if (drag.edge === 'right') { drag.duration = Math.max(1, snapped(c.startFrame + c.durationFrames + delta, c.id) - c.startFrame); item.style.width = Math.max(14, seconds(drag.duration) * S.zoom) + 'px'; }
      else { const shift = Math.max(-c.sourceStartFrame, -c.startFrame, Math.min(c.durationFrames - 1, snapped(c.startFrame + delta, c.id) - c.startFrame)); drag.start = c.startFrame + shift; drag.source = c.sourceStartFrame + shift; drag.duration = c.durationFrames - shift; item.style.left = seconds(drag.start) * S.zoom + 'px'; item.style.width = Math.max(14, seconds(drag.duration) * S.zoom) + 'px'; }
    });
    item.addEventListener('pointerup', e => {
      if (!drag || e.pointerId !== drag.pointer) return; const finished = drag; drag = null; item.releasePointerCapture(e.pointerId);
      if (!finished.changed) return; suppressClick = true; S.selected = c.id;
      if (S.project.id !== finished.projectId || S.project.revision !== finished.revision) { renderTimeline(); status(tr('studio.dragStale', 'The project changed during the drag. Review and move the clip again.'), true); return; }
      run(async () => { try {
        const op = finished.edge === 'move' ? { type: 'moveClip', clipId: c.id, startFrame: finished.start } : { type: 'trimClip', clipId: c.id, sourceStartFrame: finished.source, durationFrames: finished.duration, startFrame: finished.start };
        await change([op]);
      } finally { renderTimeline(); } });
    });
    item.addEventListener('pointercancel', () => { drag = null; renderTimeline(); });
  }
  async function splitSelected() { const c = clip(); if (!c) return; if (S.playhead <= c.startFrame || S.playhead >= c.startFrame + c.durationFrames) throw new Error(tr('studio.splitInside', 'Place the playhead inside the selected clip before splitting.')); await change([{ type: 'splitClip', clipId: c.id, atFrame: S.playhead }]); }
  async function removeSelected() { const c = clip(); if (c) { await change([{ type: 'removeClip', clipId: c.id }]); S.selected = null; renderInspector(); } }
  function renderInspector() {
    if (!S.project) return; ui.inspector.replaceChildren();
    const tabs = el('div', 'ts-tabbar'); tabs.setAttribute('role', 'tablist');
    for (const pair of [['clip', tr('studio.properties', 'Properties')], ['generate', tr('studio.generate', 'Generate')], ['assistant', tr('studio.assistant', 'Assistant')], ['jobs', tr('studio.jobs', 'Jobs')]]) {
      const b = button(pair[1], () => { S.tab = pair[0]; renderInspector(); }); b.setAttribute('role', 'tab'); b.setAttribute('aria-selected', String(S.tab === pair[0])); b.setAttribute('aria-pressed', String(S.tab === pair[0])); tabs.appendChild(b);
    }
    ui.inspector.appendChild(tabs); const body = el('div', 'ts-stack'); ui.inspector.appendChild(body);
    if (S.tab === 'generate') generationForm(body); else if (S.tab === 'assistant') assistantForm(body); else if (S.tab === 'jobs') jobsPanel(body); else properties(body);
  }
  function properties(body) {
    const c = clip(), caption = S.project.captions.find(c => c.id === S.selectedCaption);
    if (c) {
      const a = asset(c.assetId); body.append(el('h3', '', a ? a.name : c.id), el('p', 'ts-hint', tr('studio.frameHint', 'All edit values are integer project frames. Drag a clip to move it; drag its edges to trim.')));
      const grid = el('div', 'ts-fields'); body.appendChild(grid);
      const start = num(grid, tr('studio.start', 'Timeline start (frames)'), c.startFrame), source = num(grid, tr('studio.sourceStart', 'Source in (frames)'), c.sourceStartFrame), duration = num(grid, tr('studio.duration', 'Duration (frames)'), c.durationFrames, 1);
      const track = field(grid, tr('studio.track', 'Track'), c.trackId, 'select', S.project.tracks.map(t => ({ value: t.id, label: t.name })));
      const volume = num(grid, tr('studio.volume', 'Volume multiplier'), c.volume, 0, 4, .05), fadeIn = num(grid, tr('studio.fadeIn', 'Fade in (frames)'), c.fadeInFrames), fadeOut = num(grid, tr('studio.fadeOut', 'Fade out (frames)'), c.fadeOutFrames);
      const audioFadeOut = num(grid, tr('studio.audioFadeOut', 'Audio-only fade out (frames)'), c.audioFadeOutFrames || 0);
      const fit = field(grid, tr('studio.fit', 'Image fit'), c.fit, 'select', [{ value: 'contain', label: tr('studio.contain', 'Fit inside') }, { value: 'cover', label: tr('studio.cover', 'Fill / crop') }]);
      const linked = checked(body, tr('studio.linked', 'Apply timing edits to linked clips'), true);
      body.appendChild(button(tr('studio.apply', 'Apply changes'), () => change([
        { type: 'moveClip', clipId: c.id, trackId: track.value, startFrame: Number(start.value), applyToLinked: linked.checked },
        { type: 'trimClip', clipId: c.id, sourceStartFrame: Number(source.value), durationFrames: Number(duration.value), applyToLinked: linked.checked },
        { type: 'updateClip', clipId: c.id, volume: Number(volume.value), fadeInFrames: Number(fadeIn.value), fadeOutFrames: Number(fadeOut.value), audioFadeOutFrames: Number(audioFadeOut.value), fit: fit.value }
      ]), 'ts-primary'));
      const controls = row(body); controls.append(button(tr('studio.split', 'Split'), splitSelected), button(tr('studio.duplicate', 'Duplicate'), () => change([{ type: 'duplicateClip', clipId: c.id }])), button(tr('studio.remove', 'Remove'), removeSelected, 'ts-danger'));
      const previous = S.project.clips.find(other => other.id !== c.id && other.trackId === c.trackId && other.startFrame < c.startFrame && other.startFrame + other.durationFrames > c.startFrame && other.startFrame + other.durationFrames < c.startFrame + c.durationFrames);
      if (previous && (S.project.tracks.find(t => t.id === c.trackId) || {}).kind === 'video') {
        body.appendChild(button(tr('studio.dissolve', 'Dissolve from overlapping previous clip'), () => change([{ type: 'dissolve', clipId: c.id, otherClipId: previous.id }])));
        body.appendChild(el('p', 'ts-hint', tr('studio.dissolveHelp', 'The overlap becomes a picture and audio dissolve. Clip positions stay fixed. Fade controls remain editable.')));
      }
      const linkGroup = field(body, tr('studio.linkGroup', 'Linked edit group (optional)'), c.linkGroup || '');
      body.appendChild(button(tr('studio.saveLink', 'Set link group'), () => change([{ type: 'updateClip', clipId: c.id, linkGroup: linkGroup.value }])));
      body.appendChild(el('p', 'ts-hint', tr('studio.takeHint', 'Select another library asset’s “Use take” to replace this clip while preserving its placement. The original stays in the library.')));
      if (a && a.generation) { const details = el('details'); details.append(el('summary', '', tr('studio.provenance', 'Generation provenance')), el('pre', 'ts-provenance', JSON.stringify(a.generation, null, 2))); body.appendChild(details); }
    } else if (caption) captionForm(body, caption);
    else body.append(el('h3', '', tr('studio.noSelection', 'Project settings')), el('p', 'ts-hint', tr('studio.selectHint', 'Select a clip on the timeline to edit its timing, sound, or framing.')));
    const settings = el('details'); if (!c && !caption) settings.open = true; settings.appendChild(el('summary', '', tr('studio.projectSettings', 'Project and export settings'))); body.appendChild(settings);
    const name = field(settings, tr('studio.projectName', 'Name'), S.project.name); const size = el('div', 'ts-fields'); settings.appendChild(size); const width = num(size, tr('studio.width', 'Width'), S.project.width, 16, 4096), height = num(size, tr('studio.height', 'Height'), S.project.height, 16, 4096);
    settings.appendChild(button(tr('studio.saveSettings', 'Save settings'), () => change([{ type: 'settings', name: name.value, width: Number(width.value), height: Number(height.value) }])));
    settings.appendChild(button(tr('studio.vertical', 'Vertical 720 × 1280'), () => change([{ type: 'settings', width: 720, height: 1280 }])));
    settings.appendChild(el('p', 'ts-hint', tr('studio.compositionOrder', 'Tracks compose in list order, bottom layer first. Fades affect picture and sound. Timeline audio is mixed at export.')));
    const captionDetails = el('details'); captionDetails.appendChild(el('summary', '', tr('studio.addCaption', 'Add caption / title'))); captionForm(captionDetails, null); body.appendChild(captionDetails);
    if (S.capabilities && S.capabilities.packageSupported) {
      const packages = el('details'); packages.appendChild(el('summary', '', tr('studio.package', 'Portable project package')));
      packages.appendChild(button(tr('studio.packageExport', 'Download project package'), exportPackage));
      const input = el('input'); input.type = 'file'; input.accept = '.zip,.tsstudio'; input.setAttribute('aria-label', tr('studio.packageImport', 'Import project package'));
      input.addEventListener('change', () => run(async () => { if (input.files[0]) await importPackage(input.files[0]); })); packages.appendChild(input); body.appendChild(packages);
    }
  }
  function captionForm(body, existing) {
    if (S.capabilities && S.capabilities.captionAvailable === false) { body.appendChild(el('p', 'ts-hint', tr('studio.captionUnavailable', 'Caption rendering is unavailable on this host.'))); return; }
    const text = field(body, tr('studio.captionText', 'Caption text'), existing ? existing.text : '', 'textarea'); text.maxLength = 500;
    const grid = el('div', 'ts-fields'); body.appendChild(grid); const start = num(grid, tr('studio.start', 'Timeline start (frames)'), existing ? existing.startFrame : S.playhead), duration = num(grid, tr('studio.duration', 'Duration (frames)'), existing ? existing.durationFrames : frames(3), 1);
    if (S.capabilities && S.capabilities.captionCharacterSet) body.appendChild(el('p', 'ts-hint', String(S.capabilities.captionCharacterSet)));
    body.appendChild(button(existing ? tr('studio.saveCaption', 'Save caption') : tr('studio.addCaption', 'Add caption / title'), () => change([{ type: existing ? 'updateCaption' : 'addCaption', captionId: existing ? existing.id : id(), text: text.value, startFrame: Number(start.value), durationFrames: Number(duration.value) }])));
    if (existing) body.appendChild(button(tr('studio.deleteCaption', 'Remove caption'), async () => { await change([{ type: 'removeCaption', captionId: existing.id }]); S.selectedCaption = null; renderInspector(); }));
    body.appendChild(el('p', 'ts-hint', tr('studio.transcriptionUnavailable', 'Automatic transcription and word-based cuts are not yet available in Studio. Captions remain manually editable.')));
  }
  async function exportPackage() {
    const a = el('a'); a.href = API + '/projects/' + encodeURIComponent(S.project.id) + '/package'; a.download = S.project.name + '.tsstudio'; document.body.appendChild(a); a.click(); a.remove();
  }
  async function importPackage(file) {
    const value = await response(await fetch(API + '/packages/import?fileName=' + encodeURIComponent(file.name), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/octet-stream', 'X-TensorSharp-Studio': '1' }, body: file }));
    receiveProject(value); S.selected = null; resetPreview(); renderProject(); await listProjects(); rememberProject();
  }
  function generationForm(body) {
    const models = S.capabilities && S.capabilities.models || [];
    body.appendChild(el('h3', '', tr('studio.generateTitle', 'Generate a new take')));
    body.appendChild(el('p', 'ts-hint', tr('studio.generateHelp', 'Native generation runs on this host. New takes remain in the library until you add or accept them.')));
    body.appendChild(button(tr('studio.refreshModels', 'Refresh model capabilities'), refreshCapabilities));
    if (!models.length) { body.appendChild(el('p', 'ts-empty', tr('studio.noModel', 'No video model capabilities are available. Load a supported video model in the host to generate footage. Imported-media editing remains available.'))); return; }
    const previousAsset = S.regenerateAsset && asset(S.regenerateAsset); const previous = previousAsset && previousAsset.generation && previousAsset.generation.request;
    const selectedModel = models.find(m => m.available && previous && m.modelId === previous.modelId) || models.find(m => m.available) || models[0];
    const modelSelect = field(body, tr('studio.model', 'Video model'), selectedModel.modelId, 'select', models.map(m => ({ value: m.modelId, label: m.displayName + (m.available ? '' : ' · ' + tr('studio.unavailable', 'unavailable')) })));
    const form = el('div', 'ts-stack'); body.appendChild(form);
    function paintModel() {
      form.replaceChildren(); const m = models.find(m => m.modelId === modelSelect.value); if (!m) return;
      if (!m.available) { form.appendChild(el('p', 'ts-empty', m.unavailableReason || tr('studio.notLoaded', 'Load this model in the host before generating.'))); return; }
      const reuse = previous && previous.modelId === m.modelId ? previous : null;
      // Generation geometry belongs to the loaded model/host, not the timeline canvas.
      // LTX plans motion at half the output size, so a generic 512px default loses face/hand detail.
      const defaults = { width: m.defaultWidth || 512, height: m.defaultHeight || 512,
        frameCount: m.defaultFrameCount, fps: m.fixedFps || m.defaultFps || 24,
        steps: m.fixedSteps == null ? m.defaultSteps : m.fixedSteps,
        cfgScale: m.fixedCfgScale == null ? m.defaultCfgScale : m.fixedCfgScale, generateAudio: !!m.defaultGenerateAudio };
      const prompt = field(form, tr('studio.prompt', 'Describe this shot'), reuse ? reuse.prompt : '', 'textarea'); prompt.maxLength = 12000;
      const basic = el('div', 'ts-fields'); form.appendChild(basic);
      const width = num(basic, tr('studio.width', 'Width'), reuse ? reuse.width : defaults.width, m.spatialMultiple, m.maxDimension, m.spatialMultiple);
      const height = num(basic, tr('studio.height', 'Height'), reuse ? reuse.height : defaults.height, m.spatialMultiple, m.maxDimension, m.spatialMultiple);
      const count = num(basic, tr('studio.frames', 'Output frames'), reuse ? reuse.frameCount : defaults.frameCount, m.temporalOffset || m.temporalMultiple, m.maxFrames, m.temporalMultiple);
      const rate = num(basic, tr('studio.fps', 'Frames per second'), reuse ? reuse.fps : defaults.fps, 1, 120); if (m.fixedFps) { rate.value = m.fixedFps; rate.disabled = true; }
      const duration = el('p', 'ts-hint'); const updateDuration = () => { duration.textContent = tr('studio.effectiveDuration', '{seconds}s · {frames} frames at {fps} fps. Size uses multiples of {grid}.', { seconds: (Number(count.value) / Number(rate.value)).toFixed(2), frames: count.value, fps: rate.value, grid: m.spatialMultiple }); };
      count.addEventListener('input', updateDuration); rate.addEventListener('input', updateDuration); updateDuration(); form.appendChild(duration);
      let condition = null, last = null;
      const imageOptions = [{ value: '', label: tr('studio.none', 'None') }].concat(S.project.assets.filter(a => a.kind === 'image' && !a.missing).map(a => ({ value: a.id, label: a.name })));
      if (m.imageConditioning) condition = field(form, tr('studio.firstFrame', 'Image guidance / first frame'), reuse ? reuse.conditioningAssetId || '' : '', 'select', imageOptions);
      if (m.lastFrameConditioning) last = field(form, tr('studio.lastFrame', 'Last-frame image'), reuse ? reuse.endConditioningAssetId || '' : '', 'select', imageOptions);
      if (m.imageConditioning) form.appendChild(el('p', 'ts-hint', tr('studio.temporalGuidance', 'Image guidance conditions frames in time. It does not promise identity consistency.')));
      let audio = null; if (m.audioOutput) audio = checked(form, tr('studio.audioGenerate', 'Generate audio'), reuse ? !!reuse.generateAudio : defaults.generateAudio);
      const advanced = el('details'); advanced.appendChild(el('summary', '', tr('studio.advanced', 'Advanced settings'))); form.appendChild(advanced);
      const negative = field(advanced, tr('studio.negativePrompt', 'Negative prompt'), reuse ? reuse.negativePrompt || '' : '', 'textarea'); negative.maxLength = 12000;
      const advancedFields = el('div', 'ts-fields'); advanced.appendChild(advancedFields);
      const steps = num(advancedFields, tr('studio.steps', 'Sampling steps'), reuse ? reuse.steps : m.defaultSteps, 1, 200);
      const cfg = num(advancedFields, tr('studio.cfg', 'Guidance scale'), reuse ? reuse.cfgScale : m.defaultCfgScale, 0, 30, .1);
      if (m.fixedSteps != null) { steps.value = m.fixedSteps; steps.disabled = true; }
      if (m.fixedCfgScale != null) { cfg.value = m.fixedCfgScale; cfg.disabled = true; }
      const seed = num(advanced, tr('studio.seed', 'Seed (−1 = random)'), reuse ? reuse.seed : -1, -1, Number.MAX_SAFE_INTEGER);
      const temporalRows = [], references = [];
      if (m.temporalGuidance) {
        advanced.appendChild(el('h3', '', tr('studio.temporalKeyframes', 'Temporal image keyframes')));
        advanced.appendChild(el('p', 'ts-hint', tr('studio.keyframesHelp', 'Each image guides a specific output frame. This is separate from identity references.')));
        const guides = el('div', 'ts-stack'); advanced.appendChild(guides);
        function addGuide(initial) {
          if (temporalRows.length >= 16) throw new Error(tr('studio.keyframeLimit', 'At most 16 temporal image guides can be submitted.'));
          const group = el('div', 'ts-asset'); const assetInput = field(group, tr('studio.guideImage', 'Guide image'), initial && initial.assetId || '', 'select', imageOptions);
          const grid = el('div', 'ts-fields'); group.appendChild(grid); const at = num(grid, tr('studio.guideFrame', 'Output frame index'), initial ? initial.frameIndex : 0, 0, m.maxFrames - 1), strength = num(grid, tr('studio.guideStrength', 'Strength'), initial ? initial.strength : 1, 0, 1, .05);
          const entry = { group, assetInput, at, strength }; temporalRows.push(entry); group.appendChild(button(tr('studio.removeGuide', 'Remove guide'), () => { temporalRows.splice(temporalRows.indexOf(entry), 1); group.remove(); })); guides.appendChild(group);
        }
        for (const guide of reuse && reuse.temporalConditions || []) addGuide(guide);
        advanced.appendChild(button(tr('studio.addGuide', 'Add image keyframe'), () => addGuide(null)));
      }
      if (m.identityReferences) {
        advanced.appendChild(el('h3', '', tr('studio.identityReferences', 'Identity-reference images')));
        advanced.appendChild(el('p', 'ts-hint', tr('studio.referencesHelp', 'Select up to {count} model-supported references. Consistency depends on the loaded model.', { count: m.maxReferences })));
        for (const image of S.project.assets.filter(a => a.kind === 'image' && !a.missing)) references.push({ id: image.id, input: checked(advanced, image.name, !!(reuse && (reuse.referenceAssetIds || []).includes(image.id))) });
      }
      form.appendChild(button(tr('studio.useModelDefaults', 'Use model defaults ({width} × {height})', { width: defaults.width, height: defaults.height }), () => {
        width.value = defaults.width; height.value = defaults.height; count.value = defaults.frameCount; rate.value = defaults.fps;
        steps.value = defaults.steps; cfg.value = defaults.cfgScale; if (audio) audio.checked = defaults.generateAudio; updateDuration();
      }));
      form.appendChild(el('p', 'ts-hint', tr('studio.quality', 'These settings are sent as shown. No automatic quality reduction or provider substitution.')));
      if (m.deferredCapabilities && m.deferredCapabilities.length) form.appendChild(el('p', 'ts-hint', tr('studio.unsupportedCapabilities', 'Unavailable in Studio: {list}', { list: m.deferredCapabilities.join(', ') })));
      const generate = button(tr('studio.generateTake', 'Generate take'), async () => {
        if (!prompt.value.trim()) throw new Error(tr('studio.promptRequired', 'Describe the shot before generating.'));
        for (const input of [width, height, count, rate, steps, cfg, seed]) if (!input.checkValidity()) { input.reportValidity(); return; }
        const temporal = (Number(count.value) - (m.temporalOffset || 0)) % m.temporalMultiple;
        if (temporal !== 0) throw new Error(tr('studio.temporalGrid', 'Frame count must be {multiple} × n + {offset}.', { multiple: m.temporalMultiple, offset: m.temporalOffset || 0 }));
        if (!m.textToVideo && !(condition && condition.value)) throw new Error(tr('studio.conditionRequired', 'This model requires an image-conditioning asset.'));
        const request = { projectId: S.project.id, requestId: id(), modelId: m.modelId, prompt: prompt.value.trim(), negativePrompt: negative.value || null,
          width: Number(width.value), height: Number(height.value), frameCount: Number(count.value), fps: Number(rate.value), steps: Number(steps.value), cfgScale: Number(cfg.value), seed: Number(seed.value),
          generateAudio: !!(audio && audio.checked), conditioningAssetId: condition && condition.value || null, endConditioningAssetId: last && last.value || null, shotId: reuse && reuse.shotId || id() };
        request.temporalConditions = temporalRows.map(r => ({ assetId: r.assetInput.value, frameIndex: Number(r.at.value), strength: Number(r.strength.value) }));
        request.referenceAssetIds = references.filter(r => r.input.checked).map(r => r.id);
        if (request.referenceAssetIds.length > (m.maxReferences || 0)) throw new Error(tr('studio.referenceLimit', 'This model accepts at most {count} reference images.', { count: m.maxReferences || 0 }));
        if (request.temporalConditions.some(c => !c.assetId || !Number.isInteger(c.frameIndex) || c.frameIndex < 0 || c.frameIndex >= request.frameCount || c.strength < 0 || c.strength > 1)) throw new Error(tr('studio.invalidGuides', 'Each temporal guide needs an image, an output frame within the shot, and a strength between 0 and 1.'));
        generate.disabled = true;
        try { await api('jobs/generate', request); S.tab = 'jobs'; S.regenerateAsset = null; await refreshJobs(); renderInspector(); status(tr('studio.generationQueued', 'Generation queued. The finished take will appear in the library.')); }
        finally { generate.disabled = false; }
      }, 'ts-primary'); form.appendChild(generate);
    }
    modelSelect.addEventListener('change', paintModel); paintModel();
  }
  const assistantState = new Map();
  async function refreshCapabilities() { S.capabilities = await api('capabilities'); renderInspector(); previewNotice(); renderLibrary(); }
  function boundedMessages(value) {
    return (Array.isArray(value) ? value : []).filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string').slice(-8).map(m => ({ role: m.role, content: m.content.slice(0, 12000) }));
  }
  function assistantFor(projectId) {
    if (!assistantState.has(projectId)) {
      let messages = []; try { messages = boundedMessages(JSON.parse(window.localStorage.getItem('tensorsharp.studio.chat.' + projectId) || '[]')); } catch (_) { }
      assistantState.set(projectId, { messages, text: messages.filter(m => m.role === 'assistant').at(-1)?.content || '', busy: false, controller: null, error: null, sessionId: null, conversationId: null });
    }
    return assistantState.get(projectId);
  }
  async function restoreAssistantHistory(projectId, conversationId, state) {
    if (!conversationId || state.historyLoaded || state.busy) return;
    if (state.historyLoad) return state.historyLoad;
    state.historyLoad = (async () => {
      try {
        const saved = await response(await fetch('/api/agent/conversations/' + encodeURIComponent(conversationId), { credentials: 'same-origin' }));
        state.messages = boundedMessages(saved.messages); state.text = state.messages.filter(m => m.role === 'assistant').at(-1)?.content || '';
      } catch (e) { state.error = e.message; }
      finally { state.historyLoaded = true; state.historyLoad = null; if (S.project && S.project.id === projectId && S.tab === 'assistant') renderInspector(); }
    })();
    return state.historyLoad;
  }
  async function assistantSession(projectId, state, expectedRevision) {
    if (state.sessionId) {
      if (state.conversationId && S.project.conversationId !== state.conversationId) {
        if (S.busy || S.project.revision !== expectedRevision) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
        await change([{ type: 'settings', conversationId: state.conversationId }]); return S.project.revision;
      }
      return expectedRevision;
    }
    const conversation = S.project.conversationId || 'new';
    const created = await response(await fetch('/api/sessions?conversation=' + encodeURIComponent(conversation), { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: '{}' }));
    if (!created.sessionId) throw new Error(tr('studio.sessionUnavailable', 'The host could not create a dedicated Studio chat session.'));
    state.sessionId = created.sessionId; state.conversationId = created.conversationId || null;
    if (state.conversationId && created.messages) { state.messages = boundedMessages(created.messages); state.historyLoaded = true; }
    if (!S.project || S.project.id !== projectId) throw new Error(tr('studio.projectChanged', 'The active project changed. The assistant request was not sent.'));
    if (S.busy || S.project.revision !== expectedRevision) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
    if (state.conversationId && S.project.conversationId !== state.conversationId) await change([{ type: 'settings', conversationId: state.conversationId }]);
    return S.project.revision;
  }
  function assistantForm(body) {
    const state = assistantFor(S.project.id); body.appendChild(el('h3', '', tr('studio.assistantTitle', 'Edit with the assistant')));
    if (S.project.conversationId && !state.historyLoaded && !state.busy) restoreAssistantHistory(S.project.id, S.project.conversationId, state);
    const selected = clip(); body.appendChild(el('p', 'ts-hint', tr('studio.agentContext', 'Project {project} · revision {revision} · selection: {selection}', { project: S.project.name, revision: S.project.revision, selection: selected ? (asset(selected.assetId) || {}).name || selected.id : tr('studio.none', 'None') })));
    body.appendChild(el('p', 'ts-hint', tr('studio.agentHelp', 'The loaded chat model inspects this project and proposes edits. Review before applying. Edits use the same commands as the timeline.')));
    const request = field(body, tr('studio.instruction', 'Editing instruction'), state.draft || '', 'textarea'); request.maxLength = 8000; request.placeholder = tr('studio.agentExample', 'Trim the selected clip to its first four seconds.'); request.disabled = state.busy || !!state.historyLoad;
    request.addEventListener('input', () => { state.draft = request.value; });
    const actions = row(body); const send = button(tr('studio.propose', 'Propose edits'), () => askAssistant(request.value)); send.disabled = state.busy || !!state.historyLoad; actions.appendChild(send);
    if (state.busy) actions.appendChild(button(tr('studio.stop', 'Stop'), () => { if (state.controller) state.controller.abort(); }));
    actions.appendChild(button(tr('studio.refreshProposals', 'Refresh proposals'), loadProposals));
    if (state.messages.length) { const history = el('details'); history.appendChild(el('summary', '', tr('studio.assistantHistory', 'Recent conversation'))); for (const message of state.messages) history.appendChild(el('p', 'ts-provenance', message.role + ': ' + message.content)); body.appendChild(history); }
    body.appendChild(el('p', 'ts-hint', tr('studio.assistantPersistence', 'TensorAgent uses its saved conversation store. Other hosts retain up to eight recent messages in this browser; project packages exclude chat history.')));
    if (state.text) body.appendChild(el('p', 'ts-provenance', state.text)); if (state.error) body.appendChild(el('p', 'ts-hint', state.error));
    const proposal = S.proposal;
    if (proposal) {
      const card = el('div', 'ts-proposal'); card.append(el('strong', '', proposal.summary || tr('studio.proposedEdit', 'Proposed edit')), el('p', 'ts-hint', tr('studio.proposalRevision', 'Based on revision {revision} · {count} operations', { revision: proposal.expectedRevision, count: proposal.commands.length })));
      const details = el('details'); details.append(el('summary', '', tr('studio.inspectCommands', 'Inspect exact commands')), el('pre', '', JSON.stringify(proposal.commands, null, 2))); card.appendChild(details);
      const stale = proposal.expectedRevision !== S.project.revision;
      if (stale) card.appendChild(el('p', 'ts-hint', tr('studio.staleProposal', 'This proposal is stale. Ask for a new proposal using the current revision.')));
      const apply = button(tr('studio.applyProposal', 'Apply proposal'), async () => {
        receiveProject(await api('proposals/apply', { projectId: S.project.id, proposalId: proposal.id, commandId: id() })); S.proposal = null; renderProject(); status(tr('studio.proposalApplied', 'Proposal applied and saved as one undoable transaction.'));
      }, 'ts-primary'); apply.disabled = stale; card.appendChild(apply);
      card.appendChild(button(tr('studio.dismissProposal', 'Dismiss'), () => { S.proposal = null; renderInspector(); })); body.appendChild(card);
    }
    if (S.generationDrafts.length) {
      body.appendChild(el('h3', '', tr('studio.plannedShots', 'Planned shots')));
      body.appendChild(el('p', 'ts-hint', tr('studio.plannedShotsHelp', 'These are saved plans, not generated footage. Load a video model through the host’s model selector, refresh capabilities, and explicitly start each shot.')));
      body.appendChild(button(tr('studio.refreshModels', 'Refresh model capabilities'), refreshCapabilities));
      for (const draft of S.generationDrafts) {
        const card = el('div', 'ts-proposal'); card.appendChild(el('strong', '', draft.summary || tr('studio.plannedShot', 'Planned shot')));
        card.appendChild(el('pre', 'ts-provenance', JSON.stringify(draft.request, null, 2)));
        const available = (S.capabilities && S.capabilities.models || []).filter(m => m.available);
        if (draft.jobId) card.appendChild(el('p', 'ts-hint', tr('studio.shotStarted', 'Generation started. Track its result in Jobs.')));
        else if (!available.length) card.appendChild(el('p', 'ts-hint', tr('studio.shotNeedsModel', 'No video model is loaded. The plan is saved for later.')));
        else {
          const model = field(card, tr('studio.runModel', 'Run with video model'), available[0].modelId, 'select', available.map(m => ({ value: m.modelId, label: m.displayName })));
          card.appendChild(button(tr('studio.runPlannedShot', 'Generate planned shot'), async () => {
            await api('generation-drafts/start', { projectId: S.project.id, draftId: draft.id, expectedRevision: S.project.revision, modelId: model.value });
            await loadProposals(); await refreshJobs(); status(tr('studio.plannedShotQueued', 'The planned shot is queued. Its requested settings were validated without automatic quality reduction.'));
          }, 'ts-primary'));
        }
        body.appendChild(card);
      }
    }
  }
  async function askAssistant(instruction) {
    if (!instruction.trim()) throw new Error(tr('studio.instructionRequired', 'Describe the edit you want.'));
    const projectId = S.project.id, initialRevision = S.project.revision, selectedClipIds = S.selected ? [S.selected] : [];
    const state = assistantFor(projectId); if (state.busy) return; state.busy = true; state.text = ''; state.error = null; state.controller = new AbortController();
    renderInspector();
    try {
      const preparedRevision = await assistantSession(projectId, state, initialRevision);
      if (!S.project || S.project.id !== projectId) throw new Error(tr('studio.projectChanged', 'The active project changed. The assistant request was not sent.'));
      if (S.project.revision !== preparedRevision || (S.selected || null) !== (selectedClipIds[0] || null)) throw new Error(tr('studio.contextChanged', 'The project changed while preparing the assistant. Review your selection and send again.'));
      const expectedRevision = S.project.revision, messages = boundedMessages(state.messages).concat([{ role: 'user', content: instruction.trim().slice(0, 8000) }]);
      const res = await fetch('/api/chat', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-TensorSharp-Studio': '1' }, body: JSON.stringify({ sessionId: state.sessionId, messages, maxTokens: 2048, studio: { projectId, expectedRevision, selectedClipIds } }), signal: state.controller.signal });
      if (res.status === 404) state.sessionId = null;
      if (!res.ok) { await response(res); return; }
      if (!res.body) throw new Error(tr('studio.streamUnavailable', 'The host did not provide an assistant response stream.'));
      const reader = res.body.getReader(), decoder = new TextDecoder(); let buffer = '', complete = false;
      try {
        while (!complete) {
          const chunk = await reader.read(); if (chunk.done) break; buffer += decoder.decode(chunk.value, { stream: true });
          if (buffer.length > 512000) throw new Error(tr('studio.responseTooLarge', 'Assistant response exceeded the bounded context limit.'));
          let end;
          while ((end = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, end).trimEnd(); buffer = buffer.slice(end + 1); if (!line.startsWith('data:')) continue;
            const data = line.slice(5).trim(); if (!data || data === '[DONE]') continue; let event; try { event = JSON.parse(data); } catch (_) { continue; }
            if (event.error) throw new Error(typeof event.error === 'string' ? event.error : event.error.message || tr('studio.agentError', 'Assistant request failed.'));
            if (event.token && state.text.length < 64000) state.text += String(event.token).slice(0, 64000 - state.text.length);
            if (event.done) complete = true;
          }
          if (S.project && S.project.id === projectId && S.tab === 'assistant') renderInspector();
        }
      } finally { try { await reader.cancel(); } catch (_) { } reader.releaseLock(); }
      state.messages = boundedMessages(messages.concat(state.text ? [{ role: 'assistant', content: state.text.slice(0, 12000) }] : []));
      if (!state.conversationId) { try { window.localStorage.setItem('tensorsharp.studio.chat.' + projectId, JSON.stringify(state.messages)); } catch (_) { } }
      if (S.project && S.project.id === projectId) await loadProposals();
    } catch (e) { state.error = e.name === 'AbortError' ? tr('studio.agentStopped', 'Assistant stopped. Any saved proposal remains available for review.') : e.message; }
    finally { state.busy = false; state.controller = null; if (S.project && S.project.id === projectId && S.tab === 'assistant') renderInspector(); }
  }
  async function loadProposals() {
    const projectId = S.project.id; const [value, drafts] = await Promise.all([api('proposals/list', { projectId }), api('generation-drafts/list', { projectId })]); if (S.project.id !== projectId) return;
    const proposals = Array.isArray(value) ? value : value.proposals || []; S.proposal = proposals.sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))[0] || null;
    S.generationDrafts = Array.isArray(drafts) ? drafts : drafts.drafts || [];
    if (S.tab === 'assistant') renderInspector();
  }
  function jobsPanel(body) {
    body.append(el('h3', '', tr('studio.jobsTitle', 'Background jobs')), el('p', 'ts-hint', tr('studio.jobPersistence', 'Jobs belong to the host. Closing this panel does not cancel them. Interrupted work is reported after restart.')));
    if (!renderAllowed()) body.appendChild(el('p', 'ts-hint', (S.capabilities || {}).renderUnavailableReason || tr('studio.renderUnsupported', 'Local rendering is unavailable on this host.')));
    if (!S.jobs.length) body.appendChild(el('p', 'ts-empty', tr('studio.noJobs', 'No generation or render jobs yet.')));
    for (const job of [...S.jobs].sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))) {
      const card = el('article', 'ts-job'); card.append(el('strong', '', job.kind + ' · ' + job.state), el('p', 'ts-hint', job.stage + ' · ' + tr('studio.revision', 'revision {revision}', { revision: job.projectRevision })));
      if (['queued', 'running', 'cancelling'].includes(job.state)) {
        const progress = el('progress'); progress.max = 1; if (Number.isFinite(job.progress)) progress.value = job.progress; progress.setAttribute('aria-label', tr('studio.jobProgress', 'Reported job progress')); card.appendChild(progress);
        const cancel = button(job.state === 'cancelling' ? tr('studio.cancelling', 'Cancelling…') : tr('studio.cancel', 'Cancel'), async () => { await api('jobs/cancel', { jobId: job.id }); await refreshJobs(); }); cancel.disabled = job.state === 'cancelling'; card.appendChild(cancel);
      }
      if (job.error) card.appendChild(el('p', 'ts-hint', job.error));
      if (job.state === 'interrupted') card.appendChild(el('p', 'ts-hint', tr('studio.interrupted', 'Restart the operation from its settings. Mid-generation checkpoint recovery is unavailable.')));
      if (job.state === 'completed' && job.kind === 'render') {
        const actions = row(card); actions.appendChild(button(tr('studio.playRender', 'Play rendered revision'), () => showRender(job))); const download = el('a', '', tr('studio.download', 'Download MP4')); download.href = outputUrl(job); download.download = S.project.name + '-r' + job.projectRevision + '.mp4'; actions.appendChild(download);
      }
      if (job.outputAssetId) { const a = asset(job.outputAssetId); if (a) card.appendChild(button(tr('studio.addGenerated', 'Add take to timeline'), () => insertAsset(a.id))); }
      body.appendChild(card);
    }
  }
  function keydown(e) {
    if (S.deleteDialog) return;
    if (e.key === 'Escape') { e.preventDefault(); close(); return; }
    if (e.key === 'Tab') {
      const focusable = Array.from(S.root.querySelectorAll('button, input, select, textarea, a[href], summary')).filter(n => !n.disabled && n.getClientRects().length);
      if (focusable.length && e.shiftKey && document.activeElement === focusable[0]) { e.preventDefault(); focusable[focusable.length - 1].focus(); }
      else if (focusable.length && !e.shiftKey && document.activeElement === focusable[focusable.length - 1]) { e.preventDefault(); focusable[0].focus(); } return;
    }
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); run(() => history(e.shiftKey ? 'redo' : 'undo')); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') { e.preventDefault(); if (clip()) run(() => change([{ type: 'duplicateClip', clipId: S.selected }])); }
    else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); run(removeSelected); }
    else if (e.key.toLowerCase() === 's' && !e.ctrlKey && !e.metaKey) { e.preventDefault(); run(splitSelected); }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); seek(S.playhead + (e.key === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 10 : 1)); }
    else if (e.code === 'Space' && ui.video && ui.video.src) { e.preventDefault(); if (ui.video.paused) run(() => ui.video.play()); else ui.video.pause(); }
  }
  function localMediaUrl(value) {
    try { const url = new URL(value, window.location.href); if (url.origin !== window.location.origin || !/^\/(?:uploads\/|api\/uploads\/|api\/code\/artifacts\/)/.test(url.pathname)) return null; return url.pathname + url.search; } catch (_) { return null; }
  }
  function attachChatActions() {
    const chat = document.getElementById('chat-container') || document.getElementById('chat'); if (!chat) return;
    function scan() {
      for (const media of chat.querySelectorAll('video')) {
        if (media.dataset.studioAttached) continue; const source = media.currentSrc || media.getAttribute('src') || (media.querySelector('source') || {}).src;
        const url = localMediaUrl(source); if (!url) continue; media.dataset.studioAttached = 'true';
        const action = button(tr('studio.openMedia', 'Open in Studio'), () => { media.pause(); return open({ url }); }, 'ts-studio-chat-action'); media.insertAdjacentElement('afterend', action);
      }
    }
    scan(); const observer = new MutationObserver(scan); observer.observe(chat, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });
  }
  function init() {
    const controls = document.querySelector('.header-controls') || document.querySelector('.drawerfoot');
    if (controls && !document.getElementById('ts-studio-open')) { const entry = button(tr('studio.title', 'Video Studio'), () => open(), 'ts-studio-open' + (controls.classList.contains('drawerfoot') ? ' opt' : '')); entry.id = 'ts-studio-open'; controls.prepend(entry); }
    attachChatActions();
    if (window.location.hash === '#studio') run(() => open());
  }
  window.TensorSharpStudio = Object.freeze({ open: options => run(() => open(options)), close });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
