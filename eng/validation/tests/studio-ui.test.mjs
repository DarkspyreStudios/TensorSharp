// Shared Studio client contract tests. These simulate the host contract; they do not
// constitute a browser, real-device, media-rendering, or inference validation.
import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const source = readFileSync(fileURLToPath(new URL('../../../TensorSharp.Chat/WebUi/studio.js', import.meta.url)), 'utf8');
function simulatedDocument() {
  const document = { readyState: 'loading', addEventListener: () => {}, activeElement: null };
  class Element {
    constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.listeners = {}; this.attributes = {}; this.dataset = {}; this.style = {}; this.className = ''; this.textContent = ''; this.inert = false; }
    append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } }
    appendChild(node) { this.append(node); return node; }
    replaceChildren(...nodes) { for (const node of this.children) node.parentNode = null; this.children = []; this.append(...nodes); }
    remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(node => node !== this); this.parentNode = null; }
    setAttribute(key, value) { this.attributes[key] = value; }
    addEventListener(event, handler) { (this.listeners[event] ||= []).push(handler); }
    dispatch(event, properties = {}) { const e = { target: this, preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.propagationStopped = true; }, ...properties }; return { event: e, results: (this.listeners[event] || []).map(handler => handler(e)) }; }
    click() { return Promise.all(this.dispatch('click').results); }
    checkValidity() { return true; }
    focus() { document.activeElement = this; }
    get isConnected() { return this === document.body || !!this.parentNode?.isConnected; }
    querySelectorAll(selector) { const matches = node => selector.startsWith('.') ? node.className.split(' ').includes(selector.slice(1)) : node.tagName === selector.toUpperCase(); return this.children.flatMap(node => [...(matches(node) ? [node] : []), ...node.querySelectorAll(selector)]); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  }
  document.createElement = tag => new Element(tag); document.createTextNode = value => Object.assign(new Element('#text'), { textContent: value }); document.body = new Element('body');
  return document;
}
function harness() {
  const calls = [], handlers = [];
  const document = simulatedDocument();
  const context = vm.createContext({
    console, URL, Map, Set, Date, Math, JSON, Promise, Error, Object, Array, String, Number, TextDecoder, AbortController,
    setTimeout, clearTimeout,
    window: { location: { href: 'http://127.0.0.1:5000/', origin: 'http://127.0.0.1:5000' }, crypto: { randomUUID: () => '11111111-2222-4333-8444-555555555555' } },
    document,
    fetch: async (url, options) => {
      calls.push({ url, options, body: options?.body && JSON.parse(options.body) });
      const next = handlers.shift(); if (!next) throw new Error('Unexpected network call: ' + url);
      return next;
    }
  });
  // Expose private helpers only in this VM, leaving the production public API small.
  const instrumented = source.replace('  function renderProject() {', '  function renderProject() { if (window.__skipRender) return;').replace('  window.TensorSharpStudio = Object.freeze', '  window.__test = { S, ui, api, change, receiveProject, localMediaUrl, frames, seconds, snapped, renderAllowed, history, response, boundedMessages, restoreAssistantHistory, deleteAsset, confirmAssetDeletion, renderLibrary, generationForm, keydown };\n  window.TensorSharpStudio = Object.freeze');
  vm.runInContext(instrumented, context, { filename: 'studio.js' });
  const api = context.window.__test;
  api.ui.save = { textContent: '' }; api.ui.status = { textContent: '', dataset: {} };
  const project = { id: 'a'.repeat(32), name: 'Test', revision: 4, frameRate: { numerator: 30000, denominator: 1001 }, tracks: [], clips: [], assets: [], captions: [] };
  api.S.project = project;
  return { api, calls, handlers, project, context, document };
}
function response(value, status = 200) { return { ok: status < 400, status, statusText: 'Failure', text: async () => JSON.stringify(value) }; }
function deletionHarness(usageCount = 0) {
  const h = harness(), { api, project, document, context } = h;
  context.window.__skipRender = true;
  api.S.root = document.createElement('section'); document.body.appendChild(api.S.root);
  api.ui.library = document.createElement('aside'); api.S.root.appendChild(api.ui.library);
  const trigger = document.createElement('button'); api.ui.library.appendChild(trigger); trigger.focus();
  project.assets.push({ id: 'asset-a', name: '<image & video>', kind: 'video', durationSeconds: 1 });
  project.clips = Array.from({ length: usageCount }, (_, i) => ({ id: 'clip-' + i, assetId: 'asset-a' }));
  return { ...h, trigger, dialog: () => api.S.root.querySelector('.ts-delete-dialog'), dialogButtons: () => api.S.root.querySelector('.ts-delete-dialog').querySelectorAll('button') };
}

function generationHarness(previous = null, overrides = {}) {
  const h = harness(), { api, project, document } = h;
  const model = { modelId: 'ltx-test', displayName: 'LTX 2.5', available: true, textToVideo: true,
    imageConditioning: true, lastFrameConditioning: true, audioOutput: true, temporalGuidance: true,
    spatialMultiple: 64, temporalMultiple: 8, temporalOffset: 1, maxDimension: 2048, maxFrames: 257,
    defaultWidth: 1024, defaultHeight: 1024, defaultFrameCount: 145, defaultFps: 24,
    defaultSteps: 8, fixedSteps: 8, defaultCfgScale: 1, fixedCfgScale: 1, defaultGenerateAudio: true, ...overrides };
  api.S.capabilities = { models: [model] };
  project.width = 1280; project.height = 720;
  project.assets.push({ id: 'banner', kind: 'image', name: 'banner_1.png', width: 1536, height: 1024 });
  if (previous) {
    project.assets.push({ id: 'old-take', kind: 'video', name: 'Old take', generation: { request: previous } });
    api.S.regenerateAsset = 'old-take';
  }
  api.ui.inspector = document.createElement('aside');
  const body = document.createElement('div'); api.generationForm(body);
  const field = label => body.querySelectorAll('label').find(node => node.textContent === label).children[0];
  const audio = body.querySelectorAll('input').find(node => node.type === 'checkbox');
  const generate = body.querySelectorAll('button').find(node => node.textContent === 'Generate take');
  const defaults = body.querySelectorAll('button').find(node => node.textContent.startsWith('Use model defaults'));
  return { ...h, body, field, audio, generate, defaults };
}

test('new Studio image take submits the host quality defaults without reducing them to timeline or preview size', async () => {
  const { field, audio, generate, handlers, calls } = generationHarness();
  assert.equal(Number(field('Width').value), 1024); assert.equal(Number(field('Height').value), 1024);
  assert.equal(Number(field('Output frames').value), 145); assert.equal(audio.checked, true);
  field('Describe this shot').value = '保持人物全身入镜，让角色移动双手。';
  field('Image guidance / first frame').value = 'banner'; field('Seed (−1 = random)').value = '42';
  handlers.push(response({ id: 'job' }), response([])); await generate.click();
  assert.equal(calls[0].body.action, 'jobs/generate');
  const request = calls[0].body.payload;
  assert.equal(request.width, 1024); assert.equal(request.height, 1024); assert.equal(request.frameCount, 145);
  assert.equal(request.fps, 24); assert.equal(request.steps, 8); assert.equal(request.cfgScale, 1); assert.equal(request.generateAudio, true);
  assert.equal(request.prompt, '保持人物全身入镜，让角色移动双手。'); assert.equal(request.conditioningAssetId, 'banner'); assert.equal(request.seed, 42);
});

test('Studio uses advertised defaults for other host geometry and frame rate', () => {
  const { field, audio, defaults } = generationHarness(null, { defaultWidth: 1280, defaultHeight: 768, defaultFps: 30, defaultGenerateAudio: false });
  assert.equal(Number(field('Width').value), 1280); assert.equal(Number(field('Height').value), 768);
  assert.equal(Number(field('Frames per second').value), 30); assert.equal(audio.checked, false);
  assert.equal(defaults.textContent, 'Use model defaults (1280 × 768)');
});

test('explicit Studio size and audio choices remain in the submitted take', async () => {
  const { field, audio, generate, handlers, calls } = generationHarness();
  field('Describe this shot').value = 'A dancer raises a hand.'; field('Width').value = '768'; field('Height').value = '512'; audio.checked = false;
  handlers.push(response({ id: 'job' }), response([])); await generate.click();
  const request = calls[0].body.payload;
  assert.equal(request.width, 768); assert.equal(request.height, 512); assert.equal(request.generateAudio, false);
});

test('an old take preserves its settings until current model defaults are explicitly selected', async () => {
  const previous = { modelId: 'ltx-test', prompt: 'A dancer raises a hand.', width: 512, height: 512, frameCount: 81,
    fps: 24, steps: 8, cfgScale: 1, generateAudio: false, seed: 42, conditioningAssetId: 'banner',
    temporalConditions: [{ assetId: 'banner', frameIndex: 80, strength: .75 }], shotId: 'same-shot' };
  const { field, audio, defaults, generate, handlers, calls } = generationHarness(previous);
  assert.equal(Number(field('Width').value), 512); assert.equal(Number(field('Output frames').value), 81); assert.equal(audio.checked, false);
  await defaults.click();
  assert.equal(Number(field('Width').value), 1024); assert.equal(Number(field('Height').value), 1024);
  assert.equal(Number(field('Output frames').value), 145); assert.equal(audio.checked, true);
  handlers.push(response({ id: 'job' }), response([])); await generate.click();
  const request = calls[0].body.payload;
  assert.equal(request.width, 1024); assert.equal(request.height, 1024); assert.equal(request.frameCount, 145);
  assert.equal(request.prompt, previous.prompt); assert.equal(request.seed, 42); assert.equal(request.shotId, 'same-shot');
  assert.equal(request.conditioningAssetId, 'banner'); assert.deepEqual(request.temporalConditions, previous.temporalConditions);
  assert.equal(previous.width, 512, 'The saved provenance must remain unchanged.');
});

test('asset library renders a named Delete asset action even for missing sources', () => {
  const { api, project } = deletionHarness(); project.assets[0].missing = true;
  api.renderLibrary();
  const action = api.ui.library.querySelectorAll('button').find(b => b.textContent === 'Delete asset');
  assert.ok(action); assert.equal(action.attributes['aria-label'], 'Delete asset: <image & video>');
  assert.equal(action.disabled, undefined);
});

test('deletion confirmation defaults to Cancel, traps focus and cancels with Escape without host mutation', async () => {
  const { api, calls, document, trigger, dialog, dialogButtons } = deletionHarness(2);
  const pending = api.deleteAsset('asset-a'), backdrop = dialog(), [cancel, confirm] = dialogButtons();
  assert.equal(document.activeElement, cancel); assert.equal(api.ui.library.inert, true);
  const content = backdrop.children[0]; assert.equal(content.attributes.role, 'alertdialog'); assert.equal(content.attributes['aria-modal'], 'true');
  assert.match(content.children[1].textContent, /2 timeline clip\(s\)/); assert.match(content.children[2].textContent, /does not free disk space/);
  assert.equal(content.children[0].textContent, 'Delete “<image & video>”?');
  const backwards = backdrop.dispatch('keydown', { key: 'Tab', shiftKey: true }); assert.equal(backwards.event.defaultPrevented, true); assert.equal(document.activeElement, confirm);
  backdrop.dispatch('keydown', { key: 'Tab', shiftKey: false }); assert.equal(document.activeElement, cancel);
  const escaped = backdrop.dispatch('keydown', { key: 'Escape' }); await pending;
  assert.equal(escaped.event.propagationStopped, true); assert.equal(calls.length, 0); assert.equal(api.S.deleteDialog, null); assert.equal(api.ui.library.inert, false); assert.equal(document.activeElement, trigger); assert.equal(dialog(), null);
});

test('Cancel button preserves an unused asset and its project revision', async () => {
  const { api, calls, project, dialogButtons, dialog } = deletionHarness();
  const pending = api.deleteAsset('asset-a'); assert.match(dialog().children[0].children[1].textContent, /not used on the timeline/);
  await dialogButtons()[0].click(); await pending;
  assert.equal(calls.length, 0); assert.equal(project.assets.length, 1); assert.equal(project.revision, 4);
});

for (const usageCount of [0, 2]) test('confirmed asset deletion sends reviewed revision and explicit clip consent: ' + usageCount + ' uses', async () => {
  const { api, calls, handlers, project, dialogButtons } = deletionHarness(usageCount);
  handlers.push(response({ ...project, revision: 5, assets: [], clips: [] }));
  const pending = api.deleteAsset('asset-a'); await dialogButtons()[1].click(); await pending;
  assert.equal(calls.length, 1); assert.equal(calls[0].body.action, 'commands/apply');
  assert.equal(calls[0].body.payload.projectId, project.id); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.deepEqual(calls[0].body.payload.commands, [{ type: 'deleteAsset', assetId: 'asset-a', removeTimelineClips: usageCount > 0 }]);
  assert.match(api.ui.status.textContent, /Use Undo/); assert.equal(api.S.project.assets.length, 0);
});

for (const change of ['revision', 'project', 'epoch', 'busy']) test('deletion refuses a changed confirmation context: ' + change, async () => {
  const { api, calls, project, dialogButtons } = deletionHarness(1);
  const pending = api.deleteAsset('asset-a');
  if (change === 'revision') { project.revision++; project.clips.push({ id: 'new-use', assetId: 'asset-a' }); }
  if (change === 'project') api.S.project = { ...project, id: 'b'.repeat(32) };
  if (change === 'epoch') api.S.epoch++;
  if (change === 'busy') api.S.busy = true;
  const rejection = assert.rejects(pending, /Nothing was deleted/); await dialogButtons()[1].click(); await rejection;
  assert.equal(calls.length, 0); assert.equal(project.assets.length, 1);
});

test('server revision conflict preserves the asset and does not clear preview or selection', async () => {
  const { api, calls, handlers, project, dialogButtons } = deletionHarness(1);
  api.S.selected = 'clip-0'; api.S.regenerateAsset = 'asset-a'; api.S.sourcePreviewAssetId = 'asset-a'; api.S.sourcePreview = true;
  handlers.push(response({ code: 'revision_conflict', message: 'Project changed.' }, 409));
  const pending = api.deleteAsset('asset-a'), rejection = assert.rejects(pending, error => error.code === 'revision_conflict');
  await dialogButtons()[1].click(); await rejection;
  assert.equal(calls.length, 1); assert.equal(api.S.project, project); assert.equal(api.S.selected, 'clip-0'); assert.equal(api.S.sourcePreview, true); assert.equal(api.S.regenerateAsset, 'asset-a');
});

test('successful deletion clears removed clip, source preview, and regeneration selection', async () => {
  const { api, handlers, project, dialogButtons } = deletionHarness(1);
  api.S.selected = 'clip-0'; api.S.regenerateAsset = 'asset-a'; api.S.sourcePreviewAssetId = 'asset-a'; api.S.sourcePreview = true;
  let paused = false, cleared = false, loaded = false;
  api.ui.video = { pause() { paused = true; }, removeAttribute(name) { cleared = name === 'src'; }, load() { loaded = true; }, hidden: false };
  handlers.push(response({ ...project, revision: 5, assets: [], clips: [] }));
  const pending = api.deleteAsset('asset-a'); await dialogButtons()[1].click(); await pending;
  assert.equal(api.S.selected, null); assert.equal(api.S.regenerateAsset, null); assert.equal(api.S.sourcePreviewAssetId, null); assert.equal(api.S.sourcePreview, false);
  assert.equal(paused && cleared && loaded && api.ui.video.hidden, true);
});

test('only same-host managed media URLs can enter chat import', () => {
  const { api } = harness();
  assert.equal(api.localMediaUrl('/uploads/video.mp4'), '/uploads/video.mp4');
  assert.equal(api.localMediaUrl('http://127.0.0.1:5000/api/code/artifacts/run/video.mp4'), '/api/code/artifacts/run/video.mp4');
  for (const url of ['https://remote.example/video.mp4', '//remote.example/uploads/private.mp4', 'file:///etc/passwd', 'javascript:alert(1)', 'data:video/mp4;base64,AQ==', '/uploads/../private.mp4', '/private.mp4', 'http://localhost:5000/uploads/a.mp4']) assert.equal(api.localMediaUrl(url), null, url);
});

test('fractional project rate maps frames consistently and snapping excludes self', () => {
  const { api, project } = harness();
  assert.equal(api.frames(1001 / 30000), 1);
  assert.equal(api.seconds(30000), 1001);
  assert.equal(api.frames(1 / (30000 / 1001) / 2), 1);
  project.clips.push({ id: 'selected', startFrame: 45, durationFrames: 30 }, { id: 'other', startFrame: 100, durationFrames: 20 });
  api.S.zoom = 65; api.S.playhead = 0;
  assert.equal(api.snapped(99, 'selected'), 100);
  assert.equal(api.snapped(44, 'selected'), 44);
  api.S.snap = false; assert.equal(api.snapped(99, 'selected'), 99);
});

test('manual edits send revision and stable command IDs without optimistic state mutation', async () => {
  const { api, calls, handlers, project } = harness();
  handlers.push(response({ ...project, revision: 5 }));
  await api.change([{ type: 'moveClip', clipId: 'clip-a', startFrame: 30 }]);
  const call = calls[0]; assert.equal(call.url, '/api/studio'); assert.equal(call.options.headers['X-TensorSharp-Studio'], '1');
  assert.equal(call.options.credentials, 'same-origin'); assert.equal(call.body.action, 'commands/apply');
  assert.equal(call.body.payload.projectId, project.id); assert.equal(call.body.payload.expectedRevision, 4); assert.match(call.body.payload.commandId, /^[a-f\d]{32}$/);
  assert.equal(call.body.payload.commands[0].startFrame, 30); assert.equal(api.S.project.revision, 5); assert.equal(project.revision, 4); assert.equal(api.S.busy, false);
});

test('revision conflict preserves previous project and releases saving state', async () => {
  const { api, handlers, project } = harness();
  handlers.push(response({ code: 'revision_conflict', message: 'Project changed.' }, 409));
  await assert.rejects(api.change([{ type: 'removeClip', clipId: 'clip-a' }]), e => e.status === 409 && e.code === 'revision_conflict');
  assert.equal(api.S.project, project); assert.equal(api.S.project.revision, 4); assert.equal(api.S.busy, false);
});

test('undo and redo use the same revision-bound host protocol', async () => {
  const { api, handlers, calls, project } = harness();
  handlers.push(response({ ...project, revision: 5 })); await api.history('undo');
  handlers.push(response({ ...project, revision: 6 })); await api.history('redo');
  assert.equal(calls[0].body.action, 'commands/undo'); assert.equal(calls[0].body.payload.expectedRevision, 4);
  assert.equal(calls[1].body.action, 'commands/redo'); assert.equal(calls[1].body.payload.expectedRevision, 5);
});

test('late undo response cannot replace a newly selected project', async () => {
  const { api, handlers, project } = harness();
  const other = { ...project, id: 'b'.repeat(32), revision: 1 };
  handlers.push({ ok: true, text: async () => { api.S.project = other; return JSON.stringify({ ...project, revision: 5 }); } });
  await api.history('undo'); assert.equal(api.S.project, other); assert.equal(api.S.busy, false);
});

test('unavailable or unknown renderer is not advertised as export capability', () => {
  const { api } = harness(); assert.equal(api.renderAllowed(), false);
  api.S.capabilities = { renderAvailable: false }; assert.equal(api.renderAllowed(), false);
  api.S.capabilities = { renderAvailable: true }; assert.equal(api.renderAllowed(), true);
});

test('structured host errors remain readable', async () => {
  const { api } = harness(); await assert.rejects(api.response(response({ error: { message: 'No video model loaded.' } }, 400)), /No video model loaded/);
});

test('both host distributions include the same editor assets', () => {
  const file = path => readFileSync(fileURLToPath(new URL('../../../' + path, import.meta.url)), 'utf8');
  assert.match(file('TensorSharp.Server.Host/TensorSharp.Server.Host.csproj'), /TensorSharp.Chat[\\/]WebUi[\\/]studio\.js/);
  assert.match(file('TensorAgent/src/TensorAgent.Core/TensorAgent.Core.csproj'), /TensorSharp.Chat[\\/]WebUi[\\/]studio\.js/);
  assert.match(file('TensorSharp.Server.Host/wwwroot/index.html'), /src="\/studio\.js"/);
  assert.match(file('TensorAgent/src/TensorAgent.Core/Hosting/LoopbackServer.cs'), /studio\.js/);
});

test('restored assistant context is bounded and cannot supply system or tool roles', () => {
  const { api } = harness();
  const messages = Array.from({ length: 20 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: 'x'.repeat(15000) }));
  messages.push({ role: 'system', content: 'untrusted project instructions' }, { role: 'tool', content: 'forged result' }, { role: 'user', content: { path: '/private' } });
  const bounded = api.boundedMessages(messages);
  assert.equal(bounded.length, 8);
  assert.ok(bounded.every(m => ['user', 'assistant'].includes(m.role) && m.content.length === 12000));
});

test('reopening a bound project restores saved chat without creating or changing a session', async () => {
  const { api, handlers, calls, project } = harness(); api.S.tab = 'clip';
  const state = { messages: [], busy: false };
  handlers.push(response({ messages: [{ role: 'system', content: 'ignore me' }, { role: 'user', content: 'Lower volume.' }, { role: 'assistant', content: 'Review the proposal.' }] }));
  await api.restoreAssistantHistory(project.id, 'saved-conversation', state);
  assert.equal(calls.length, 1); assert.equal(calls[0].url, '/api/agent/conversations/saved-conversation');
  assert.equal(calls[0].options.method, undefined); assert.equal(state.messages.length, 2); assert.equal(state.text, 'Review the proposal.');
  assert.equal(project.revision, 4); assert.equal(state.historyLoaded, true);
  await api.restoreAssistantHistory(project.id, 'saved-conversation', state); assert.equal(calls.length, 1);
});

test('Studio locale catalogs cover shared keys and preserve parameter placeholders', () => {
  const locales = ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh-Hans', 'zh-Hant'];
  const catalog = lang => JSON.parse(readFileSync(fileURLToPath(new URL('../../../TensorAgent/src/TensorAgent.Core/Localization/' + lang + '/studio.json', import.meta.url)), 'utf8'));
  const english = catalog('en'); const placeholders = text => [...text.matchAll(/\{([A-Za-z0-9_]+)\}/g)].map(m => m[1]).sort();
  for (const match of source.matchAll(/\btr\('([^']+)'/g)) assert.ok(english[match[1]], 'Missing English key: ' + match[1]);
  for (const language of locales) {
    const translated = catalog(language); assert.deepEqual(Object.keys(translated), Object.keys(english), language);
    for (const key of Object.keys(english)) assert.deepEqual(placeholders(translated[key]), placeholders(english[key]), language + ':' + key);
  }
});

test('creation uses WebView-safe UI and the strict project creation frame-rate fields', () => {
  assert.doesNotMatch(source, /window\.prompt\(/);
  const creation = source.match(/api\('projects\/create',[^\n]+/)[0];
  assert.match(creation, /fpsNumerator: 24, fpsDenominator: 1/);
  assert.doesNotMatch(creation, /frameRate:/);
});
