#!/usr/bin/env bun
/** Foreground, same-server comparison of Jev structured reads and the unchanged LocalJev Engine.
 * bun eng/jev-localjev-compare.ts --url http://127.0.0.1:5057 --model MODEL.gguf 
 *   --localjev artifacts/jev-reference/localjev --limit 3 --repeats 1 --out artifacts/jev/comparison
 * No server is installed, launched, or modified. Source fixtures are an integration screen.
 */
import { appendFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const FIXTURES = resolve(ROOT, "TensorSharp.TestMatrix/Inputs/jev/decisions.json");
type Json = any;
type Fetcher = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;
export interface Options {
  url: string; model: string; localjev: string; limit: number; repeats: number; out: string;
  timeout: number; samples: number; warmup: number; seed: number;
  serverMaxTokens: number; referenceMaxTokens: number; minimumAccuracy: number; description: string;
}
const LIMITATIONS = [
  "This compares original LocalJev generated probability JSON with TensorSharp selected-logit reads; it is not a vLLM comparison.",
  "Both paths use the same declared server/checkpoint/hardware and state/question inputs; their internal prompts and prompt lengths differ.",
  "LocalJev runs directly in this foreground client, without its optional extra HTTP interposer hop; its original Engine and settings implementation are imported unchanged.",
  "Small original integration fixtures do not establish general accuracy, calibrated probabilities, or statistical significance.",
  "Each pair alternates path order, but cache state and thermals are not reset; repeat measurements are not independent quality samples.",
  "Concurrency is one; this does not measure saturated batching throughput. Model loading is excluded; no warmup is implicit.",
  "LocalJev confidence is normalized entropy; the structured path confidence is maximum conditional label probability. Those scores are not treated as equivalent.",
  "Server token cap is a declared launch setting, not queried or verified by this client. Length-limited responses and retries are retained in evidence.",
  "Client instrumentation, HTTP, prompt generation, validation, and any corrective retries are included in measured wall time.",
];

function hash(value: string | Uint8Array): string { return createHash("sha256").update(value).digest("hex"); }
function record(value: unknown): value is Record<string, any> { return value !== null && typeof value === "object" && !Array.isArray(value); }
function probability(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) throw new Error(`${name}: invalid probability`);
  return value;
}
function keysMatch(value: unknown, keys: string[]): boolean {
  return record(value) && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
}
function argmax(probabilities: number[]): number { return probabilities.reduce((top, p, i) => p > probabilities[top]! ? i : top, 0); }

export function validateAnswers(questions: Json, answers: Json, expected: Json): Json[] {
  if (!keysMatch(answers, Object.keys(questions))) throw new Error("Answer keys differ from this request");
  return Object.entries(questions).map(([id, question]: [string, any]) => {
    const answer = answers[id];
    if (!record(answer) || answer.type !== question.type) throw new Error(`${id}: incorrect answer type`);
    let probabilities: number[], gold: number, prediction: number, scoreError: number | null = null;
    if (question.type === "noul") {
      const yes = probability(answer.noul, `${id}.noul`);
      probabilities = [1 - yes, yes]; gold = Number(expected[id]); prediction = Number(yes >= 0.5);
    } else {
      const labels = question.type === "choice" ? Object.keys(question.criteria) : question.criteria.map((_: unknown, i: number) => String(i));
      if (!keysMatch(answer.probabilities, labels)) throw new Error(`${id}: probability keys differ from criteria`);
      probabilities = labels.map((label: string) => probability(answer.probabilities[label], `${id}.${label}`));
      if (Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) > 1e-5) throw new Error(`${id}: probabilities do not sum to one`);
      probability(answer.confidence, `${id}.confidence`);
      prediction = argmax(probabilities);
      gold = question.type === "choice" ? labels.indexOf(expected[id]) : expected[id];
      if (question.type === "choice") {
        if (!labels.includes(answer.choice) || Math.abs(answer.probabilities[answer.choice] - probabilities[prediction]!) > 1e-6) throw new Error(`${id}: choice does not select an argmax`);
      } else {
        if (!keysMatch(answer.legend, labels) || labels.some((label: string, i: number) => JSON.stringify(answer.legend[label]) !== JSON.stringify(question.criteria[i]))) throw new Error(`${id}: invalid ordered legend`);
        const weighted = probabilities.reduce((sum, p, i) => sum + i * p, 0);
        if (typeof answer.score !== "number" || !Number.isFinite(answer.score) || Math.abs(answer.score - weighted) > 1e-5) throw new Error(`${id}: score must be the zero-indexed weighted mean`);
        scoreError = Math.abs(answer.score - gold);
      }
    }
    if (!Number.isInteger(gold) || gold < 0 || gold >= probabilities.length) throw new Error(`${id}: invalid fixture gold`);
    return { id, type: question.type, probabilities, gold, prediction, correct: prediction === gold,
      brier: probabilities.reduce((sum, p, i) => sum + (p - Number(i === gold)) ** 2, 0),
      nll: -Math.log(Math.max(probabilities[gold]!, 1e-12)), score_absolute_error: scoreError };
  });
}

function baseUrl(value: string): string {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error("Use an HTTP(S) URL without credentials/query/fragment");
  return value.replace(/\/+$/, "").replace(/\/v1$/, "");
}
function positive(value: string, name: string, minimum = 1): number {
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
  return number;
}
export function parseArgs(argv: string[]): Options {
  const values: Record<string, string> = {};
  const allowed = new Set(["url", "model", "localjev", "limit", "repeats", "out", "timeout", "samples", "warmup", "seed", "server-max-tokens", "reference-max-tokens", "min-accuracy", "description"]);
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]?.replace(/^--/, "");
    if (!argv[index]?.startsWith("--") || !key || !allowed.has(key) || values[key] !== undefined || argv[index + 1] === undefined) throw new Error(`Unknown, duplicate, or valueless argument: ${argv[index]}`);
    values[key] = argv[index + 1]!;
  }
  for (const key of ["url", "model", "localjev", "out"]) if (!values[key]) throw new Error(`--${key} is required`);
  const minimumAccuracy = Number(values["min-accuracy"] ?? 1);
  if (!Number.isFinite(minimumAccuracy) || minimumAccuracy < 0 || minimumAccuracy > 1) throw new Error("--min-accuracy must be between zero and one");
  return { url: baseUrl(values.url!), model: values.model!, localjev: resolve(values.localjev!), out: resolve(values.out!),
    limit: positive(values.limit ?? "3", "--limit"), repeats: positive(values.repeats ?? "1", "--repeats"),
    timeout: positive(values.timeout ?? "600", "--timeout"), samples: positive(values.samples ?? "1", "--samples"),
    warmup: positive(values.warmup ?? "0", "--warmup", 0), seed: positive(values.seed ?? "42", "--seed", 0),
    serverMaxTokens: positive(values["server-max-tokens"] ?? "256", "--server-max-tokens"),
    referenceMaxTokens: positive(values["reference-max-tokens"] ?? "2048", "--reference-max-tokens"),
    minimumAccuracy, description: values.description ?? "" };
}

async function command(args: string[], cwd = ROOT): Promise<string | null> {
  try {
    const child = Bun.spawn(args, { cwd, stdout: "pipe", stderr: "pipe" });
    const [stdout, , code] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
    return code === 0 ? stdout.trim() : null;
  } catch { return null; }
}
function under(path: string, root: string): boolean {
  const remainder = relative(root, path);
  return remainder === "" || (!isAbsolute(remainder) && remainder !== ".." && !remainder.startsWith(".." + sep));
}
function percentile(values: number[], fraction: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b), offset = (sorted.length - 1) * fraction;
  return sorted[Math.floor(offset)]! + (sorted[Math.ceil(offset)]! - sorted[Math.floor(offset)]!) * (offset - Math.floor(offset));
}
function mean(values: number[]): number | null { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null; }

export function summarize(rows: Json[]): Json {
  const valid = rows.filter(row => row.valid), decisions = valid.flatMap(row => row.decisions);
  const total = rows.reduce((sum, row) => sum + Object.keys(row.request.questions).length, 0);
  const latency = valid.map(row => row.elapsed_ms);
  return { requests: rows.length, valid_requests: valid.length, failed_requests: rows.length - valid.length,
    unique_cases: new Set(rows.map(row => row.case)).size, decisions: total,
    accuracy_failures_count_as_wrong: total ? decisions.filter(decision => decision.correct).length / total : null,
    mean_brier_valid_only: mean(decisions.map(decision => decision.brier)), mean_nll_valid_only: mean(decisions.map(decision => decision.nll)),
    score_mae_valid_only: mean(decisions.filter(decision => decision.score_absolute_error !== null).map(decision => decision.score_absolute_error)),
    successful_latency_ms: { p50: percentile(latency, 0.5), p95: percentile(latency, 0.95), mean: mean(latency) },
    mean_input_tokens_valid_only: mean(valid.map(row => row.body.usage.input_tokens)),
    mean_output_tokens_valid_only: mean(valid.map(row => row.body.usage.output_tokens)),
    upstream_attempts: rows.reduce((sum, row) => sum + row.attempts.length, 0),
    length_limited_attempts: rows.flatMap(row => row.attempts).filter(attempt => attempt.response?.choices?.[0]?.finish_reason === "length").length,
    elapsed_seconds: rows.reduce((sum, row) => sum + row.elapsed_ms, 0) / 1000,
  };
}

/** Optional fetch injection is only for no-model harness tests. CLI always uses real HTTP. */
export async function runComparison(options: Options, fetchImpl: Fetcher = fetch): Promise<Json> {
  options = { ...options, url: baseUrl(options.url), localjev: resolve(options.localjev), out: resolve(options.out) };
  if (![resolve(ROOT, "artifacts"), resolve(ROOT, "docs/validation")].some(root => under(options.out, root))) throw new Error("Generated evidence must stay under artifacts/ or docs/validation/");
  if (await Bun.file(resolve(options.out, "manifest.json")).exists() || await Bun.file(resolve(options.out, "results.jsonl")).exists()) throw new Error("Choose a new output directory; existing evidence will not be overwritten");
  const referenceRevision = await command(["git", "rev-parse", "HEAD"], options.localjev);
  const referenceStatus = await command(["git", "status", "--porcelain", "--untracked-files=no"], options.localjev);
  if (!referenceRevision || referenceStatus !== "") throw new Error("--localjev must be an unchanged Git checkout (tracked modifications are refused)");
  const { Engine } = await import(pathToFileURL(resolve(options.localjev, "src/engine.ts")).href);
  const { loadSettings } = await import(pathToFileURL(resolve(options.localjev, "src/config.ts")).href);
  const fixtureBytes = await readFile(FIXTURES), suite = JSON.parse(fixtureBytes.toString("utf8"));
  const cases = suite.cases.slice(0, options.limit);
  if (!cases.length) throw new Error("No selected fixtures");
  const settings = loadSettings({ upstream: options.url, upstreamModel: options.model, timeoutMs: options.timeout * 1000,
    maxInflight: 1, maxOutputTokens: options.referenceMaxTokens });
  const key = settings.upstreamApiKey;
  const manifest = { started_utc: new Date().toISOString(), ...options, fixtures: FIXTURES,
    fixtures_sha256: hash(fixtureBytes), case_ids: cases.map((item: Json) => item.id), available_cases: suite.cases.length,
    harness_sha256: hash(await readFile(fileURLToPath(import.meta.url))), reference_revision: referenceRevision, reference_tracked_status: referenceStatus,
    reference_engine_sha256: hash(await readFile(resolve(options.localjev, "src/engine.ts"))),
    reference_config_sha256: hash(await readFile(resolve(options.localjev, "src/config.ts"))),
    tensorsharp_revision: await command(["git", "rev-parse", "HEAD"]), tensorsharp_status: await command(["git", "status", "--short"]),
    ggml_revision: await command(["git", "rev-parse", "HEAD"], resolve(ROOT, "ExternalProjects/ggml")),
    ggml_status: await command(["git", "status", "--porcelain", "--untracked-files=no"], resolve(ROOT, "ExternalProjects/ggml")),
    gpu: await command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    bun: Bun.version, platform: process.platform, architecture: process.arch,
    reference_settings: { timeout_ms: settings.timeoutMs, maximum_output_tokens_requested: settings.maxOutputTokens,
      malformed_retries: settings.malformedRetries, temperature: settings.temperature, maximum_inflight: settings.maxInflight,
      questions_per_call: settings.questionsPerCall, outcomes_per_call: settings.outcomesPerCall },
    server_max_tokens_declared: options.serverMaxTokens, server_token_cap_verified: false, concurrency: 1,
    fetch_mode: fetchImpl === fetch ? "real_http" : "injected_mock_no_model", limitations: LIMITATIONS };
  await mkdir(options.out, { recursive: true });
  await writeFile(resolve(options.out, "manifest.json"), JSON.stringify(manifest, null, 2));
  let attempts: Json[] = [];
  const instrumentedFetch: Fetcher = async (input, init) => {
    const started = performance.now();
    const request = typeof init?.body === "string" ? JSON.parse(init.body) : null;
    const attempt: Json = { url: String(input), request, status: null, response: null, error: null };
    attempts.push(attempt);
    try {
      const response = await fetchImpl(input, init);
      attempt.status = response.status;
      const text = await response.clone().text();
      try { attempt.response = JSON.parse(text); } catch { attempt.response = { unparsed_body: text }; }
      return response;
    } catch (error) { attempt.error = String(error); throw error; }
    finally { attempt.elapsed_ms = performance.now() - started; }
  };
  const engine = new Engine(settings, instrumentedFetch);
  const results: Json[] = [], warmups: Json[] = [];
  async function one(path: "structured" | "localjev", fixture: Json, repetition: number, order: number, phase: string): Promise<Json> {
    attempts = [];
    const state = phase === "warmup" ? { irrelevant_request_identifier: `excluded-warmup-${repetition}`, target: fixture.state } : fixture.state;
    const request = { model: options.model, state, questions: suite.schemas[fixture.schema], samples: options.samples, seed: options.seed };
    const started = performance.now();
    const row: Json = { path, phase, case: fixture.id, repetition, pair_order: order, request,
      valid: false, decisions: [], body: null, error: null, attempts };
    try {
      if (path === "structured") {
        const response = await instrumentedFetch(`${options.url}/v1/systemone`, { method: "POST",
          headers: { "Content-Type": "application/json", ...(key ? { authorization: `Bearer ${key}` } : {}) },
          body: JSON.stringify(request), signal: AbortSignal.timeout(options.timeout * 1000) });
        row.body = await response.json();
        if (!response.ok) throw new Error(`Structured endpoint returned HTTP ${response.status}`);
      } else {
        const result = await engine.decide(request.questions, state, options.seed);
        row.body = { model: options.model, answers: result.answers, usage: { input_tokens: result.inputTokens, output_tokens: result.outputTokens } };
      }
      if (!record(row.body) || typeof row.body.model !== "string" || !record(row.body.usage) ||
          ![row.body.usage.input_tokens, row.body.usage.output_tokens].every(value => Number.isSafeInteger(value) && value >= 0)) throw new Error("Invalid model or token-usage response");
      row.decisions = validateAnswers(request.questions, row.body.answers, fixture.expected);
      row.valid = true;
    } catch (error) { row.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error); }
    row.elapsed_ms = performance.now() - started;
    await appendFile(resolve(options.out, "results.jsonl"), JSON.stringify(row) + "\n");
    console.log(`${phase} ${path} ${fixture.id} r=${repetition} ${row.elapsed_ms.toFixed(1)}ms ${row.valid ? `${row.decisions.filter((d: Json) => d.correct).length}/${row.decisions.length} correct` : row.error}`);
    return row;
  }
  for (let index = 0; index < options.warmup; index++) {
    const fixture = cases[index % cases.length];
    const paths = (index % 2 ? ["localjev", "structured"] : ["structured", "localjev"]) as ("structured" | "localjev")[];
    for (const [order, path] of paths.entries()) warmups.push(await one(path, fixture, index, order, "warmup"));
  }
  let pair = 0;
  for (let repetition = 0; repetition < options.repeats; repetition++) {
    for (const fixture of cases) {
      const order = pair++ % 2 ? ["localjev", "structured"] : ["structured", "localjev"];
      for (const [index, path] of order.entries()) results.push(await one(path as "structured" | "localjev", fixture, repetition, index, "measured"));
    }
  }
  const structured = summarize(results.filter(row => row.path === "structured")), localjev = summarize(results.filter(row => row.path === "localjev"));
  const paired = results.filter(row => row.path === "structured").map(row => {
    const reference = results.find(other => other.path === "localjev" && other.case === row.case && other.repetition === row.repetition)!;
    return { case: row.case, repetition: row.repetition, both_valid: row.valid && reference.valid,
      structured_ms: row.elapsed_ms, localjev_ms: reference.elapsed_ms,
      localjev_over_structured_latency_ratio: row.valid && reference.valid ? reference.elapsed_ms / row.elapsed_ms : null,
      structured_input_tokens: row.body?.usage?.input_tokens ?? null, localjev_input_tokens: reference.body?.usage?.input_tokens ?? null,
      structured_correct: row.decisions.filter((d: Json) => d.correct).length,
      localjev_correct: reference.decisions.filter((d: Json) => d.correct).length };
  });
  const finalReferenceRevision = await command(["git", "rev-parse", "HEAD"], options.localjev);
  const finalReferenceStatus = await command(["git", "status", "--porcelain", "--untracked-files=no"], options.localjev);
  const referenceUnchanged = finalReferenceRevision === referenceRevision && finalReferenceStatus === "";
  const passed = referenceUnchanged && [structured, localjev].every(result => result.failed_requests === 0 && result.accuracy_failures_count_as_wrong >= options.minimumAccuracy);
  const summary = { completed_utc: new Date().toISOString(), passed_requested_fixture_checks: passed,
    reference_unchanged_after_run: referenceUnchanged, reference_final_revision: finalReferenceRevision,
    reference_final_tracked_status: finalReferenceStatus,
    warmup_requests: warmups.length, warmup_failures: warmups.filter(row => !row.valid).length,
    performance_scope: "Paired descriptive measurements only; no general or vLLM performance parity verdict", structured, localjev, paired,
    median_paired_latency_ratio_valid_only: percentile(paired.filter(pair => pair.both_valid).map(pair => pair.localjev_over_structured_latency_ratio!), 0.5), limitations: LIMITATIONS };
  await writeFile(resolve(options.out, "summary.json"), JSON.stringify(summary, null, 2));
  console.log(`${passed ? "PASS" : "FAIL"} requested fixture checks; evidence: ${options.out}`);
  return summary;
}

if (import.meta.main) {
  if (process.argv.includes("--help")) {
    console.log("bun eng/jev-localjev-compare.ts --url URL --model MODEL --localjev CHECKOUT --out artifacts/RUN [--limit 3] [--repeats 1] [--warmup 0] [--timeout 600] [--samples 1] [--server-max-tokens 256] [--reference-max-tokens 2048] [--min-accuracy 1] [--description TEXT]");
  } else {
    try { const result = await runComparison(parseArgs(process.argv.slice(2))); process.exitCode = result.passed_requested_fixture_checks ? 0 : 1; }
    catch (error) { console.error(error); process.exitCode = 2; }
  }
}
