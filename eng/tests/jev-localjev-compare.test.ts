import { describe, expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs, runComparison, summarize, validateAnswers } from "../jev-localjev-compare";

const root = resolve(import.meta.dir, "../..");
const suite = JSON.parse(await readFile(resolve(root, "TensorSharp.TestMatrix/Inputs/jev/decisions.json"), "utf8"));
const fixture = suite.cases[0], questions = suite.schemas[fixture.schema];
const answers = {
  department: { type: "choice", choice: "billing", probabilities: { billing: 1, technical: 0, sales: 0 }, confidence: 1 },
  urgent: { type: "noul", noul: 0 },
  frustration: { type: "score", score: 0, probabilities: { "0": 1, "1": 0, "2": 0 }, legend: { "0": "Calm and not upset", "1": "Frustrated but polite", "2": "Extremely angry" }, confidence: 1 },
};

describe("foreground LocalJev comparator (no model/GPU/server)", () => {
  test("validates all three answer types and rejects invalid score/probability/key contracts", () => {
    expect(validateAnswers(questions, answers, fixture.expected).every(row => row.correct)).toBe(true);
    const incorrect = structuredClone(answers);
    incorrect.frustration.score = 1;
    expect(() => validateAnswers(questions, incorrect, fixture.expected)).toThrow("zero-indexed");
    incorrect.frustration.score = 0;
    incorrect.urgent.noul = NaN;
    expect(() => validateAnswers(questions, incorrect, fixture.expected)).toThrow("invalid probability");
    expect(() => validateAnswers(questions, { ...answers, leaked: answers.urgent }, fixture.expected)).toThrow("keys differ");
  });

  test("parses arguments, rejects credential URLs and negative count", () => {
    const args = ["--url", "http://localhost:123/v1", "--model", "model", "--localjev", "artifacts/jev-reference/localjev", "--out", "artifacts/unused"];
    expect(parseArgs(args).url).toBe("http://localhost:123");
    expect(parseArgs(args).referenceMaxTokens).toBe(2048);
    expect(() => parseArgs([...args, "--limit", "-1"])).toThrow();
    args[1] = "http://secret@localhost:123";
    expect(() => parseArgs(args)).toThrow("credentials");
  });

  test("summary counts invalid requests as wrong rather than dropping them", () => {
    const correct = { valid: true, decisions: validateAnswers(questions, answers, fixture.expected), request: { questions }, case: fixture.id, elapsed_ms: 20,
      body: { usage: { input_tokens: 3, output_tokens: 2 } }, attempts: [] };
    const summary = summarize([correct, { ...correct, valid: false, decisions: [], elapsed_ms: 200 }]);
    expect(summary.accuracy_failures_count_as_wrong).toBe(0.5);
    expect(summary.failed_requests).toBe(1);
    expect(summary.elapsed_seconds).toBe(0.22);
  });

  test("uses original Engine with injected mock HTTP, alternates order and saves attempts", async () => {
    const directory = resolve(root, "artifacts/jev-localjev-compare-tests");
    await mkdir(directory, { recursive: true });
    const out = await mkdtemp(resolve(directory, "mock-only-"));
    const called: string[] = [];
    const mockFetch = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
      const url = String(input), body = JSON.parse(String(init?.body));
      if (url.endsWith("/systemone")) {
        called.push("structured");
        expect(body.questions).toEqual(questions);
        expect(body.state).toEqual(fixture.state);
        return Response.json({ model: "mock-no-model", answers, usage: { input_tokens: 100, output_tokens: 20 } });
      }
      called.push("localjev");
      expect(body.max_tokens).toBe(2048);
      expect(body.messages[1].content).toContain(fixture.state);
      return Response.json({ choices: [{ message: { content: JSON.stringify({ answers: { q1: [1, 0, 0], q2: 0, q3: [1, 0, 0] } }) }, finish_reason: "stop" }], usage: { prompt_tokens: 200, completion_tokens: 40 } });
    };
    const summary = await runComparison({ url: "http://mock.invalid", model: "mock-no-model", localjev: resolve(root, "artifacts/jev-reference/localjev"),
      limit: 1, repeats: 2, out, timeout: 5, samples: 1, warmup: 0, seed: 42, serverMaxTokens: 256, referenceMaxTokens: 2048,
      minimumAccuracy: 1, description: "MOCK ONLY: no real HTTP inference or model execution" }, mockFetch);
    expect(called).toEqual(["structured", "localjev", "localjev", "structured"]);
    expect(summary.passed_requested_fixture_checks).toBe(true);
    expect(summary.paired).toHaveLength(2);
    expect(summary.structured.mean_input_tokens_valid_only).toBe(100);
    expect(summary.localjev.mean_input_tokens_valid_only).toBe(200);
    const manifest = JSON.parse(await readFile(resolve(out, "manifest.json"), "utf8"));
    expect(manifest.fetch_mode).toBe("injected_mock_no_model");
    expect(manifest.reference_tracked_status).toBe("");
    expect(manifest.server_token_cap_verified).toBe(false);
    const rows = (await readFile(resolve(out, "results.jsonl"), "utf8")).trim().split("\n").map(row => JSON.parse(row));
    expect(rows).toHaveLength(4);
    expect(rows.every(row => row.attempts.length === 1 && row.attempts[0].status === 200)).toBe(true);
  }, 30_000);
});
