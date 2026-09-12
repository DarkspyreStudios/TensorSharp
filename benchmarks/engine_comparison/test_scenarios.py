"""Coverage for the client-driven multi-turn scenarios (agentic, code_edit).

Nothing here talks to a server: `engines.run_openai_chat` is replaced by a
scripted list of responses, so the tests assert what the harness SENDS on each
round trip and what its checkers accept — which is the part that decides whether
a recorded cell means anything.
"""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import engines
import run_matrix
import scenarios as scen


def tool_response(name, arguments):
    calls = [{"id": f"call_{name}", "type": "function",
              "function": {"name": name, "arguments": json.dumps(arguments)}}]
    return {"assistant_message": {"role": "assistant", "content": None,
                                  "tool_calls": calls},
            "tool_call_details": calls, "tool_calls": [name],
            "finish_reason": "tool_calls", "usage_present": True,
            "prompt_tokens": 10, "completion_tokens": 5, "ttft_ms": 1.0,
            "prefill_tps": 10.0, "decode_tps": 5.0, "total_wall_ms": 1.0,
            "output_text": ""}


def text_response(content):
    return {"assistant_message": {"role": "assistant", "content": content},
            "tool_call_details": [], "tool_calls": [], "finish_reason": "stop",
            "usage_present": True, "prompt_tokens": 20, "completion_tokens": 8,
            "ttft_ms": 2.0, "prefill_tps": 10.0, "decode_tps": 4.0,
            "total_wall_ms": 2.0, "output_text": content}


AGENTIC_TURNS = [
    tool_response("read_invoice", {"invoice_id": "INV-472"}),
    tool_response("calculate_total", {"unit_price": 13.75, "quantity": 5}),
    text_response('{"invoice_id":"INV-472","total":74.25}'),
]

SLUGIFY = ("```python\n"
           "import re\n"
           "def slugify(text):\n"
           "    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
           "```")
SLUGIFY_EDITED = ("```python\n"
                  "import re\n"
                  "def slugify_title(text, max_length=40):\n"
                  "    slug = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"
                  "    return slug[:max_length]\n"
                  "```")
CODE_EDIT_TURNS = [text_response(SLUGIFY), text_response(SLUGIFY_EDITED)]


def drive(scenario_id, responses, concurrency=1):
    """Run the scenario's conversation against scripted responses; returns the
    metrics dict plus every request body the harness sent."""
    req = scen.build_request(scenario_id, "tensorsharp", None)
    sent = []

    def fake(base_url, model_name, messages, **kwargs):
        sent.append({"messages": messages, **kwargs})
        return responses[len(sent) - 1]

    with patch.object(engines, "run_openai_chat", side_effect=fake):
        if concurrency > 1:
            m = engines.run_conversation_parallel(
                "http://unused", "m", req["messages"], concurrency=concurrency,
                followups=req["followups"], tools=req.get("tools"))
        else:
            m = engines.run_conversation(
                "http://unused", "m", req["messages"],
                followups=req["followups"], tools=req.get("tools"))
    return req, m, sent


class PortabilityTests(unittest.TestCase):
    def test_both_scenarios_resolve_against_any_config_without_being_declared(self):
        # Synthesized like `prefill_<N>`, so `--scenarios agentic` works against
        # a config written before either scenario existed — which is what makes
        # them usable for every registered model rather than one matrix.
        for sid in ("agentic", "code_edit"):
            with self.subTest(sid=sid):
                self.assertIn(sid, config.SCENARIOS)
                self.assertEqual(config.SCENARIOS[sid].kind, sid)
                self.assertIsNone(config.SCENARIOS[sid].modality)
                self.assertNotIn(sid, config.DEFAULT_SCENARIOS)

    def test_builders_do_not_look_at_the_model_or_the_engine(self):
        for sid in ("agentic", "code_edit"):
            for engine in ("tensorsharp", "llamacpp", "vllm"):
                with self.subTest(sid=sid, engine=engine):
                    a = scen.build_request(sid, engine, None)
                    b = scen.build_request(sid, "tensorsharp", object())
                    self.assertEqual(a["messages"], b["messages"])


class AgenticScenarioTests(unittest.TestCase):
    def test_three_round_trips_feed_each_tool_result_into_the_next_turn(self):
        req, metrics, sent = drive("agentic", AGENTIC_TURNS)
        self.assertEqual(len(sent), 3)
        self.assertEqual(metrics["turns"], 3)
        self.assertEqual(metrics["conversation_detail"], "")

        # Turn 2 must carry the model's own call plus its fixture result.
        second = sent[1]["messages"]
        self.assertEqual([m["role"] for m in second],
                         ["user", "assistant", "tool", "user"])
        self.assertEqual(second[2]["tool_call_id"], "call_read_invoice")
        self.assertEqual(json.loads(second[2]["content"]),
                         {"invoice_id": "INV-472", "unit_price": 13.75, "quantity": 5})

        # The last turn asks for the answer, not another call.
        self.assertEqual(sent[2]["extra_body"]["tool_choice"], "none")
        self.assertEqual([m["role"] for m in sent[2]["messages"]][-3:],
                         ["assistant", "tool", "user"])
        self.assertTrue(req["checker"](metrics))

    def test_the_second_call_must_use_the_first_tool_result(self):
        # 13.75 and 5 appear nowhere in the prompt: a call that invents its own
        # numbers has not read the result, and the workflow stops there.
        responses = [AGENTIC_TURNS[0],
                     tool_response("calculate_total", {"unit_price": 1.0, "quantity": 1})]
        req, metrics, sent = drive("agentic", responses)
        self.assertEqual(len(sent), 2)
        self.assertEqual(metrics["turns"], 2)
        self.assertIn("expected calculate_total", metrics["conversation_detail"])
        self.assertFalse(req["checker"](metrics))

    def test_an_answer_computed_instead_of_read_is_marked_wrong(self):
        # 13.75 * 5 = 68.75; the tool returned 74.25 because of a handling fee
        # the model was never told about. Only the tool result can produce it.
        responses = AGENTIC_TURNS[:2] + [
            text_response('{"invoice_id":"INV-472","total":68.75}')]
        req, metrics, _ = drive("agentic", responses)
        self.assertEqual(metrics["turns"], 3)
        self.assertFalse(req["checker"](metrics))

    def test_a_missing_structured_call_stops_the_loop_instead_of_faking_one(self):
        req, metrics, sent = drive("agentic", [text_response("The weather is nice.")])
        self.assertEqual(len(sent), 1)
        self.assertIn("finish_reason=tool_calls", metrics["conversation_detail"])
        self.assertFalse(req["checker"](metrics))

    def test_a_fenced_final_answer_is_still_read(self):
        responses = AGENTIC_TURNS[:2] + [
            text_response('```json\n{"invoice_id":"INV-472","total":74.25}\n```')]
        req, metrics, _ = drive("agentic", responses)
        self.assertTrue(req["checker"](metrics))


class CodeEditScenarioTests(unittest.TestCase):
    def test_the_edit_request_carries_the_generated_program(self):
        req, metrics, sent = drive("code_edit", CODE_EDIT_TURNS)
        self.assertEqual(len(sent), 2)
        self.assertEqual(metrics["turns"], 2)
        self.assertEqual([m["role"] for m in sent[1]["messages"]],
                         ["user", "assistant", "user"])
        self.assertEqual(sent[1]["messages"][1]["content"], SLUGIFY)
        self.assertIn("slugify_title", sent[1]["messages"][2]["content"])
        self.assertTrue(req["checker"](metrics))

    def test_a_rename_that_never_uses_the_new_parameter_is_not_the_edit(self):
        half = ("```python\ndef slugify_title(text, max_length=40):\n"
                "    return text.lower()\n```")
        req, metrics, _ = drive("code_edit", [CODE_EDIT_TURNS[0], text_response(half)])
        self.assertFalse(req["checker"](metrics))

    def test_the_declared_default_has_to_be_the_one_that_was_asked_for(self):
        wrong = CODE_EDIT_TURNS[1]["assistant_message"]["content"].replace("40", "80")
        req, metrics, _ = drive("code_edit", [CODE_EDIT_TURNS[0], text_response(wrong)])
        self.assertFalse(req["checker"](metrics))

    def test_source_that_does_not_parse_cannot_pass(self):
        broken = "```python\ndef slugify_title(text, max_length=40)\n    return text\n```"
        req, metrics, _ = drive("code_edit", [CODE_EDIT_TURNS[0], text_response(broken)])
        self.assertFalse(req["checker"](metrics))

    def test_an_edit_of_a_program_that_was_never_written_cannot_pass(self):
        req, metrics, _ = drive(
            "code_edit", [text_response("Sure!"), CODE_EDIT_TURNS[1]])
        self.assertFalse(req["checker"](metrics))

    def test_an_unfenced_reply_is_still_read(self):
        bare = [text_response(SLUGIFY.split("```python\n")[1].rsplit("```", 1)[0]),
                text_response(SLUGIFY_EDITED.split("```python\n")[1].rsplit("```", 1)[0])]
        req, metrics, _ = drive("code_edit", bare)
        self.assertTrue(req["checker"](metrics))


class ConversationMetricsTests(unittest.TestCase):
    def test_the_reported_metrics_belong_to_the_final_turn(self):
        _, metrics, _ = drive("agentic", AGENTIC_TURNS)
        last = AGENTIC_TURNS[-1]
        for field in ("prompt_tokens", "completion_tokens", "ttft_ms", "decode_tps"):
            self.assertEqual(metrics[field], last[field], field)
        # ... while the wall clock covers every turn, not just the last one.
        self.assertEqual(metrics["final_turn_wall_ms"], last["total_wall_ms"])
        self.assertGreaterEqual(metrics["total_wall_ms"], 0.0)
        self.assertEqual(len(metrics["turn_metrics"]), 3)

    def test_parallel_conversations_each_run_their_own_loop(self):
        req = scen.build_request("agentic", "tensorsharp", None)
        sent = []

        def fake(base_url, model_name, messages, **kwargs):
            # Clients interleave, so the turn index comes from this request's own
            # history length (1, 4, 7 messages) rather than a shared counter.
            sent.append(messages)
            return AGENTIC_TURNS[(len(messages) - 1) // 3]

        with patch.object(engines, "run_openai_chat", side_effect=fake):
            metrics = engines.run_conversation_parallel(
                "http://unused", "m", req["messages"], concurrency=3,
                followups=req["followups"], tools=req.get("tools"))
        self.assertEqual(len(sent), 9)          # 3 clients x 3 round trips
        self.assertEqual(metrics["requests_ok"], 3)
        self.assertEqual(metrics["turns"], 3)
        self.assertTrue(req["checker"](metrics))


class CellRecordingTests(unittest.TestCase):
    """What a multi-turn cell writes into its result record."""

    def _cell(self, scenario_id, responses):
        model = config.ModelSpec(short_id="m", display="M", family="f",
                                 gguf=Path("/tmp/m.gguf"), mmproj=None,
                                 modalities={"text"}, size_class="medium")
        server = type("S", (), {"base_url": "http://unused", "_served_name": "m"})()
        with patch.object(engines, "run_openai_chat", side_effect=list(responses)):
            return run_matrix._run_cell(server, "tensorsharp", "ggml_cuda", model,
                                        scenario_id, 128)

    def test_a_completed_workflow_records_its_turns_and_correctness(self):
        res = self._cell("agentic", AGENTIC_TURNS)
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.turns, 3)
        self.assertIs(res.tool_call_ok, True)
        self.assertEqual(res.detail, "")

    def test_a_workflow_the_model_broke_is_served_but_marked_incorrect(self):
        # The server answered every request it was given, so the cell is `ok`
        # and keeps its timings; the detail says which turn stopped the loop.
        res = self._cell("agentic", [text_response("no tool for me")])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.turns, 1)
        self.assertIs(res.tool_call_ok, False)
        self.assertIn("conversation stopped after turn 1/3", res.detail)

    def test_a_single_request_scenario_still_records_one_turn(self):
        res = self._cell("text_short", [text_response("A transformer is ...")])
        self.assertEqual(res.status, "ok")
        self.assertEqual(res.turns, 1)
        self.assertEqual(res.turns_expected, 1)
        self.assertIsNone(res.tool_call_ok)

    def test_the_cell_records_how_many_turns_the_scenario_asked_for(self):
        # `turns` alone cannot say whether a workflow finished — that needs the
        # count the scenario intended, which is what keeps a 1-of-3 cell out of
        # the report's throughput tables.
        self.assertEqual(self._cell("agentic", AGENTIC_TURNS).turns_expected, 3)
        self.assertEqual(self._cell("code_edit", CODE_EDIT_TURNS).turns_expected, 2)
        self.assertEqual(
            self._cell("agentic", [text_response("no tool for me")]).turns_expected, 3)


class PartialWorkflowReportingTests(unittest.TestCase):
    """A workflow that stopped early must not be tabulated as if it had run.

    The last turn of an agentic workflow re-prefills the whole conversation, so
    a cell that answered one turn is a different, much cheaper workload than one
    that answered three — publishing its throughput beside a complete cell would
    compare two different things."""

    def _rec(self, turns, expected, **kw):
        rec = {"engine": "tensorsharp", "backend": "ggml_cuda", "model": "m",
               "scenario": "agentic", "status": "ok", "decode_tps": 40.0,
               "turns": turns, "turns_expected": expected}
        rec.update(kw)
        return rec

    def test_an_incomplete_workflow_is_named_rather_than_scored(self):
        import report
        self.assertEqual(report._cell(self._rec(3, 3), "decode_tps"), "40.0")
        self.assertEqual(report._cell(self._rec(1, 3), "decode_tps"), "partial 1/3")
        self.assertEqual(report._ok_value(self._rec(1, 3), "decode_tps"), 0.0)
        self.assertEqual(report._ok_value(self._rec(3, 3), "decode_tps"), 40.0)

    def test_results_recorded_before_the_field_existed_are_never_partial(self):
        import report
        legacy = {"engine": "tensorsharp", "backend": "ggml_cuda", "model": "m",
                  "scenario": "text_short", "status": "ok", "decode_tps": 40.0}
        self.assertFalse(report._partial_workflow(legacy))
        self.assertEqual(report._cell(legacy, "decode_tps"), "40.0")

    def test_one_client_that_stopped_short_makes_the_whole_cell_partial(self):
        # `concurrency N` claims N copies of the workflow ran. Reporting the
        # representative client's turn count would hide a client that broke.
        def fake(base_url, model_name, messages, **kwargs):
            # The client whose history is longest is the one still going; the
            # other is answered with prose and stops at turn 1.
            if len(messages) == 1 and fake.first:
                fake.first = False
                return text_response("no tool for me")
            return AGENTIC_TURNS[(len(messages) - 1) // 3]
        fake.first = True
        req = scen.build_request("agentic", "tensorsharp", None)
        with patch.object(engines, "run_openai_chat", side_effect=fake):
            m = engines.run_conversation_parallel(
                "http://unused", "m", req["messages"], concurrency=2,
                followups=req["followups"], tools=req.get("tools"))
        self.assertEqual(m["turns"], 1)
        self.assertFalse(req["checker"](m))


class ConversationRequestStateTests(unittest.TestCase):
    def test_a_turn_only_states_what_it_changes(self):
        # tool_choice is set on the last turn of `agentic` and must not drop the
        # run-wide reasoning mode; and what a turn does not mention carries
        # forward from the turn before it, not from the start of the run.
        req = scen.build_request("agentic", "tensorsharp", None)
        sent = []

        def fake(base_url, model_name, messages, **kwargs):
            sent.append(kwargs)
            return AGENTIC_TURNS[len(sent) - 1]

        with patch.object(engines, "run_openai_chat", side_effect=fake):
            engines.run_conversation(
                "http://unused", "m", req["messages"], followups=req["followups"],
                tools=req.get("tools"), extra_body={"think": False})
        self.assertEqual([k["extra_body"] for k in sent],
                         [{"think": False}, {"think": False},
                          {"think": False, "tool_choice": "none"}])
        # The tool catalogue stays declared on every turn, tool_choice included.
        self.assertTrue(all(k["tools"] for k in sent))

    def test_a_turn_that_changes_tools_is_not_undone_by_the_next_one(self):
        replacement = [{"type": "function", "function": {"name": "only_this"}}]
        followups = [lambda m, msgs: {"messages": list(msgs), "tools": replacement},
                     lambda m, msgs: {"messages": list(msgs)}]
        sent = []

        def fake(base_url, model_name, messages, **kwargs):
            sent.append(kwargs)
            return text_response("ok")

        with patch.object(engines, "run_openai_chat", side_effect=fake):
            engines.run_conversation("http://unused", "m", [{"role": "user", "content": "hi"}],
                                     followups=followups, tools=[{"original": True}])
        self.assertEqual([k["tools"] for k in sent],
                         [[{"original": True}], replacement, replacement])


if __name__ == "__main__":
    unittest.main()
