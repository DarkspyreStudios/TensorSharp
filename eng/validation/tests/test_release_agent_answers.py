"""Tool-round narration is evidence, not the final-answer oracle."""
import importlib.util
import io
import json
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "validate-release-agent-workflows.py"
spec = importlib.util.spec_from_file_location("release_agent_answers", SOURCE)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FakeClient:
    def __init__(self, events):
        self.response = io.BytesIO(b"".join(
            b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
        ))
        self.response.status = 200
        self.response.fp = None
        self.closed = False
        self.deleted = False

    def json_request(self, method, path, *_):
        if method == "POST" and path == "/api/sessions":
            return 200, {"sessionId": "test-session"}
        if method == "DELETE" and path == "/api/sessions/test-session":
            self.deleted = True
            return 200, {}
        raise AssertionError((method, path))

    def open_sse(self, path, body, timeout):
        if path != "/api/chat" or body["sessionId"] != "test-session":
            raise AssertionError((path, body))
        return self, self.response

    def close(self):
        self.closed = True


class FinalAgentAnswerTests(unittest.TestCase):
    expected = harness.CASES["skill_selection"]["expected"]

    def tool_step(self, round=1, ok=True):
        # Host order is completed invocation, then progress finished, then the
        # next model generation. Progress text is never assistant content.
        return [
            {"tool_progress": "running", "tool": "skills_read", "text": "tool output"},
            {"skill_step": "skills_read", "skill": "release-validation", "round": round, "ok": ok},
            {"tool_progress": "finished", "tool": "skills_read"},
        ]

    def run_events(self, events):
        client = FakeClient(events)
        result = harness.run(client, "skill_selection", "test", 2, True)
        self.assertTrue(client.closed)
        self.assertTrue(client.deleted)
        self.assertEqual(result["events"], events)
        return result

    def test_pre_tool_narration_does_not_fail_the_correct_final_answer(self):
        narration = "\n\nI'll load the release-validation skill to look this up.\n"
        events = [{"token": narration}, *self.tool_step(),
                  {"thinking": "I found the record."}, {"token": self.expected[:12]},
                  {"token": self.expected[12:] + "\n"}, {"done": True}]
        result = self.run_events(events)
        self.assertEqual(result["status"], "ok", result.get("detail"))
        self.assertEqual(result["answer"], narration + self.expected + "\n")
        self.assertEqual(result["final_answer"], self.expected + "\n")

    def test_only_the_answer_after_the_last_tool_round_is_validated(self):
        events = [{"token": "I will read the skill."}, *self.tool_step(),
                  {"token": "I will read its reference."}, *self.tool_step(round=2),
                  {"token": self.expected}, {"done": True}]
        result = self.run_events(events)
        self.assertEqual(result["status"], "ok", result.get("detail"))
        self.assertEqual(result["final_answer"], self.expected)
        self.assertIn("I will read its reference.", result["answer"])

    def test_expected_value_in_an_earlier_round_cannot_hide_a_wrong_final_answer(self):
        for final in ("wrong-value", "", f"The verification string is {self.expected}."):
            with self.subTest(final=final):
                events = [{"token": self.expected}, *self.tool_step(),
                          {"token": final}, {"done": True}]
                result = self.run_events(events)
                self.assertEqual(result["status"], "fail")
                self.assertEqual(result["final_answer"], final)
                self.assertIn("Final answer differs", result["detail"])

    def test_later_failed_tool_step_cannot_reuse_an_earlier_correct_answer(self):
        events = [*self.tool_step(), {"token": self.expected},
                  *self.tool_step(round=2, ok=False), {"done": True}]
        result = self.run_events(events)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["successful_tools"], ["skills_read"])
        self.assertEqual(result["final_answer"], "")

    def test_replace_updates_final_answer_without_reusing_its_draft(self):
        events = [{"token": "Loading the record."}, *self.tool_step(),
                  {"replace": "draft"}, {"replace": self.expected}, {"done": True}]
        result = self.run_events(events)
        self.assertEqual(result["status"], "ok", result.get("detail"))
        self.assertEqual(result["final_answer"], self.expected)
        self.assertEqual(result["events"], events)

    def test_correct_answer_still_requires_successful_tool_and_terminal_events(self):
        for events in (
            [{"token": self.expected}, {"done": True}],
            [*self.tool_step(ok=False), {"token": self.expected}, {"done": True}],
            [*self.tool_step(), {"token": self.expected}],
            [*self.tool_step(), {"token": self.expected}, {"done": True, "truncated": True}],
        ):
            with self.subTest(events=events):
                self.assertEqual(self.run_events(events)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
