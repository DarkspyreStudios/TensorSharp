"""Opt-in native Chrome form regression; never uses a retained user browser.

Run with TENSORSHARP_BROWSER_HANDOFF_TEST=1 and TENSORSHARP_PLAYWRIGHT_CLI set
to the existing pinned @playwright/cli@0.1.21 executable. Requires Node and its
installed browser. Evidence stays under ignored artifacts/validation/.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("TENSORSHARP_BROWSER_HANDOFF_TEST") == "1",
                     "Opt-in real browser regression; not run by ordinary unit tests")
class NativeSecretHandoffBrowserTests(unittest.TestCase):
    def test_native_form_origin_and_authenticated_submission(self):
        cli = Path(os.environ.get("TENSORSHARP_PLAYWRIGHT_CLI", "")).resolve()
        self.assertTrue(cli.is_file(), "Set TENSORSHARP_PLAYWRIGHT_CLI to the existing pinned CLI executable")
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node must already be installed")
        package = json.loads((cli.parent / "package.json").read_text())
        self.assertEqual(("@playwright/cli", "0.1.21"), (package["name"], package["version"]))
        helper_path = ROOT / "eng/validation/browser-secret-handoff.py"
        helper = load("native_handoff_helper", helper_path)
        fixtures = load("native_handoff_fixture", ROOT / "eng/validation/validate-browser-skill.py")
        evidence_parent = ROOT / "artifacts/validation"
        evidence_parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="native-secret-handoff-", dir=evidence_parent))
        daemon = work / "daemon"
        daemon.mkdir()
        env = dict(os.environ, PWTEST_DAEMON_SESSION_DIR=str(daemon))
        target = "native-target-" + secrets.token_hex(6)
        entry = "native-entry-" + secrets.token_hex(6)
        fixture = fixtures.AuthenticatedForumFixture()
        needles = (fixture.password.encode(), quote_plus(fixture.password).encode())
        report = {"status": "fail", "cli_version": package["version"],
                  "helper_sha256": hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                  "sessions": [target, entry], "daemon_directory": str(daemon),
                  "home_unchanged": env.get("HOME") == os.environ.get("HOME"),
                  "secret_in_argv_or_output": False, "cleanup_errors": [],
                  "limitations": "Synthetic local form and installed Chrome only; no real-account, CAPTCHA, OS-memory, or arbitrary-site logging claim."}
        source, bridge, source_thread, bridge_thread = None, None, None, None
        log_number = 0

        def call(session, arguments):
            nonlocal log_number
            argv = [node, str(cli), "--session", session, *arguments]
            self.assertFalse(fixture.password in json.dumps(argv), "Synthetic password must not enter CLI arguments")
            result = subprocess.run(argv, cwd=work, env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=75)
            output = result.stdout + result.stderr
            leaked = any(needle in output for needle in needles)
            report["secret_in_argv_or_output"] |= leaked
            for needle in needles:
                output = output.replace(needle, b"[REDACTED-SYNTHETIC-SECRET]")
            log_number += 1
            (work / f"{log_number:02}-{arguments[0]}.log").write_bytes(output)
            self.assertFalse(leaked, "Synthetic password appeared in CLI output; evidence was redacted")
            self.assertEqual(0, result.returncode, "CLI command failed; see the isolated evidence directory")
            return result.stdout.decode("utf-8", "replace")

        try:
            # The test entry browser retrieves its synthetic value from memory,
            # so even this regression never embeds a password in a CLI argument.
            secret_path = "/" + secrets.token_urlsafe(32)
            source_used = False

            class SecretSource(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_GET(self):
                    nonlocal source_used
                    valid = self.path == secret_path and not source_used
                    source_used |= valid
                    data = fixture.password.encode() if valid else b"Unavailable"
                    self.send_response(200 if valid else 410)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)

            source = ThreadingHTTPServer(("127.0.0.1", 0), SecretSource)
            source_thread = threading.Thread(target=source.serve_forever, daemon=True)
            source_thread.start()
            source_url = f"http://127.0.0.1:{source.server_port}{secret_path}"
            call(target, ["open", fixture.url + "/login"])
            observed = call(target, ["run-code", """async (page) => {
              await page.evaluate(() => {
                const user = document.querySelector('input[name=username]');
                const pass = document.querySelector('input[name=password]');
                for (const [element, title] of [[user, 'Email or username'], [pass, 'Password']]) {
                  element.setAttribute('title', title);
                  element.setAttribute('role', 'textbox');
                  element.parentElement.replaceWith(element);
                }
                const button = document.querySelector('button');
                button.textContent = 'Log In'; button.disabled = true;
                pass.addEventListener('input', () => button.disabled = pass.value.length === 0);
              });
              await page.getByRole('textbox', {name:'Email or username', exact:true}).fill(USERNAME);
              return {labels:await page.getByLabel('Password', {exact:true}).count(),
                roles:await page.getByRole('textbox', {name:'Password', exact:true}).count(),
                disabled:await page.getByRole('button', {name:'Log In', exact:true}).isDisabled()};
            }""".replace("USERNAME", json.dumps(fixture.username))])
            preflight = json.loads(observed.split("### Result\n", 1)[1].split("\n###", 1)[0].strip())
            self.assertEqual({"labels": 0, "roles": 1, "disabled": True}, preflight)
            report["target_preflight"] = preflight
            config = SimpleNamespace(node=Path(node), cli=cli, cwd=work, daemon_session_dir=daemon,
                                     session=target, expected_origin=fixture.url, expected_path="/login",
                                     username_label="Email or username", expected_username=fixture.username,
                                     password_label="Password", submit_label="Log In")
            bridge = helper.HandoffServer(config)

            class ObserveNativePost(helper.HandoffHandler):
                # Observe only safe metadata. No policy override or forged Origin.
                def do_POST(self):
                    if self.path == self.server.form_path:
                        report["native_form_origin"] = self.headers.get_all("Origin", [])
                        report["native_form_content_type"] = self.headers.get_all("Content-Type", [])
                    super().do_POST()

                def respond(self, status, body, content_type="text/plain; charset=utf-8"):
                    if self.command == "POST" and self.path == self.server.form_path:
                        report["native_form_status"] = status
                    super().respond(status, body, content_type)

            bridge.RequestHandlerClass = ObserveNativePost
            bridge_thread = threading.Thread(target=bridge.serve_forever,
                                             kwargs={"poll_interval": 0.01}, daemon=True)
            bridge_thread.start()
            call(entry, ["open", bridge.url])
            call(entry, ["run-code", """async (page) => {
              const response = await page.request.get(SECRET_SOURCE, {maxRedirects:0});
              let value = await response.text(); await response.dispose();
              await page.getByLabel('Password', {exact:true}).fill(value); value = '';
              await page.getByRole('button', {name:'Fill and submit', exact:true}).click();
              return 'native-form-submitted';
            }""".replace("SECRET_SOURCE", json.dumps(source_url))])
            self.assertTrue(bridge.done.wait(5), "Native form submission did not reach the helper")
            self.assertEqual([bridge.origin], report.get("native_form_origin"))
            self.assertEqual(["application/x-www-form-urlencoded"], report.get("native_form_content_type"))
            self.assertEqual(200, report.get("native_form_status"))
            self.assertEqual("submitted", bridge.status)
            self.assertTrue(bridge.fetched)
            self.assertIsNone(bridge.password)
            self.assertEqual(1, len(fixture.submissions))
            self.assertTrue(fixture.submissions[0]["valid"])
            fixture.validate_login(fixture.username)
            report.update(status="ok", authenticated_browser_submissions=1, secret_cleared=True)
        finally:
            for session in (entry, target):
                try:
                    call(session, ["close"])
                except Exception:
                    report["cleanup_errors"].append(session)
            for server, thread in ((bridge, bridge_thread), (source, source_thread)):
                if server is not None:
                    server.shutdown()
                    server.server_close()
                if thread is not None:
                    thread.join(timeout=5)
            fixture.close()
            leaks = []
            for path in work.rglob("*"):
                if path.is_file():
                    data = path.read_bytes()
                    if any(needle in data for needle in needles):
                        leaks.append(str(path.relative_to(work)))
                        for needle in needles:
                            data = data.replace(needle, b"[REDACTED-SYNTHETIC-SECRET]")
                        path.write_bytes(data)
            report["secret_artifact_findings"] = leaks
            if leaks or report["cleanup_errors"] or report["secret_in_argv_or_output"]:
                report["status"] = "fail"
            (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("Native browser handoff evidence: " + str(work / "report.json"), flush=True)
            self.assertFalse(leaks, "Synthetic password appeared in artifacts; affected evidence was redacted")
            self.assertFalse(report["cleanup_errors"], "Could not close an isolated test session")


if __name__ == "__main__":
    unittest.main()
