# Playwright CLI Workflows

Use the wrapper script and read each newly linked snapshot. Request a fresh
snapshot when none was produced or the page changed outside the last command.
Assume `PWCLI` is set and `pwcli` is an alias for `"$PWCLI"`.
Keep commands in the workspace root so project config and session identity are
stable. Save screenshots under `output/playwright/` using `--filename`.

## Standard interaction loop

```bash
pwcli open https://example.com
# Read the linked snapshot and choose the actual element ref.
pwcli click e3
# Read the new snapshot linked by the interaction.
```

## Form submission

```bash
pwcli open https://example.com/form --headed
# Read the linked snapshot to obtain the form's actual refs.
pwcli fill e1 "user@example.com"
pwcli fill e2 "password123"
pwcli click e3
# Read the new snapshot linked by the submission.
pwcli screenshot
```

## Data extraction

```bash
pwcli open https://example.com
# Read the linked snapshot to identify the data and actual element refs.
pwcli eval "document.title"
pwcli eval "el => el.textContent" e12
```

## Debugging and inspection

Capture console messages and network activity after reproducing an issue:

```bash
pwcli console warning
pwcli network
```

Record a trace around a suspicious flow:

```bash
pwcli tracing-start
# reproduce the issue
pwcli tracing-stop
pwcli screenshot
```

## Sessions

Use sessions to isolate work across projects:

```bash
pwcli --session marketing open https://example.com
# Read the linked snapshot for the marketing session.
pwcli --session checkout open https://example.com/checkout
```

Or set the session once:

```bash
export PLAYWRIGHT_CLI_SESSION=checkout
pwcli open https://example.com/checkout
```

When using `skills_run`, pass the same `--session` argument on every call (or use
the default consistently). A shell export does not necessarily propagate to
separate skill-script tool calls.

## Configuration file

By default, the CLI reads `.playwright/cli.config.json` from the current directory.
Use `--config` to point at a specific file. Keep the same workspace directory
across calls so the same configuration and session are selected.

Minimal example:

```json
{
  "browser": {
    "launchOptions": {
      "headless": false
    },
    "contextOptions": {
      "viewport": { "width": 1280, "height": 720 }
    }
  }
}
```

## Troubleshooting

- If an element ref fails, run `pwcli snapshot` again and retry.
- If the page looks wrong, re-open with `--headed` and resize the window.
- If a flow depends on prior state, use a named `--session`.
- On macOS under `sandbox-exec`, create the `chromiumSandbox:false` project
  configuration described in SKILL.md before launching. A failed browser
  launch can surface only as `Session closed` from the CLI daemon.
- Use `pwcli open --help` to check supported launch flags; do not guess flags.

## Account forms, handoff, and reuse

Inspect the login form and ask only for missing choices or field values. When
the user supplies them, fill the corresponding fields and submit the authorized
login; verify the visible account before continuing. Supplying form information
is different from saying `ready`: the former enables filling, while the latter
requires checking the browser's actual state. Use manual handoff only for the
steps that still need the user, rather than handing back a form you can fill.

The default profile is in memory. Cookies survive CLI calls in the same running
session, but disappear after `close`. Use a persistent workspace profile for
manual sign-in that needs to survive browser restarts:

```bash
pwcli open https://example.com/login --headed --persistent
# Ask the user to sign in in this browser and reply when ready; keep it open.
pwcli snapshot
# Read the snapshot and verify the visible account, then navigate in this session.
pwcli goto https://example.com/search
```

Keep using the same CLI session and workspace. A persistent profile belongs to
that workspace; a new chat workspace does not automatically inherit it. Do not
open the user's normal Chrome profile directory or collect credentials from it.
If the user explicitly provides a browser CDP endpoint, the CLI supports
`attach --cdp=<url>`; consult `attach --help` and use that endpoint as supplied.
An endpoint must already exist; do not guess ports or claim that `--persistent`
connects to the user's desktop browser.

For an existing session, use `goto` instead of `open` to avoid restarting the
browser. A normal completed task closes the browser; a pending login handoff
keeps it open. Use `detach` for an externally attached browser, so the user's
browser is left running. On a remote/headless server with no user-accessible browser,
explain the missing login path rather than claiming to have used the account.
