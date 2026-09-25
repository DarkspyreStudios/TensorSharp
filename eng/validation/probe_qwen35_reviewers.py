"""Run the reported two-reviewer prompt against a live host; retain every SSE frame.

Checks table values and delegation order. The SSE serializer omits tool results,
so reviewer assignments, collected reports, and native batching require host-log
review. A zero exit code means automated checks passed, not full task validation.
"""
import argparse
import ast
from decimal import Decimal, DecimalException, Inexact, localcontext
import json
from pathlib import Path
import re

from probe_multi_agent_host import request


PROMPT = """Use two independent reviewer subagents to verify these proposals.

Give each reviewer all the numbers for its assigned proposal.
Spawn both before waiting, then collect their results.

Proposal A:

- Sell 120 units at $25 each.
- Variable cost: $13 per unit.
- Fixed cost: $300.
- Claimed profit: $1,300.

Proposal B:

- Serve 200 customers paying $18 each.
- Variable cost: $7 per customer.
- Fixed cost: $400.
- Claimed profit: $1,900.

Each reviewer should calculate revenue, total cost, actual profit,
and the exact profit overstatement.

Verify their calculations and return one comparison table.
Do not speculate about the causes of the errors."""


EXPECTED = {
    "A": {"revenue": 3000, "variable_cost": 1560, "fixed_cost": 300,
          "total_cost": 1860, "actual_profit": 1140, "claimed_profit": 1300,
          "overstatement": 160},
    "B": {"revenue": 3600, "variable_cost": 1400, "fixed_cost": 400,
          "total_cost": 1800, "actual_profit": 1800, "claimed_profit": 1900,
          "overstatement": 100},
}
REQUIRED = {"revenue", "total_cost", "actual_profit", "overstatement"}


def plain(cell):
    return re.sub(r"[*_`]", "", cell).strip()


def proposal(cell):
    match = re.fullmatch(r"(?:proposal\s+)?([ab])\s*:?$", plain(cell), re.I)
    return match.group(1).upper() if match else None


def metric(cell):
    label = plain(cell).lower()
    if "overstat" in label:
        return "overstatement"
    if "revenue" in label:
        return "revenue"
    if "cost" in label:
        # Per-unit inputs are not the aggregate cost metrics checked below. A
        # detailed comparison can legitimately include both rows/columns.
        if re.search(r"\bper[\s-]+(?:unit|customer)\b", label):
            return None
        if "variable" in label:
            return "variable_cost"
        if "fixed" in label:
            return "fixed_cost"
        if "total" in label or label in ("cost", "costs"):
            return "total_cost"
    if "profit" in label:
        if "claimed" in label:
            return "claimed_profit"
        if "actual" in label or label == "profit":
            return "actual_profit"
    return None


def arithmetic_value(expression):
    """Evaluate bounded decimal arithmetic, never names, calls or Python eval."""
    if len(expression) > 512:
        raise ValueError("Table arithmetic expression is too long")
    expression = (expression.replace("\u2212", "-").replace("\u00d7", "*")
                  .replace("\u00f7", "/"))
    expression = re.sub(r"\b[xX]\b", "*", expression)
    token = re.compile(
        r"\s*(?:(?P<number>\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
        r"(?:\s*USD\b)?|(?P<operator>[()+*/-]))", re.I)
    parts = []
    position = 0
    expression = expression.strip()
    while position < len(expression):
        match = token.match(expression, position)
        if not match:
            raise ValueError(f"Unsupported table arithmetic: {expression!r}")
        number = match.group("number")
        parts.append(number.replace("$", "").replace(",", "").strip()
                     if number is not None else match.group("operator"))
        position = match.end()
    source = " ".join(parts)
    try:
        tree = ast.parse(source, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("Table arithmetic expression is too complex")

        def compute(node, depth=0):
            if depth > 32:
                raise ValueError("Table arithmetic expression is too deeply nested")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                # Preserve decimal cents without binary float conversion.
                return Decimal(ast.get_source_segment(source, node))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = compute(node.operand, depth + 1)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                left, right = compute(node.left, depth + 1), compute(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                return left / right
            raise ValueError("Only numeric +, -, *, / and parentheses are allowed")

        with localcontext() as context:
            context.prec = 80
            # A displayed exact equality must not succeed through rounding.
            context.traps[Inexact] = True
            result = compute(tree.body)
        if not result.is_finite():
            raise ValueError("Table arithmetic is not finite")
        return result
    except (SyntaxError, DecimalException, RecursionError) as error:
        raise ValueError(f"Invalid table arithmetic: {expression!r}") from error


def monetary_value(cell, *, allow_overstated=False):
    # Remove paired Markdown emphasis, retaining single arithmetic '*'.
    cleaned = re.sub(r"(?<![\d.)])(?:\*\*(.+?)\*\*|__(.+?)__)(?![\d.(])", lambda match:
                     match.group(1) or match.group(2), cell).replace("`", "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "*_":
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.replace("\u2212", "-")
    if "=" in cleaned:
        segments = cleaned.split("=")
        if any(not segment.strip() for segment in segments):
            raise ValueError(f"Incomplete table equation: {cell!r}")
        values = [arithmetic_value(segment) for segment in segments]
        if any(value != values[-1] for value in values[:-1]):
            raise ValueError(f"Table equation segments disagree: {cell!r}")
        # verify_table then checks the shared result against the exact proposal
        # and metric, so every segment must match the expected amount.
        return values[-1]
    # Preserve a literal with a parenthesized prose annotation, such as
    # "$3,000 (120 x $25)"; an annotation is not itself an asserted equation.
    match = re.fullmatch(
        r"\s*\$?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
        r"\s*(?:USD)?\s*(?:\([^\n]*\))?"
        + (r"(?:\s+overstated)?" if allow_overstated else "") + r"\s*", cleaned, re.I)
    if not match:
        raise ValueError(f"Cannot verify table monetary value: {cell!r}")
    return Decimal(match.group(1).replace(",", ""))


def markdown_tables(answer):
    group = []
    for line in answer.splitlines() + [""]:
        if "|" in line:
            group.append([cell.strip() for cell in line.strip().strip("|").split("|")])
        else:
            if (len(group) >= 3 and len(group[0]) == len(group[1])
                    and all(re.fullmatch(r":?-{3,}:?", cell) for cell in group[1])):
                yield group[0], group[2:]
            group = []


def verify_table(answer):
    comparisons = []
    for header, rows in markdown_tables(answer):
        proposal_columns = {i: proposal(cell) for i, cell in enumerate(header)
                            if proposal(cell)}
        metric_columns = {i: metric(cell) for i, cell in enumerate(header)
                          if metric(cell)}
        values = {"A": {}, "B": {}}

        def record(label, name, cell):
            if name in values[label]:
                raise ValueError(f"Duplicate {label} / {name} entry in comparison table")
            value = monetary_value(cell, allow_overstated=name == "overstatement")
            if value != EXPECTED[label][name]:
                raise ValueError(f"Proposal {label} {name}: expected {EXPECTED[label][name]}, got {value}")
            values[label][name] = float(value)

        if set(proposal_columns.values()) == {"A", "B"}:
            if len(proposal_columns) != 2:
                raise ValueError("Duplicate proposal columns in comparison table")
            for row in rows:
                name = metric(row[0])
                if not name:
                    continue
                if len(row) != len(header):
                    raise ValueError("Ragged comparison table row")
                for index, label in proposal_columns.items():
                    record(label, name, row[index])
        elif REQUIRED.issubset(metric_columns.values()):
            for row in rows:
                label = proposal(row[0])
                if not label:
                    continue
                if len(row) != len(header):
                    raise ValueError("Ragged comparison table row")
                for index, name in metric_columns.items():
                    record(label, name, row[index])
        else:
            continue
        for label in ("A", "B"):
            missing = REQUIRED - values[label].keys()
            if missing:
                raise ValueError(f"Proposal {label} table is missing {sorted(missing)}")
        comparisons.append(values)
    if len(comparisons) != 1:
        raise ValueError(f"Expected one verifiable comparison table, found {len(comparisons)}")
    return comparisons[0]


def assess(result):
    frames = result["frames"]
    # WebUiSseEvents.SkillStep identifies the invoking agent, not the child.
    root_steps = [(i, frame) for i, frame in enumerate(frames)
                  if frame.get("agent_id") == "/root" and frame.get("skill_step")]
    spawns = [i for i, frame in root_steps
              if frame["skill_step"] == "spawn_agent" and frame.get("ok")]
    waits = [i for i, frame in root_steps if frame["skill_step"] == "wait_agent"]
    successful_waits = [i for i, frame in root_steps
                        if frame["skill_step"] == "wait_agent" and frame.get("ok")]
    assessment = {"spawn_frames": spawns, "wait_frames": waits,
                  "successful_wait_frames": successful_waits,
                  "automated_checks_passed": False, "passed": False}
    try:
        if result.get("prompt") != PROMPT:
            raise ValueError("Retained prompt does not exactly match the reported user prompt")
        if len(spawns) != 2:
            raise ValueError(f"Expected two successful root spawns, got {len(spawns)}")
        if not waits or spawns[1] >= waits[0]:
            raise ValueError("Both reviewers must spawn before the root starts waiting")
        if not successful_waits:
            raise ValueError("No successful root wait invocation")
        done = [frame for frame in frames if frame.get("done")]
        if (len(done) != 1 or done[0].get("error") or done[0].get("aborted")
                or done[0].get("truncated")):
            raise ValueError("Expected one successful, non-truncated completion")
        if any(frame.get("error") for frame in frames):
            raise ValueError("Stream contains an error frame")
        # Reconstruct from the retained stream so a stale answer field cannot
        # accidentally validate a different run.
        answer = "".join(frame.get("token", "") for frame in frames)
        if answer != result.get("answer"):
            raise ValueError("Retained answer differs from SSE token frames")
        assessment["table_values"] = verify_table(answer)
        assessment["automated_checks_passed"] = True
        assessment["passed"] = None
        assessment["status"] = "manual_review_required"
        assessment["manual_review_required"] = [
            "Verify two independent reviewer assignments include every input number.",
            "Verify both reports completed and were collected. SkillStep records only invocation success; wait results and timeout/completion status are absent from Web UI SSE.",
            "Verify the answer checks both reports and does not speculate about error causes.",
            "Verify host/native logs show actual batched decode; correct arithmetic can also use serial decode.",
        ]
    except (KeyError, TypeError, ValueError) as error:
        assessment["status"] = "failed"
        assessment["error"] = str(error)
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5037")
    parser.add_argument("--out", help="Report path; defaults to reviewers.json for a live run, or a .verified.json sibling for retained evidence.")
    parser.add_argument("--verify-existing", type=Path,
                        help="Assess a retained JSON report without HTTP or model execution.")
    args = parser.parse_args()
    output = Path(args.out) if args.out else (
        args.verify_existing.with_name(args.verify_existing.stem + ".verified.json")
        if args.verify_existing else Path("artifacts/qwen35-batched-decline/reviewers.json"))
    report = {"prompt": PROMPT, "passed": False}
    try:
        if args.verify_existing:
            report = json.loads(args.verify_existing.read_text(encoding="utf-8"))
            report["verified_from"] = str(args.verify_existing)
        else:
            report.update(request(args.endpoint, PROMPT, max_tokens=1536))
        # A previous version's verdict is not evidence for this assessment.
        report.pop("error", None)
        report.pop("manual_review_required", None)
        report.update(assess(report))
    except Exception as error:
        report["passed"] = False
        report["automated_checks_passed"] = False
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    return 0 if report["automated_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
