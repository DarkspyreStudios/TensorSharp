import importlib.util
from decimal import Decimal
from pathlib import Path
import sys
import unittest


VALIDATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION))
SPEC = importlib.util.spec_from_file_location("qwen35_reviewers", VALIDATION / "probe_qwen35_reviewers.py")
REVIEWERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEWERS)


LITERAL_TABLE = """| Metric | Proposal A | Proposal B |
|---|---|---|
| Revenue | **$3,000** | $3,600 USD |
| Total cost | $1,860 | $1,800 |
| Actual profit | $1,140 | $1,800 |
| Claimed profit | $1,300 | $1,900 |
| Profit overstatement | $160 | $100 |
"""
EQUATION_TABLE = """| | Proposal A | Proposal B |
|---|---|---|
| **Revenue** | 120 × $25 = **$3,000** | 200 * $18 = **$3,600** |
| **Total cost** | (120 × $13) + $300 = $1,560 + $300 = **$1,860** | (200 x $7) + $400 = $1,400 + $400 = **$1,800** |
| **Actual profit** | $3,000 − $1,860 = **$1,140** | $3,600 - $1,800 = **$1,800** |
| **Claimed profit** | $1,300 | $1,900 |
| **Profit overstatement** | $1,300 − $1,140 = **$160** | $1,900 - $1,800 = **$100** |
"""


class Qwen35ReviewersTests(unittest.TestCase):
    def test_literal_and_equation_tables_have_identical_metric_assignments(self):
        expected = {label: {key: value for key, value in metrics.items()
                            if key not in ("variable_cost", "fixed_cost")}
                    for label, metrics in REVIEWERS.EXPECTED.items()}
        self.assertEqual(REVIEWERS.verify_table(LITERAL_TABLE), expected)
        self.assertEqual(REVIEWERS.verify_table(EQUATION_TABLE), expected)

    def test_transposed_table_preserves_proposal_and_metric_mapping(self):
        table = """| Proposal | Revenue | Total cost | Actual profit | Profit overstatement |
|---|---|---|---|---|
| A | 120 * 25 = $3,000 | 1560 + 300 = 1860 | 3000 - 1860 = 1140 | 1300 - 1140 = 160 |
| B | 200 * 18 = $3,600 | 1400 + 400 = 1800 | 3600 - 1800 = 1800 | 1900 - 1800 = 100 |
"""
        result = REVIEWERS.verify_table(table)
        for label in ("A", "B"):
            self.assertEqual(result[label], {key: REVIEWERS.EXPECTED[label][key]
                                             for key in REVIEWERS.REQUIRED})

    def test_per_unit_input_rows_do_not_replace_total_cost_metrics(self):
        table = LITERAL_TABLE.replace(
            "| Total cost |", "| Variable cost per unit / customer | $13 | $7 |\n"
            "| Total variable cost | $1,560 | $1,400 |\n| Total cost |")
        result = REVIEWERS.verify_table(table)
        self.assertEqual(result["A"]["variable_cost"], 1560)
        self.assertEqual(result["B"]["variable_cost"], 1400)
        with self.assertRaisesRegex(ValueError, "Proposal A variable_cost"):
            REVIEWERS.verify_table(table.replace("$1,560", "$1,500"))

    def test_per_customer_input_column_is_not_a_total_cost_column(self):
        table = """| Proposal | Revenue | Variable cost per customer | Total variable cost | Total cost | Actual profit | Profit overstatement |
|---|---|---|---|---|---|---|
| A | $3,000 | $13 | $1,560 | $1,860 | $1,140 | $160 |
| B | $3,600 | $7 | $1,400 | $1,800 | $1,800 | $100 |
"""
        result = REVIEWERS.verify_table(table)
        self.assertEqual(result["A"]["variable_cost"], 1560)
        self.assertEqual(result["B"]["variable_cost"], 1400)

    def test_overstated_literal_qualifier_preserves_exact_amount_and_direction(self):
        table = LITERAL_TABLE.replace("| $160 | $100 |", "| **$160 overstated** | $100 overstated |")
        self.assertEqual(REVIEWERS.verify_table(table), REVIEWERS.verify_table(LITERAL_TABLE))
        for incorrect in (table.replace("$160", "$159"), table.replace("overstated**", "understated**"),
                          table.replace("**$3,000**", "**$3,000 overstated**")):
            with self.subTest(incorrect=incorrect):
                with self.assertRaises(ValueError):
                    REVIEWERS.verify_table(incorrect)

    def test_literals_and_annotations_remain_supported(self):
        for cell in ("$3,000", "**$3,000**", "*$3,000*", "_3000_",
                     "`$3,000`", "$3,000 USD", "$3,000 (120 x $25)"):
            with self.subTest(cell=cell):
                self.assertEqual(REVIEWERS.monetary_value(cell), Decimal(3000))

    def test_decimal_division_parentheses_and_signs_are_exact(self):
        for expression, expected in (("($600 / 2) + -$10 = $290", "290"),
                                     ("0.1 + 0.2 = 0.30", "0.30"),
                                     ("6 ÷ 2 = +3", "3")):
            with self.subTest(expression=expression):
                self.assertEqual(REVIEWERS.monetary_value(expression), Decimal(expected))

    def test_wrong_intermediate_even_with_correct_final_is_rejected(self):
        incorrect = EQUATION_TABLE.replace("$1,560 + $300", "$1,500 + $300")
        with self.assertRaisesRegex(ValueError, "segments disagree"):
            REVIEWERS.verify_table(incorrect)

    def test_wrong_expression_or_final_is_rejected(self):
        for incorrect in (EQUATION_TABLE.replace("120 × $25", "120 × $24"),
                          EQUATION_TABLE.replace("**$3,000**", "**$3,001**")):
            with self.subTest(incorrect=incorrect):
                with self.assertRaisesRegex(ValueError, "segments disagree"):
                    REVIEWERS.verify_table(incorrect)

    def test_consistent_but_wrong_equation_fails_expected_value(self):
        incorrect = EQUATION_TABLE.replace("120 × $25 = **$3,000**", "120 * $24 = $2,880")
        with self.assertRaisesRegex(ValueError, "Proposal A revenue: expected 3000, got 2880"):
            REVIEWERS.verify_table(incorrect)

    def test_reversed_metric_values_are_not_accepted(self):
        incorrect = LITERAL_TABLE.replace("**$3,000**", "$1,140").replace(
            "| Actual profit | $1,140", "| Actual profit | $3,000")
        with self.assertRaisesRegex(ValueError, "Proposal A revenue"):
            REVIEWERS.verify_table(incorrect)

    def test_missing_required_metric_is_not_accepted(self):
        incorrect = "\n".join(line for line in EQUATION_TABLE.splitlines()
                              if "overstatement" not in line)
        with self.assertRaisesRegex(ValueError, "missing.*overstatement"):
            REVIEWERS.verify_table(incorrect)

    def test_code_like_or_unsupported_ast_expressions_are_rejected(self):
        expressions = (
            "__import__('os').system('echo should-never-run') = 3000",
            "Decimal('3000') = 3000", "(3000).__class__ = 3000",
            "True * 3000 = 3000", "revenue = 3000", "[3000][0] = 3000",
            "3000 if 1 else 0 = 3000", "(lambda: 3000)() = 3000",
            "1 ** 0 = 10", "1**0**0 = 100", "6000 // 2 = 3000", "3000 % 4000 = 3000",
            "NaN = 3000", "Infinity = 3000", "1,00 = 100", "0x10 = 16",
            "3000 / 0 = 3000", "3000 == 3000", "= 3000", "3000 =",
        )
        for expression in expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    REVIEWERS.monetary_value(expression)

    def test_expression_work_is_bounded(self):
        for expression in ("1+" * 300 + "1 = 301", "-" * 40 + "3000 = 3000"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    REVIEWERS.monetary_value(expression)

    def test_rounded_decimal_arithmetic_cannot_validate_an_exact_equality(self):
        # More than 80 significant digits would otherwise round away the +1.
        large = "1" + "0" * 81
        with self.assertRaises(ValueError):
            REVIEWERS.monetary_value(f"{large} + 1 = {large}")

    def test_equations_do_not_bypass_delegation_or_stream_checks(self):
        frames = [
            {"agent_id": "/root", "skill_step": "spawn_agent", "ok": True},
            {"agent_id": "/root", "skill_step": "spawn_agent", "ok": True},
            {"agent_id": "/root", "skill_step": "wait_agent", "ok": True},
            {"token": EQUATION_TABLE}, {"done": True},
        ]
        report = {"prompt": REVIEWERS.PROMPT, "answer": EQUATION_TABLE, "frames": frames}
        assessment = REVIEWERS.assess(report)
        self.assertTrue(assessment["automated_checks_passed"])
        self.assertIsNone(assessment["passed"])
        self.assertEqual(assessment["status"], "manual_review_required")
        report["frames"] = [frames[0], frames[2], frames[1], *frames[3:]]
        self.assertFalse(REVIEWERS.assess(report)["automated_checks_passed"])
        report["frames"] = frames
        report["answer"] = LITERAL_TABLE
        self.assertFalse(REVIEWERS.assess(report)["automated_checks_passed"])


if __name__ == "__main__":
    unittest.main()
