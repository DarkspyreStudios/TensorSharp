#!/usr/bin/env python3
"""Local fake-ABI lifecycle checks; no model, native library or GPU is executed."""
import argparse
import contextlib
import ctypes as C
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / 'eng/dsv41-smoke.py'
RUNNER = ROOT / 'eng/tests/dsv41-final-smoke18.py'
spec = importlib.util.spec_from_file_location('smoke_helper', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
spec = importlib.util.spec_from_file_location('smoke_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Function:
    def __init__(self, call): self.call = call
    def __call__(self, *args): return self.call(*args)


class FakeNative:
    def __init__(self, fail=None, trace_delta=0):
        self.fail, self.trace_delta = fail, trace_delta
        self.events, self.slots, self.active = [], {0: 0}, 0
        self.freed = False
        functions = {'LoadModel': self.load, 'Forward': self.forward, 'Free': self.free,
                     'VocabSize': lambda handle: 32, 'NPast': self.npast,
                     'SlotAlloc': self.alloc, 'SetActiveSlot': self.select, 'SlotFree': self.slot_free}
        for name, call in functions.items(): setattr(self, 'TSGgml_Dsv4' + name, Function(call))

    def load(self, *args):
        self.events.append(('load',))
        return 91

    def alloc(self, handle):
        self.events.append(('alloc',))
        if self.fail == 'alloc_exception': raise RuntimeError('injected allocation exception')
        if self.fail == 'alloc': return -1
        self.slots[1] = 0
        return 1

    def select(self, handle, slot):
        self.events.append(('select', slot))
        if self.fail == f'select{slot}': return -1
        self.active = slot
        return 0

    def slot_free(self, handle, slot):
        self.events.append(('slot_free', slot))
        if slot == self.active: raise AssertionError('attempted to free active slot')
        if self.fail == 'slot_free': return -1
        del self.slots[slot]
        return 0

    def npast(self, handle):
        if self.fail == 'fresh_not_zero' and self.active == 1 and self.slots[1] == 0: return 3
        return self.slots[self.active]

    def forward(self, handle, tokens_address, count, logits_address):
        tokens = np.ctypeslib.as_array(C.cast(tokens_address, C.POINTER(C.c_int32)), shape=(count,)).tolist()
        trace = os.environ.get('TS_DSV41_TRACE_DIR')
        self.events.append(('forward', self.active, self.slots[self.active], tokens, trace))
        if self.fail == 'primary' and self.active == 0: return -4
        if self.fail == 'trace_exception' and self.active == 1: raise RuntimeError('injected trace execution exception')
        if self.fail == 'trace_status' and self.active == 1: return -4
        value = np.arange(32, dtype=np.float32) / 100
        value[22 if self.slots[self.active] == 0 else 1] = 10
        if self.active == 1:
            value[5] += self.trace_delta
            if self.fail == 'trace_nonfinite': value[5] = np.nan
        np.ctypeslib.as_array(C.cast(logits_address, C.POINTER(C.c_float)), shape=(32,))[:] = value
        self.slots[self.active] += count
        if self.active == 1 and self.fail == 'trace_wrong_position': self.slots[1] += 1
        if trace and self.fail != 'no_trace_files':
            p = Path(trace)
            p.mkdir(parents=True, exist_ok=True)
            value.tofile(p / 'p000000_v41.40.logits.f32')
        return 0

    def free(self, handle):
        self.events.append(('free_model',))
        self.freed = True
        self.slots.clear()


class TraceChecks(unittest.TestCase):
    def invoke(self, *, fail=None, trace_delta=0, mode='after', inherited=None, extra=(), before=None):
        with tempfile.TemporaryDirectory(prefix='dsv41-trace-abi-') as td:
            p = Path(td)
            tokens, library, output, traces = p / 'tokens.json', p / 'lib-fake', p / 'smoke.json', p / 'trace'
            tokens.write_text('[2,3,4]')
            library.write_bytes(b'fake library bytes, never loaded')
            if before: before(p, traces)
            argv = [str(HELPER), str(p / 'unused.gguf'), '--library', str(library), '--tokens', str(tokens),
                    '--output', str(output), '--steps', '2']
            if mode == 'after': argv += ['--trace-after-dir', str(traces)]
            elif mode == 'primary': argv += ['--trace-dir', str(traces)]
            argv += list(extra)
            native = FakeNative(fail, trace_delta)
            env = {} if inherited is None else {'TS_DSV41_TRACE_DIR': inherited}
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', argv), \
                 patch.object(helper.C, 'CDLL', return_value=native), \
                 contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try: code = helper.main()
                except SystemExit as error: code = error.code
                environment_after = dict(os.environ)
            report = json.loads(output.read_text()) if output.exists() else None
            primary = np.load(output.with_suffix('.first-logits.npy')) if output.with_suffix('.first-logits.npy').exists() else None
            traced = np.load(output.with_suffix('.traced-first-logits.npy')) if output.with_suffix('.traced-first-logits.npy').exists() else None
            return code, report, native, primary, traced, environment_after, stderr.getvalue()

    def assert_primary(self, result):
        code, report, native, primary, traced, env, _ = result
        self.assertTrue(report['passed'])
        self.assertEqual(report['generated_tokens'], [22, 1])
        self.assertFalse(report['first_forward_traced'])
        self.assertEqual(int(primary.argmax()), 22)
        forwards = [e for e in native.events if e[0] == 'forward']
        self.assertEqual(forwards[:2], [('forward', 0, 0, [2, 3, 4], None), ('forward', 0, 3, [22], None)])
        self.assertTrue(native.freed)
        self.assertEqual(env, {})

    def test_default_keeps_original_generation_and_no_slot_calls(self):
        result = self.invoke(mode=None)
        self.assert_primary(result)
        self.assertEqual(result[0], 0)
        self.assertNotIn('trace_diagnostic', result[1])
        self.assertFalse(any(e[0] in ('alloc', 'select', 'slot_free') for e in result[2].events))

    def test_success_replays_fresh_slot_and_restores_before_free(self):
        result = self.invoke()
        self.assert_primary(result)
        self.assertEqual(result[0], 0)
        diagnostic = result[1]['trace_diagnostic']
        self.assertTrue(diagnostic['complete'])
        self.assertTrue(diagnostic['bitwise_equal'])
        self.assertTrue(diagnostic['primary_slot_restored'])
        self.assertEqual(diagnostic['primary_n_past_before'], 4)
        self.assertEqual(diagnostic['n_past_after'], 3)
        self.assertEqual(result[2].events[-3:], [('select', 0), ('slot_free', 1), ('free_model',)])
        traced_forward = [e for e in result[2].events if e[0] == 'forward'][-1]
        self.assertEqual(traced_forward[1:4], (1, 0, [2, 3, 4]))
        self.assertIsNotNone(traced_forward[4])
        np.testing.assert_array_equal(result[3], result[4])

    def test_trace_numerical_difference_is_reported_not_reclassified_as_execution_failure(self):
        result = self.invoke(trace_delta=0.125)
        self.assert_primary(result)
        self.assertEqual(result[0], 0)
        diagnostic = result[1]['trace_diagnostic']
        self.assertTrue(diagnostic['complete'])
        self.assertFalse(diagnostic['bitwise_equal'])
        self.assertFalse(diagnostic['strict_allclose_atol_rtol_2e_5'])
        self.assertAlmostEqual(diagnostic['max_absolute_error'], 0.125, places=6)

    def assert_diagnostic_failure(self, fail):
        result = self.invoke(fail=fail)
        self.assert_primary(result)
        self.assertEqual(result[0], 1)
        self.assertFalse(result[1]['trace_diagnostic']['complete'])
        return result

    def test_allocation_failure_keeps_primary(self): self.assert_diagnostic_failure('alloc')
    def test_allocation_exception_keeps_primary(self): self.assert_diagnostic_failure('alloc_exception')
    def test_fresh_slot_selection_failure_cleans_inactive_slot(self):
        result = self.assert_diagnostic_failure('select1')
        self.assertIn(('slot_free', 1), result[2].events)
    def test_fresh_slot_must_start_empty(self): self.assert_diagnostic_failure('fresh_not_zero')
    def test_trace_forward_status_failure_keeps_primary(self): self.assert_diagnostic_failure('trace_status')
    def test_trace_forward_exception_keeps_primary(self): self.assert_diagnostic_failure('trace_exception')
    def test_trace_nonfinite_logits_rejected(self): self.assert_diagnostic_failure('trace_nonfinite')
    def test_trace_missing_files_rejected(self): self.assert_diagnostic_failure('no_trace_files')
    def test_trace_wrong_consumed_position_rejected(self): self.assert_diagnostic_failure('trace_wrong_position')
    def test_primary_restore_failure_defers_active_slot_cleanup_to_model_free(self):
        result = self.assert_diagnostic_failure('select0')
        self.assertNotIn(('slot_free', 1), result[2].events)
        self.assertIn('cleanup_errors', result[1]['trace_diagnostic'])
    def test_slot_free_failure_retains_primary_and_destroys_model(self):
        result = self.assert_diagnostic_failure('slot_free')
        self.assertIn('cleanup_errors', result[1]['trace_diagnostic'])
    def test_primary_failure_does_not_attempt_diagnostic(self):
        result = self.invoke(fail='primary')
        self.assertEqual(result[0], 1)
        self.assertFalse(result[1]['passed'])
        self.assertNotIn('trace_diagnostic', result[1])
        self.assertFalse(any(e[0] == 'alloc' for e in result[2].events))
        self.assertTrue(result[2].freed)
    def test_conflicting_modes_rejected_before_loading(self):
        result = self.invoke(extra=('--trace-dir', '/unused'))
        self.assertEqual(result[0], 2)
        self.assertFalse(result[2].events)
    def test_inherited_trace_rejected_before_loading(self):
        result = self.invoke(inherited='/inherited')
        self.assertEqual(result[0], 2)
        self.assertFalse(result[2].events)
        self.assertEqual(result[5], {'TS_DSV41_TRACE_DIR': '/inherited'})
    def test_empty_inherited_trace_rejected_because_native_checks_presence(self):
        result = self.invoke(inherited='')
        self.assertEqual(result[0], 2)
        self.assertFalse(result[2].events)
    def test_existing_trace_directory_rejected(self):
        result = self.invoke(before=lambda p, traces: traces.mkdir())
        self.assertEqual(result[0], 2)
        self.assertFalse(result[2].events)
    def test_original_primary_trace_mode_still_works_and_restores_environment(self):
        result = self.invoke(mode='primary', inherited='/prior')
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1]['generated_tokens'], [22, 1])
        self.assertTrue(result[1]['first_forward_traced'])
        self.assertNotIn('trace_diagnostic', result[1])
        forwards = [e for e in result[2].events if e[0] == 'forward']
        self.assertIsNotNone(forwards[0][4])
        self.assertIsNone(forwards[1][4])
        self.assertEqual(result[5], {'TS_DSV41_TRACE_DIR': '/prior'})


class CoverageChecks(unittest.TestCase):
    def coverage(self, mutate=None):
        with tempfile.TemporaryDirectory(prefix='dsv41-trace-coverage-') as td:
            p = Path(td)
            for layer in range(40):
                for stage in ('attn_output', 'ffn_output'):
                    (p / f'p000000_v41.{layer:02d}.{stage}.f32').write_bytes(bytes(3 * 8 * 4))
            (p / 'p000000_v41.40.logits.f32').write_bytes(bytes(16 * 4))
            if mutate: mutate(p)
            return runner.trace_coverage(p, 3, 16, embedding_size=8)

    def test_all_forty_layers_and_terminal_logits_required(self):
        result = self.coverage()
        self.assertTrue(result['complete'])
        self.assertEqual(result['required_files'], 81)

    def test_one_nonempty_stage_missing_is_incomplete(self):
        result = self.coverage(lambda p: (p / 'p000000_v41.39.ffn_output.f32').unlink())
        self.assertFalse(result['complete'])
        self.assertEqual(result['missing'], ['p000000_v41.39.ffn_output.f32'])

    def test_nonempty_but_truncated_stage_rejected(self):
        result = self.coverage(lambda p: (p / 'p000000_v41.12.attn_output.f32').write_bytes(b'four'))
        self.assertFalse(result['complete'])
        self.assertEqual(result['wrong_sizes'][0]['actual_bytes'], 4)

    def test_terminal_logit_size_rejected(self):
        result = self.coverage(lambda p: (p / 'p000000_v41.40.logits.f32').write_bytes(bytes(32)))
        self.assertFalse(result['complete'])
        self.assertEqual(result['wrong_sizes'][0]['expected_bytes'], 64)

    def test_runner_trace_flags_exclusive_before_any_inspection(self):
        with patch.object(sys, 'argv', [str(RUNNER), '--trace', '--trace-after']), \
             patch.object(runner.subprocess, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: runner.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (TraceChecks, CoverageChecks))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'scope': __doc__, 'checks_run': result.testsRun, 'passed': result.wasSuccessful(),
              'failures': len(result.failures), 'errors': len(result.errors),
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [HELPER, RUNNER, Path(__file__).resolve()]}}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return not result.wasSuccessful()


if __name__ == '__main__':
    raise SystemExit(main())
