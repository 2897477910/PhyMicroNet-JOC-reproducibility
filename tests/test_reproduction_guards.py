"""Negative controls for complete-output and strict table verification; standard unittest."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduction', ROOT / 'reproduce.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class VerificationGuards(unittest.TestCase):
    def setUp(self):
        (ROOT / 'reproduced').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'reproduced')
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_outputs_cannot_pass(self):
        for name in ['robustness', 'exposure_sensitivity', 'prior_diagnostics', 'external_domain', 'cases/r5_final', 'pka', 'integrated_diagnostics']:
            (self.path / name).mkdir(parents=True)
        with patch.object(r, 'run_command'):
            with self.assertRaisesRegex(AssertionError, 'Missing or unexpected'):
                r.statistics(self.path, [])

    def test_csv_changed_label_fails(self):
        a, b = self.path / 'a.csv', self.path / 'b.csv'
        a.write_text('name,value\na,1.0\n', encoding='utf-8')
        b.write_text('name,value\nb,1.0\n', encoding='utf-8')
        with self.assertRaises(AssertionError):
            r.compare_csv(a, b)

    def test_csv_changed_number_fails(self):
        a, b = self.path / 'a.csv', self.path / 'b.csv'
        a.write_text('value\n1.0\n', encoding='utf-8')
        b.write_text('value\n1.001\n', encoding='utf-8')
        with self.assertRaises(AssertionError):
            r.compare_csv(a, b)

    def test_npz_changed_nan_position_fails(self):
        a, b = self.path / 'a.npz', self.path / 'b.npz'
        np.savez(a, values=np.array([np.nan, 1.0]))
        np.savez(b, values=np.array([1.0, np.nan]))
        with self.assertRaises(AssertionError):
            r.compare_npz(a, b)

    def test_npz_changed_keys_fails(self):
        a, b = self.path / 'a.npz', self.path / 'b.npz'
        np.savez(a, first=np.array([1.0]))
        np.savez(b, second=np.array([1.0]))
        with self.assertRaises(AssertionError):
            r.compare_npz(a, b)


if __name__ == '__main__':
    unittest.main()
