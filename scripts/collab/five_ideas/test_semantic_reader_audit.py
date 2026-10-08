import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.collab.five_ideas.analyze_quantum_semantic_reader_screen import check_receipts, join_cases, norm, stats


class SemanticAuditTests(unittest.TestCase):
    def test_small_parameter_norm_uses_float64(self):
        values = np.asarray([1e-30, -1e-30], dtype=np.float32)
        self.assertEqual(float(np.linalg.norm(values)), 0)
        self.assertGreater(norm(values), 1e-30)

    def test_nonfinite_statistics_rejected(self):
        with self.assertRaises(ValueError):
            stats([float("nan")])

    def test_zero_signal_is_counted(self):
        result = stats(np.zeros(50))
        self.assertEqual(result["nonzero"], 0)
        self.assertEqual(result["max"], 0)

    def test_wrong_population_rejected(self):
        with self.assertRaises(ValueError):
            join_cases({"cases": []}, {"cases": []})

    def test_manifest_and_receipt_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"fixture"
            digest = hashlib.sha256(content).hexdigest()
            (root / "summary.json").write_bytes(content)
            manifest = {"target_directory": "five_ideas/fixture/001", "exchange_files": [{"name": "summary.json", "sha256": digest, "bytes": len(content)}]}
            (root / "upload_manifest.json").write_text(json.dumps(manifest))
            for name in ("summary.json", "upload_manifest.json"):
                path = root / name
                receipt = {"status": "stored", "path": manifest["target_directory"] + "/" + name,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
                (root / (name + ".upload.json")).write_text(json.dumps(receipt))
            self.assertEqual(len(check_receipts(root)[1]), 2)
            (root / "summary.json").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                check_receipts(root)


if __name__ == "__main__":
    unittest.main()
