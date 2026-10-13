"""Integration/write/readback tests with a random one-layer DPR architecture.

These mocks test engineering plumbing, never certify the pinned checkpoint.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import DPRConfig, DPRReader
from scripts.collab.five_ideas import run_quantum_reader_rotation_preflight as runner


class FakeTokenizer:
    all_special_ids = [0, 1, 2]
    sep_token_id = 2
    def __call__(self, questions, texts, **_kwargs):
        rows = [[1, 3, 4, 2, 5, 6, 7+(i%30), 2] for i in range(len(texts))]
        return {"input_ids": torch.tensor(rows), "attention_mask": torch.ones(len(rows), 8, dtype=torch.long)}


def architecture_fixture(root):
    """Persist a clearly marked random-weight architecture fixture for auditing."""
    cfg = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    root = Path(root)
    snapshot = root/cfg["reader"]["revision"]
    snapshot.mkdir(parents=True, exist_ok=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reader = DPRReader(DPRConfig(vocab_size=64, hidden_size=768, num_hidden_layers=1,
            num_attention_heads=12, intermediate_size=768, max_position_embeddings=64,
            hidden_dropout_prob=0., attention_probs_dropout_prob=0.))
    original_sha = runner.sha
    def mocked_sha(path):
        return cfg["reader"]["config_file_sha256"] if Path(path) == snapshot/"config.json" else original_sha(path)
    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot)), \
         patch("transformers.DPRReader.from_pretrained", return_value=reader), \
         patch("transformers.DPRReaderTokenizer.from_pretrained", return_value=FakeTokenizer()), \
         patch.object(runner, "sha", side_effect=mocked_sha), contextlib.redirect_stdout(io.StringIO()):
        runner.run(SimpleNamespace(verify_only=None, device="cpu", output_root=root/"outputs"))
    directory = sorted((root/"outputs").iterdir())[-1]
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    manifest["model_identity"] = {"model_id": "random-one-layer-DPRReader-architecture-fixture", "checkpoint_pinned": False,
        "seed": 42, "tokenizer": "explicit fixed synthetic token IDs", "formal_readiness": "not_certified"}
    runner.write_json(directory/"manifest.json", manifest)
    trace = json.loads((directory/"sample_trace.json").read_text(encoding="utf-8"))
    trace["experiment"]["model_identity"] = manifest["model_identity"]
    trace["experiment"]["dataset"] = "explicit synthetic tokens and random one-layer DPRReader; not pinned checkpoint"
    runner.write_json(directory/"sample_trace.json", trace)
    validation = json.loads((directory/"validation.json").read_text(encoding="utf-8"))
    validation["pinned_reader_readiness"] = "not_certified"
    runner.write_json(directory/"validation.json", validation)
    return directory


class PreflightTests(unittest.TestCase):
    def test_full_fixture_reopen_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            directory = architecture_fixture(task_tmp)
            result = runner.verify(directory)
            self.assertEqual(result["optimizer_update_replays"], 4)
            self.assertEqual(result["head_replays"], 8)
            self.assertFalse(result["task_utility_tested"])
            trace = json.loads((directory/"sample_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(len(trace["samples"]), 2)
            for rep in trace["samples"][0]["representations"]:
                self.assertEqual(len(rep["axis_semantics"]), len(rep["shape"]))
            # Inventory omission cannot turn into a self-declared complete trace.
            trace["task_contract"]["required_value_representations"].remove("hidden")
            runner.write_json(directory/"sample_trace.json", trace)
            with self.assertRaisesRegex(ValueError, "omitted mechanism"):
                runner.verify(directory)


if __name__ == "__main__": unittest.main()
