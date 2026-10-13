import json
from pathlib import Path
import tempfile
import unittest
from collections import OrderedDict
import torch
from safetensors.torch import load_file
from applications.rag.reader_checkpoint_cache import convert_legacy, load_streaming_reader


class CacheTests(unittest.TestCase):
    def test_streaming_actual_dpr_modules_match_resident_reader(self):
        from transformers import DPRConfig, DPRReader
        from safetensors.torch import save_file
        from applications.rag.quantum_reader_rotation import make_rotation_head, reader_rotation_hook
        config = DPRConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=16, hidden_dropout_prob=0., attention_probs_dropout_prob=0.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            original = DPRReader(config).eval().requires_grad_(False)
        inputs = {"input_ids": torch.tensor([[1,3,2,4,5,2],[1,3,2,6,7,2]]), "attention_mask": torch.ones(2,6,dtype=torch.long)}
        with tempfile.TemporaryDirectory() as task_tmp:
            archive = Path(task_tmp)/"model.safetensors"
            save_file(original.state_dict(), archive)
            streamed, unused = load_streaming_reader(config, archive)
            self.assertEqual(unused, [])
            expected, actual = original(**inputs), streamed(**inputs)
            for name in ("relevance_logits", "start_logits", "end_logits"):
                # Mapped and resident tensor alignment can change CPU FP32
                # accumulation by an ulp; weights remain byte-identical.
                torch.testing.assert_close(getattr(expected,name), getattr(actual,name), rtol=1e-6, atol=1e-7)
            self.assertTrue(all(p.is_meta for n,p in streamed.named_parameters() if n.startswith("span_predictor.encoder.")))
            head = make_rotation_head(streamed.span_predictor.qa_classifier.weight, streamed.span_predictor.qa_classifier.bias)
            qm = torch.tensor([[False,True,False,False,False,False]]*2)
            pm = torch.tensor([[False,False,False,True,True,False]]*2)
            capture = {}
            with reader_rotation_hook(streamed, head, qm, pm, capture):
                result = streamed(**inputs)
                result.relevance_logits.sum().backward()
            self.assertTrue(bool(head.theta.grad.isfinite().all()))
            self.assertGreater(float(head.theta.grad.norm()), 0)
            torch.testing.assert_close(result.start_logits, expected.start_logits, rtol=1e-6, atol=1e-7)

    def test_exact_bytes_and_reuse_and_corruption(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            original = OrderedDict(weight=torch.arange(24, dtype=torch.float32).reshape(6,4), bias=torch.tensor([-1.25,.125]))
            source = root/"pytorch_model.bin"
            torch.save(original, source, _use_new_zipfile_serialization=False)
            receipt = convert_legacy(source, root/"cache")
            values = load_file(root/"cache/model.safetensors")
            for key, value in original.items(): torch.testing.assert_close(value, values[key], rtol=0, atol=0)
            self.assertEqual(receipt, convert_legacy(source, root/"cache"))
            self.assertEqual(receipt["precision"], "unchanged_FP32")
            with (root/"cache/model.safetensors").open("r+b") as stream:
                stream.seek(-1, 2); stream.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "integrity"):
                convert_legacy(source, root/"cache")

    def test_shared_or_nonfloat_storage_rejected(self):
        with tempfile.TemporaryDirectory() as task_tmp:
            root = Path(task_tmp)
            for index, values in enumerate((OrderedDict(a=torch.ones(4, dtype=torch.float64)),
                    OrderedDict(a=torch.arange(6.).reshape(2,3).T))):
                source = root/f"source{index}.bin"
                torch.save(values, source, _use_new_zipfile_serialization=False)
                with self.assertRaises(ValueError): convert_legacy(source, root/f"cache{index}")


if __name__ == "__main__": unittest.main()
