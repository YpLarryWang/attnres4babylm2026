"""Exercise export, model loading, and external conversion-cache metadata."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]
from model import GPT, GPTConfig
from diag_dev_series import ensure_converted


class ExportTest(unittest.TestCase):
    def test_export_load_and_cache_for_three_architectures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in ("baseline", "static", "dynamic"):
                with self.subTest(arm=arm):
                    args = dict(
                        n_layer=4, n_head=2, n_embd=32, block_size=32,
                        vocab_size=16000, bias=False, dropout=0.0,
                        use_rmsnorm=True, use_swiglu=True, use_rope=True,
                        use_attn_gate=True, use_attn_res=arm != "baseline",
                        use_static_attn_res=arm == "static", attn_res_block_size=2,
                    )
                    native = GPT(GPTConfig(**args)).eval()
                    checkpoint = root / f"{arm}.pt"
                    torch.save(dict(model=native.state_dict(), model_args=args,
                                    iter_num=2, checkpoint_role="final"), checkpoint)
                    converter = ROOT / "eval/convert_nanogpt_to_hf.py"
                    output = root / arm
                    subprocess.run(
                        [sys.executable, str(converter), "--ckpt", str(checkpoint),
                         "--tokenizer", str(ROOT / "tokenizers/10m/tokenizer.json"),
                         "--out", str(output)], check=True, capture_output=True,
                    )
                    self.assertNotIn("nanogpt_checkpoint", json.loads((output / "config.json").read_text()))
                    self.assertFalse((output / "checkpoint_source.json").exists())
                    tokenizer = AutoTokenizer.from_pretrained(output)
                    hf = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).eval()
                    inputs = tokenizer("The child reads a book", return_tensors="pt")
                    with torch.no_grad():
                        expected, _ = native(inputs.input_ids, inputs.input_ids)
                        actual = hf(**inputs).logits
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    item = dict(checkpoint_path=str(checkpoint), blimp_model_name=arm, iter_num=2)
                    kwargs = dict(cache_root=root / "cache", converter=converter,
                                  tokenizer=ROOT / "tokenizers/10m/tokenizer.json",
                                  eval_python=sys.executable, dtype="float32")
                    cached = ensure_converted(item, **kwargs)
                    self.assertEqual(ensure_converted(item, **kwargs), cached)
                    metadata = json.loads((root / "cache" / f"{arm}.source.json").read_text())
                    self.assertEqual(metadata["filename"], checkpoint.name)
                    self.assertFalse((cached / "checkpoint_source.json").exists())


if __name__ == "__main__":
    unittest.main()
