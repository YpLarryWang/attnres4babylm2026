import importlib.util, sys, unittest
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis"), str(ROOT / "eval/hf_nanogpt")]
from model import GPT, GPTConfig
from modeling_nanogpt import NanoGPTConfig, NanoGPTForCausalLM
from diag_make_figures import pava_nonincreasing, interpolate_on_nonincreasing_x

spec = importlib.util.spec_from_file_location(
    "masked_model", ROOT / "analysis/masking/modeling_nanogpt.py"
)
mask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mask)


class CoreTest(unittest.TestCase):
    def test_all_three_architectures_have_identical_hf_logits(self):
        for arm in ["baseline", "static", "dynamic"]:
            with self.subTest(arm=arm):
                torch.manual_seed(1337)
                args = dict(
                    n_layer=4,
                    n_head=2,
                    n_embd=64,
                    block_size=32,
                    vocab_size=128,
                    bias=False,
                    dropout=0.0,
                    use_rmsnorm=True,
                    use_swiglu=True,
                    use_rope=True,
                    use_attn_gate=True,
                    use_attn_res=arm != "baseline",
                    use_static_attn_res=arm == "static",
                    attn_res_block_size=2,
                )
                native = GPT(GPTConfig(**args)).eval()
                hf = NanoGPTForCausalLM(NanoGPTConfig(**args)).eval()
                hf.load_state_dict(native.state_dict(), strict=True)
                x = torch.randint(128, (2, 16))
                with torch.no_grad():
                    a, _ = native(x, x)
                    b = hf(x).logits
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_plateau_boundary_and_no_extrapolation(self):
        x = pava_nonincreasing([4.4, 4.2, 4.1, 4.14])
        np.testing.assert_allclose(x, [4.4, 4.2, 4.12, 4.12])
        y = [60, 64, 66, 70]
        self.assertAlmostEqual(interpolate_on_nonincreasing_x(4.12, x, y), 68)
        self.assertAlmostEqual(interpolate_on_nonincreasing_x(4.16, x, y), 65)
        self.assertIsNone(interpolate_on_nonincreasing_x(4.08, x, y))

    def test_random_control_preserves_mask_counts_and_embedding(self):
        for block_start in [False, True]:
            for site in ["q1", "q2"]:
                args = dict(
                    num_blocks=5, layer_idx=8, router_site=site, block_start=block_start
                )
                old = mask.build_attn_res_keep_mask(
                    **args, mode="nonrecent", control_seed=20260718
                )
                for seed in range(20260718, 20260723):
                    random = mask.build_attn_res_keep_mask(
                        **args, mode="random_count_matched", control_seed=seed
                    )
                    self.assertEqual(int(old.sum()), int(random.sum()))
                    self.assertTrue(random[0])
                    self.assertTrue(
                        random[-1] if not (site == "q1" and block_start) else True
                    )


if __name__ == "__main__":
    unittest.main()
