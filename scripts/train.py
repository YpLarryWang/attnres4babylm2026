#!/usr/bin/env python3
"""Run a paper configuration or a six-update pipeline smoke test on one GPU."""

import argparse, json, os, shlex, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--track", choices=["10m", "100m"], default="10m")
p.add_argument("--depth", type=int, choices=[16, 32], default=32)
p.add_argument("--arm", choices=["baseline", "static", "dynamic"], default="dynamic")
p.add_argument("--seed", type=int, choices=[1337, 1338, 1339], default=1337)
p.add_argument("--smoke", action="store_true")
p.add_argument("--dry-run", action="store_true")
p.add_argument("--checkpoints", choices=["endpoint", "trajectory"], default="endpoint")
a = p.parse_args()
if (a.track == "100m" and a.depth != 32) or (
    a.arm == "static" and (a.track != "10m" or a.depth != 32)
):
    p.error(
        "This configuration was not part of the paper; do not silently extend the experiment matrix."
    )
batch, accum, steps, warmup = (
    (8, 64, 471, 40) if a.track == "10m" else (16, 32, 4797, 100)
)
block = 4 if a.depth == 16 else 8
name = f"bl{a.track}-d512L{a.depth}-do0.1-gate"
if a.arm != "baseline":
    name += f"-attnres{block}"
if a.arm == "static":
    name += "-static"
name += f"-offdev-{a.checkpoints}-b{batch}ga{accum}-s{a.seed}"
dataset = "babylm_officialdev" if a.track == "10m" else "babylm_100m_officialdev"
schedule = f"config/generated/{a.track}-b{batch}ga{accum}-s{a.seed}.json"
flags = dict(
    dataset=dataset,
    n_layer=a.depth,
    n_embd=512,
    n_head=8,
    block_size=512,
    bias=False,
    use_rmsnorm=True,
    use_swiglu=True,
    swiglu_mult=8 / 3,
    use_rope=True,
    use_attn_gate=True,
    use_attn_res=a.arm != "baseline",
    use_static_attn_res=a.arm == "static",
    attn_res_block_size=block,
    use_muon=False,
    use_hybrid=False,
    sampler="shuffle",
    sampler_seed=a.seed,
    seed=a.seed,
    dropout=0.1,
    batch_size=batch,
    gradient_accumulation_steps=accum,
    eval_batch_size=32,
    eval_interval=50,
    eval_iters=50,
    max_iters=steps,
    lr_decay_iters=steps,
    warmup_iters=warmup,
    checkpoint_schedule=schedule,
    endpoint_only=a.checkpoints == "endpoint",
    compile=True,
    wandb_log=True,
    wandb_project="babylm",
    wandb_resume="never",
)
if a.smoke:
    name += "-smoke-u6-save2-4-6-eval2x8-nocompile"
    flags.update(
        dataset="babylm_smoke",
        max_iters=6,
        checkpoint_schedule="",
        endpoint_only=False,
        save_iters=[2, 4],
        save_best_checkpoint=False,
        eval_interval=6,
        eval_iters=2,
        eval_batch_size=8,
        compile=False,
        log_interval=1,
    )
# Keep the paper LR schedule in the smoke test: this tests the first six updates, not a six-update decay schedule.
flags.update(
    wandb_run_name=name,
    wandb_run_id=name,
    out_dir=f"outputs/training/{name}",
    experiment_log_path="outputs/experiments.jsonl",
)
cmd = [
    sys.executable,
    "train.py",
    "config/train_babylm.py",
    *[
        f"--{k}={v!r}" if isinstance(v, (list, str)) else f"--{k}={v}"
        for k, v in flags.items()
    ],
]
# nanoGPT configurator expects unquoted strings; repr is only needed for list syntax.
cmd = [
    c.replace("='", "=", 1)[:-1] if "='" in c and c.endswith("'") else c for c in cmd
]
print(shlex.join(cmd), flush=True)
if a.dry_run:
    raise SystemExit(0)
os.chdir(ROOT)
if Path(flags["out_dir"]).exists():
    raise SystemExit(
        "Output already exists; choose a distinct experiment identity instead of overwriting."
    )
if not (ROOT / "data" / flags["dataset"] / "train.bin").exists():
    raise SystemExit(
        "Run scripts/prepare_data.py first; smoke uses an explicit symlink to its official data."
    )
subprocess.run(cmd, check=True)
if a.smoke:
    manifest = json.loads(
        (Path(flags["out_dir"]) / "checkpoint_manifest.json").read_text()
    )
    assert len(list(Path(flags["out_dir"]).glob("*.pt"))) == 3, (
        "Smoke must retain exactly three checkpoints"
    )
    print(
        json.dumps(
            {
                "smoke_complete": True,
                "out_dir": flags["out_dir"],
                "roles": manifest["roles"],
            }
        )
    )
