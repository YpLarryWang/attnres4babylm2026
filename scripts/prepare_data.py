#!/usr/bin/env python3
"""Restore official splits with the original tokenizer; build exposure schedules."""

import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--track", choices=["10m", "100m"], default="10m")
p.add_argument("--skip-test", action="store_true")
a = p.parse_args()
d = (
    ROOT
    / "data"
    / ("babylm_officialdev" if a.track == "10m" else "babylm_100m_officialdev")
)
(d / "tokenizer").mkdir(parents=True, exist_ok=True)


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


shutil.copy2(
    ROOT / "tokenizers" / a.track / "tokenizer.json", d / "tokenizer/bpe-16000.json"
)
manifest = d / "source_manifest.json"
if manifest.exists():
    saved = json.loads(manifest.read_text())
    assert saved["track"] == a.track
    for split, meta in saved["splits"].items():
        for item in meta["files"]:
            assert (
                hashlib.sha256(
                    (d / "raw" / split / item["filename"]).read_bytes()
                ).hexdigest()
                == item["sha256"]
            ), "Existing raw data changed"
else:
    run("data/babylm/fetch_offdev.py", "--track", a.track, "--data-dir", d)
for src, dst in [("train", "train"), ("dev", "val")]:
    run(
        "data/babylm/clean.py",
        "--input-split",
        src,
        "--raw-dir",
        d / "raw" / src,
        "--out-dir",
        d / "clean" / dst,
    )
run("data/babylm/prepare.py", "--data-dir", d)
if not a.skip_test:
    test_manifest = d / "test_source_manifest.json"
    if test_manifest.exists():
        for item in json.loads(test_manifest.read_text())["files"]:
            assert (
                hashlib.sha256(
                    (d / "raw/test" / item["filename"]).read_bytes()
                ).hexdigest()
                == item["sha256"]
            ), "Existing test data changed"
    else:
        run("data/babylm/fetch_offtest.py", "--data-dir", d)
    run(
        "data/babylm/clean.py",
        "--input-split",
        "test",
        "--raw-dir",
        d / "raw/test",
        "--out-dir",
        d / "clean/test",
    )
    tokenized = d / "test_manifest.json"
    if tokenized.exists():
        saved = json.loads(tokenized.read_text())
        assert (
            hashlib.sha256((d / "test.bin").read_bytes()).hexdigest()
            == saved["bin"]["sha256"]
        )
        assert (
            hashlib.sha256((d / "tokenizer/bpe-16000.json").read_bytes()).hexdigest()
            == saved["tokenizer"]["sha256"]
        )
        for item in saved["clean_inputs"]:
            assert (
                hashlib.sha256(
                    (d / "clean/test" / f"{item['source']}.txt").read_bytes()
                ).hexdigest()
                == item["sha256"]
            )
    else:
        run("data/babylm/prepare_test.py", "--data-dir", d)
b, ga, n = (8, 64, 471) if a.track == "10m" else (16, 32, 4797)
(ROOT / "config/generated").mkdir(exist_ok=True)
for seed in [1337, 1338, 1339]:
    run(
        "data/babylm/build_checkpoint_schedule.py",
        "--data-dir",
        d,
        "--batch-size",
        b,
        "--global-grad-accum",
        ga,
        "--max-iters",
        n,
        "--sampler-seed",
        seed,
        "--output",
        ROOT / f"config/generated/{a.track}-b{b}ga{ga}-s{seed}.json",
    )
schedule_name = (
    "bl10m-offdev-b8ga64-dual.json"
    if a.track == "10m"
    else "bl100m-offdev-b32ga16-dual.json"
)
expected = json.loads(
    (ROOT / "config/checkpoint_schedules" / schedule_name).read_text()
)["fingerprints"]
for key, filename in [
    ("train_bin_sha256", "train.bin"),
    ("val_bin_sha256", "val.bin"),
    ("tokenizer_sha256", "tokenizer/bpe-16000.json"),
]:
    assert hashlib.sha256((d / filename).read_bytes()).hexdigest() == expected[key], (
        filename + " differs from paper data"
    )
if a.track == "10m":
    smoke = ROOT / "data/babylm_smoke"
    if not smoke.exists():
        smoke.symlink_to(d.name, target_is_directory=True)
print(
    "Official data prepared; no custom train/dev/test splitting or tokenizer retraining."
)
