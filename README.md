# Attention Residuals for BabyLM 2026

Code and pretrained models for **What Does Input-Dependent Depth Routing Buy in Data-Limited Language Modeling?**

We study depth routing in language models trained on the BabyLM 10M- and 100M-word tracks. The repository provides three architectures: a residual baseline, **Block Attention Residuals** with input-dependent routing, and a **static-routing control** with learned input-independent source weights. It includes training, evaluation, runtime measurement, and the analyses used in the paper.

[Pretrained models](https://huggingface.co/collections/ypwhere/attnres4babylm2026-6aa30019b9b7940a86a4cbc7) · [Reproduction guide](docs/reproduction.md) · [Model index](docs/models.json)

## Models

All models use hidden size 512, eight attention heads, a 16,000-token byte-level BPE vocabulary, and a context length of 512 tokens. Each configuration is available for seeds 1337, 1338, and 1339.

| Training track | Transformer layers | Architectures | Models |
|---|---:|---|---:|
| 10M words | 16 | Baseline, Block AttnRes | 6 |
| 10M words | 32 | Baseline, Block AttnRes, static routing | 9 |
| 100M words | 32 | Baseline, Block AttnRes | 6 |

The collection contains 21 final checkpoints with their tokenizers and Transformers model code. Download a specific model with:

```bash
.venv-eval/bin/python scripts/download_model.py --track 10m --depth 32 --arm dynamic --seed 1337
```

The downloader selects the fixed revision in `docs/models.json`. Each model includes `SHA256SUMS`, whose lines pair a SHA256 digest with the exact downloaded filename. In the index, `weights_file` names the weight file (`model.safetensors`) and `weights_sha256` checks that file; `revision` identifies the complete repository version. See the model card for a Transformers loading example.

## Installation

For Linux/CUDA training and evaluation, use Python 3.12 and separate environments. Training uses PyTorch 2.7.1 with CUDA 12.8; the paper's evaluator uses PyTorch 2.7.0 with CUDA 12.6 and Transformers 4.51.3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

```bash
uv venv --python 3.12 .venv-train
uv pip install --python .venv-train/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-train/bin/python -r environments/train.txt

mkdir -p external
git clone https://github.com/babylm-org/babylm-eval.git external/babylm-eval
git -C external/babylm-eval checkout 3d57ddc8c40ee795c0b5e41b3a20251a9457a593
uv venv --python 3.12 .venv-eval
uv pip install --python .venv-eval/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-eval/bin/python -r environments/eval.txt
```

Run the following commands from this repository's root directory. Training and model evaluation require a CUDA GPU; analysis of the included results can run on a CPU.

## Train

Log in to Weights & Biases using `.venv-train/bin/wandb login`, then prepare the official data splits and start a run:

```bash
.venv-train/bin/python scripts/prepare_data.py --track 10m
.venv-train/bin/python scripts/train.py --track 10m --depth 32 --arm dynamic --seed 1337
```

Use `--arm baseline` or `--arm static` to select a control. Use `--depth 16` for the 10M baseline and dynamic models. To train on 100M words, use `--track 100m` in both commands and `--depth 32`.

Data preparation downloads the official train, dev, and test splits, applies the paper's preprocessing, and uses the included tokenizer. Training retains the final checkpoint by default. Add `--checkpoints trajectory` to save the word- and token-exposure milestones used for developmental analyses.

## Evaluate

Export a native checkpoint to Transformers format:

```bash
.venv-eval/bin/python eval/convert_nanogpt_to_hf.py \
  --ckpt PATH_TO_CHECKPOINT \
  --tokenizer tokenizers/10m/tokenizer.json \
  --out hf-models/MODEL
```

Use `tokenizers/100m/tokenizer.json` for a 100M model. Prepare and run the full six-suite zero-shot evaluation:

```bash
.venv-eval/bin/python scripts/prepare_evaluation.py --eval-repo external/babylm-eval --profile full
.venv-eval/bin/python scripts/prepare_full_extras.py --eval-repo external/babylm-eval
.venv-eval/bin/python scripts/evaluate.py --model hf-models/MODEL --eval-repo external/babylm-eval --profile full
```

EWoK requires accepting its [dataset access conditions](https://huggingface.co/datasets/ewok-core/ewok-core-1.0) and logging in with an authorized Hugging Face account. For a shorter evaluation, use `--profile fast` in the preparation and evaluation commands. The fast profile covers BLiMP, Supplement, EWoK, and Entity Tracking. Add `--reading` to evaluate human reading behavior separately.

The six-suite mean uses BLiMP, Supplement, EWoK, Entity Tracking, COMPS, and Global PIQA, with Global PIQA's parallel and nonparallel scores averaged first. Reading and AoA are separate metrics.

## Reproduce the analyses

These commands work on Linux or macOS without CUDA, model downloads, or a training environment.

```bash
uv venv --python 3.12 .venv-analysis
uv pip install --python .venv-analysis/bin/python numpy==2.2.5 matplotlib==3.10.3
.venv-analysis/bin/python analysis/endpoints.py
.venv-analysis/bin/python analysis/reproduce.py
.venv-analysis/bin/python analysis/figures.py --skip-previews
```

The [results guide](results/paper/README.md) defines the tables, units, and masking contrasts. These commands use the included numerical results to produce endpoint summaries, developmental and masking statistics, and seven diagnostic figures under `outputs/`. Endpoint standard deviations are computed across the three seed-level results.

Omit `--skip-previews` to also generate PNG previews; this requires Poppler (`brew install poppler` on macOS or `apt-get install poppler-utils` on Ubuntu). The [reproduction guide](docs/reproduction.md) describes matched-NLL interpolation, routing interventions, and runtime measurement.

## Repository structure

| Path | Contents |
|---|---|
| `model.py`, `train.py` | Model architectures and training loop |
| `scripts/` | Data preparation, training, evaluation, and benchmarking entry points |
| `config/`, `tokenizers/` | Training settings, checkpoint schedules, and tokenizers |
| `eval/` | Transformers export and held-out NLL evaluation |
| `analysis/` | Endpoint aggregation, developmental analyses, interventions, and figures |
| `results/paper/` | Numerical results used in the paper |
| `environments/` | Python dependency versions |
| `tests/` | Model equivalence and analysis checks |

See [testing](docs/testing.md) for CPU checks and a short GPU training/evaluation test.

## Acknowledgments and license

The implementation builds on [nanoGPT](https://github.com/karpathy/nanoGPT) and uses the [official BabyLM evaluation pipeline](https://github.com/babylm-org/babylm-eval). Code is distributed under the [MIT license](LICENSE). Model licensing is specified in the individual model cards.
