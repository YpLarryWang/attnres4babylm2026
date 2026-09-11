# Reproduction guide

## What is reproducible here

There are two distinct entry points: recompute the paper's statistics from the included numerical results, or rerun training and official evaluation. The former is CPU-only and needs no checkpoint downloads. The latter needs a CUDA GPU and authorized access to any gated benchmark data. The model index records immutable Hugging Face revisions for the 21 trained endpoints.

The supported experiment matrix is 10M/L16 baseline and dynamic, 10M/L32 baseline, dynamic and static, and 100M/L32 baseline and dynamic, each with seeds 1337, 1338 and 1339. Static routing has learned input-independent source logits. Dynamic routing computes source weights from the current representations. The static arm was run only at 10M/L32.

## Data and training

Run the preparation and training commands in the root README from the repository root. Preparation downloads pinned **official train, dev and test releases separately**, uses the original track-specific tokenizer, and applies the original cleaning and tokenization. No held-out split is sampled from training data. Repeated preparation verifies existing raw files against their recorded SHA256 values. Tokenized train/dev and tokenizer hashes must match the included checkpoint schedules for both tracks.

Every update has 262,144 input tokens (context 512). The portable single-GPU configurations use microbatch 8 × accumulation 64 for 10M, and 16 × 32 for 100M. They run 471 and 4,797 optimizer updates respectively. Changing microbatch shape can change floating-point results even when effective batch size is constant. The dependency files and checkpoint schedules record the software versions and exposure counts used in the experiments.

By default only the final checkpoint is retained. To reproduce trajectory analyses, add `--checkpoints trajectory`. `checkpoint_manifest.json` maps checkpoint roles to files. Milestone checkpoints retain model weights and metadata; final checkpoints also include optimizer and RNG state. At L32 these are approximately 465 MB per milestone and 1.4 GB for a full final checkpoint. Training logs configuration and metrics to W&B.

For a short installation check, see [testing](testing.md).

## Official evaluation

Keep the evaluator in its own checkout and environment. The paper scorer SHA is `3d57ddc8c40ee795c0b5e41b3a20251a9457a593`. Scores from other evaluator versions may differ because of changes to datasets or scoring. See [testing](testing.md) for an additional supported evaluator version.

`prepare_evaluation.py` pins and records Hub dataset revision `8d52da9424a9ff30b9e8266c4f751aba9c504233`; `--dataset-revision` allows an explicit override. The fast profile downloads four small zero-shot suites and Reading data. `evaluate.py --profile fast --reading` runs the official entry points without changing the scorer.

For full evaluation, run `prepare_evaluation.py --profile full --eval-repo external/babylm-eval`, then `prepare_full_extras.py --eval-repo external/babylm-eval`, using the evaluation Python environment. The latter runs the official Global PIQA and EWoK download/filter scripts while pinning their dataset revisions to the versions used for the static-routing evaluation. EWoK requires accepting its dataset access conditions and an authorized Hugging Face login. Missing data causes the wrapper to stop. The six full suites are BLiMP, Supplement, EWoK, Entity Tracking, COMPS and Global PIQA. Compute Global PIQA's mean across parallel and nonparallel splits before the six-suite mean.

Reading measures alignment with human reading behavior. It is separate from sentence zero-shot accuracy. AoA analysis requires the developmental checkpoint series, selected with `--checkpoints trajectory`.

Native held-out NLL is available through `eval/full_dev_loss.py --help`. HF conversion uses the correct track tokenizer and float32 weights. The custom wrapper does not implement an efficient autoregressive KV cache.

## Analysis and interventions

`analysis/reproduce.py` recomputes and verifies five included summary tables: developmental trajectories, 100M summaries, individual mask draws, masking contrasts and masking correlations. `analysis/figures.py` writes seven diagnostic figures under `outputs/figures`. The numerical results are stored in `results/paper`; it contains metrics rather than benchmark examples or checkpoints.

PAVA enforces non-increasing baseline NLL over exposure. For a target exactly equal to a flat fitted NLL segment, matched baseline BLiMP is the mean BLiMP of all checkpoints on that segment. For a target strictly between levels, use the adjacent boundary checkpoints and interpolate their original BLiMP values. No extrapolation is performed outside the baseline's fitted NLL range. An exact-plateau value therefore need not equal either one-sided limit.

To rerun a routing intervention on a trained 10M dynamic checkpoint:

```bash
.venv-eval/bin/python analysis/mask.py \
  --checkpoint PATH_TO_DYNAMIC_CHECKPOINT \
  --eval-repo external/babylm-eval \
  --mode nonrecent --output-dir outputs/masks/nonrecent
```

`nonrecent` masks completed-block sources with age ≥ 2. `none` is unmasked. `random_count_matched` removes the same number of completed sources at every target sublayer; rerun with `--mask-seed` 20260718 through 20260722 for the five paper controls. Embedding and partial-block sources remain available. The intervention masks the routers feeding the attention and feed-forward sublayers and leaves the final output router unchanged. Masked logits become −∞ before softmax, so retained weights renormalize. The CLI verifies the instrumented model with masks off before evaluation.

The lower-level `analysis/diag_dev_series.py` and `analysis/diag_parse_results.py` scripts collect and aggregate checkpoint results; inspect each `--help` for its required input inventories. They do not automatically download missing checkpoint trajectories. Use the included numerical results for immediate numerical reproduction, or train with the original checkpoint schedule to generate new trajectories.

The 100M developmental checkpoints were regenerated with the same configurations and data. They are not bitwise identical to the reference runs: comparisons at three reference checkpoints found maximum differences of 0.012 nats in NLL and 0.72 percentage points in overall BLiMP accuracy.

## Runtime and compute

To measure optimizer-update time:

```bash
.venv-train/bin/python scripts/benchmark.py \
  --arm dynamic --warmup 10 --updates 20 --output outputs/runtime/dynamic.json
```

Run baseline, static and dynamic separately on an otherwise idle GPU. Each measured unit is a complete optimizer update: 64 forward/backward microbatches, gradient clipping and AdamW. CUDA synchronization brackets every measured update. Inputs already reside on the GPU; compilation, data loading, evaluation and checkpoint I/O are excluded. Report the hardware/software, per-update mean/median/IQR, and 262,144 divided by mean seconds as tokens/s. `--profile` additionally saves a Chrome trace after timing. A profiler-enabled run is not substituted for clean timing measurements.

The files `results/paper/benchmark-summary.json` and `results/paper/compute-metadata.json` document the actual paper measurements. FLOP estimates and measured elapsed time answer different questions; runtime cannot be inferred just from parameter count. Dense attention/MLP matrix multiplication, routing reductions and training backward passes must be included according to an explicit counting convention. No compute-optimality claim is established by the fixed-exposure results.
