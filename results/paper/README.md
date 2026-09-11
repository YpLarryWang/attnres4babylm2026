# Results used in the paper

These tables contain model scores and training-exposure measurements. Run `analysis/endpoints.py`, `analysis/reproduce.py`, and `analysis/figures.py` from the repository root to aggregate the results and generate figures.

| File | Contents |
|---|---|
| `offdev_results.csv` | Final-checkpoint scores for the 21 models |
| `diag_supp_checkpoint_inventory.csv` | Developmental checkpoint settings and exposure counts |
| `diag_supp_behavior_long.csv` | BLiMP scores by checkpoint, linguistic category, and routing intervention |
| `diag_supp_dev_loss.csv` | Full-dev NLL at developmental checkpoints |
| `diag_supp_position_nll.csv` | NLL sums and token counts by context position |
| `diag_supp_trajectory_summary.csv` | Equal-exposure and matched-NLL developmental summaries |
| `diag_supp_100m_fresh_summary.csv` | 100M developmental comparisons at 10M–90M words of exposure |
| `diag_supp_masking_draws.csv` | Individual random-control draws and paired contrasts |
| `diag_supp_masking_contrasts.csv` | Contrasts averaged across control draws and training seeds |
| `diag_supp_masking_correlations.csv` | Correlations across linguistic categories |
| `benchmark-summary.json` | Measured optimizer-update time and throughput |
| `compute-metadata.json` | Model dimensions, routing-source counts, and compute workload |

## Units and identifiers

Accuracy is reported as a percentage; differences ending in `_pp` are percentage points. NLL is in nats per token. `words_seen` and `tokens_seen` are cumulative training exposure, whereas `train_words` and `corpus` identify the 10M- or 100M-word source corpus. The 100M table's `fresh` label refers to exposure before completing the first pass through that corpus. `run_name` and `blimp_model_name` are identifiers used to join records; choose downloadable models by track, depth, architecture, and seed in `docs/models.json`.

`architecture=attnres` denotes dynamic Block AttnRes. A `mask_mode` of `none` is unmasked; `nonrecent` masks completed-block sources of age at least two; `random_count_matched` masks the same number of randomly selected completed-block sources at each target sublayer. A `training_seed` of `mean` denotes an aggregate across training seeds.

## Masking contrasts

With unmasked accuracy $A_U$, nonrecent-masked accuracy $A_N$, paired baseline accuracy $A_B$, and random-masked accuracy $A_R$:

- `raw_nonrecent_cost_pp` is $A_U-A_N$.
- `random_deletion_cost_pp` is $A_U-A_R$.
- `nonrecent_excess_cost_pp` is $A_R-A_N$; aggregated rows use the mean over the five random draws.
- `attnres_relative_gain_pp` is $A_U-A_B$.
- `absolute_ability_pp` is $A_U-50$.

The `draw_aggregate_min_pp` and `draw_aggregate_max_pp` fields give the range across five control draws after averaging each draw across training seeds. They are not confidence intervals. Correlation fields use the same distinction between the mean control and individual draws.

## Endpoint aggregation and file hashes

The main six-suite mean averages BLiMP, Supplement, EWoK, Entity Tracking, COMPS, and Global PIQA. Global PIQA first averages its parallel and nonparallel scores. `analysis/endpoints.py` computes this mean separately for each training seed, then reports the mean and sample standard deviation across three seeds. The source table also includes `avg5` (the five suites excluding Global PIQA) and `reliable4` (those five excluding Entity Tracking); these are not the paper's six-suite mean. Reading and AoA remain separate measures.

Data hashes identify specific input files: `train_bin_sha256` checks `train.bin`, `val_bin_sha256` checks `val.bin`, `word_map_sha256` checks `train.word_starts.uint8`, and `tokenizer_sha256` checks the corresponding track's `tokenizers/10m/tokenizer.json` or `tokenizers/100m/tokenizer.json`. For downloadable model weights, use `weights_file` and `weights_sha256` in `docs/models.json`, or the model package's filename-by-filename `SHA256SUMS`.
