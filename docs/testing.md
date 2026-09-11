# Testing

## CPU checks

Run the tests in the evaluation environment, with Matplotlib installed:

```bash
uv pip install --python .venv-eval/bin/python matplotlib==3.10.3
.venv-eval/bin/python -m unittest discover -s tests
.venv-eval/bin/python analysis/reproduce.py
```

The tests compare the native and Transformers implementations for all three architectures, check the PAVA plateau and interpolation rules, and verify count-matched random masks. The analysis command recomputes five summary tables from the included results and compares them with the archived values.

When modifying the model or exporter, compare the logits of a native checkpoint and its export:

```bash
.venv-eval/bin/python eval/parity_check.py --ckpt PATH_TO_CHECKPOINT --hf hf-models/MODEL
```

This developer check tests whether the two implementations produce the same output for the same input. It is separate from file integrity: use a model's `SHA256SUMS` to check a downloaded package.

## Short GPU test

Prepare the 10M data and the official fast evaluation datasets, then run:

```bash
.venv-train/bin/python scripts/prepare_data.py --track 10m
.venv-eval/bin/python scripts/prepare_evaluation.py --eval-repo external/babylm-eval --profile fast
.venv-train/bin/python scripts/smoke_suite.py \
  --train-python "$PWD/.venv-train/bin/python" \
  --eval-python "$PWD/.venv-eval/bin/python" \
  --eval-repo external/babylm-eval
```

The suite trains each L32 architecture for six optimizer updates, saves checkpoints at updates 2, 4, and 6, and evaluates each final checkpoint on the four fast zero-shot suites and Reading. Dynamic checkpoints at updates 2 and 4 also receive BLiMP evaluation. Outputs are written to `outputs/smoke/`.

Compilation is disabled and the full learning-rate schedule is retained. These short runs check the training and evaluation pipeline; their scores and elapsed times should not be used as trained-model results or throughput measurements. The suite does not run GLUE or AoA.

The implementation has also been tested with official evaluator commit `6f825c291e2c4c78ad33b1935fd64d45f52642dc`. To use that version, create a separate evaluator checkout at that commit and install its `strict/requirements.txt` in a separate environment. The paper's evaluator revision remains pinned in the main README.
