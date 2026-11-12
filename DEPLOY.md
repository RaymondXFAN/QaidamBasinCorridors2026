# Deployment & Reproduction Manual

**Project:** `filmstgat` — an audited, leakage-free reference implementation and diagnostic
study of spatio-temporal graph networks for annual permafrost ground-temperature forecasting.

This manual takes you from a bare machine to regenerated figures, in order. Every command is
copy-pasteable, every expected output is stated, and every failure mode we hit ourselves is
listed in §11.

---

## Contents

1. [What this repository contains](#1-what-this-repository-contains)
2. [System requirements](#2-system-requirements)
3. [Environment setup](#3-environment-setup)
4. [Obtaining the data](#4-obtaining-the-data)
5. [Data cleaning and preparation](#5-data-cleaning-and-preparation)
6. [Code layout](#6-code-layout)
7. [How to run](#7-how-to-run)
8. [How to inspect results](#8-how-to-inspect-results)
9. [How to make figures and tables](#9-how-to-make-figures-and-tables)
10. [Reproducibility audit](#10-reproducibility-audit)
11. [Troubleshooting](#11-troubleshooting)
12. [Packaging and uploading a release](#12-packaging-and-uploading-a-release)
13. [License and citation](#13-license-and-citation)

---

## 1. What this repository contains

The repository is the single source of truth for every number, table and figure in the
manuscript. It contains:

* one training loop, one evaluator, one model family — deliberately no duplicate code paths;
* the corrected evaluation protocol (per-origin descriptors, seeded initialisation,
  temporally disjoint splits) together with the leakage diagnostics that motivate it;
* all result files of the reported runs, so that every figure can be rebuilt **without a GPU
  and without the raw observations**;
* a self-audit script that checks the archive is complete and portable before you trust it.

It does **not** contain the raw observations (see §4).

---

## 2. System requirements

| Task | Requires | Notes |
|---|---|---|
| Offline self-tests | Python ≥ 3.10, numpy, scipy | no data, no GPU, ~10 s |
| Rebuild figures/tables | the above + matplotlib | no data, no GPU, ~1 min |
| Re-run experiments | the above + PyTorch | GPU strongly recommended |

Reference setup used for the reported results: Python 3.12, numpy 1.26, scipy 1.11,
torch 2.4 (CUDA 12.4), single RTX 4090D, 24 GB. A single training run takes 5–30 s on that
GPU and roughly 10–20× longer on CPU.

Disk: the code plus shipped results is ≈ 6 MB; a full experiment tree with datasets and runs
is ≈ 1–2 GB.

---

## 3. Environment setup

```bash
# 3.1 get the code
tar -xzf filmstgat-v2.8.4-release.tar.gz
cd filmstgat-v2.8.4

# 3.2 create an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3.3 install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` installs a CPU/CUDA-agnostic torch wheel. If your driver needs a specific
build, install torch first from <https://pytorch.org/get-started/locally/> and then run
`pip install -r requirements.txt` again.

```bash
# 3.4 verify the installation (no data, no GPU)
bash run_all.sh check
python scripts/repro_audit.py
```

Expected:

```
[tests/test_core.py]            10/10 passed
[tests/test_data_rebuild.py]     8/8 passed
[tests/test_e2_design.py]       11/11 通过
[tests/test_synth.py]            8/8 通过
All offline suites passed.

FAIL=0  WARN=0  PASS=17
```

If `torch` is missing, `tests/test_torch.py` is skipped automatically and the rest still run.

---

## 4. Obtaining the data

### 4.1 What the data are

Annual mean annual ground temperature (MAGT, °C) on a ~1 km pixel grid, 1981–2020, together
with three per-pixel geological descriptors (terrain, thermal properties, land cover).
Full provenance, licensing and the exact file used are in **`docs/DATA.md`**.

### 4.2 Two routes

**Route A — you were given the prepared archives.** Place them at:

```
<ROOT>/datasets_v2/real_magt_data_v2_clean.npz        # five-year protocol (gap = 4)
<ROOT>/datasets_v2_h1/real_magt_data_v2_clean.npz     # one-year protocol (gap = 0)
```

`<ROOT>` is any writable directory; it is where all outputs will go. It is **not** the code
directory.

**Route B — you hold the upstream product.** Build the two archives from the legacy
single-file bundle (see `docs/DATA.md` §4):

```bash
mkdir -p $ROOT/datasets_v2 $ROOT/datasets_v2_h1

python - <<'PY'
from filmstgat.data.real import build_dataset_v2
build_dataset_v2("<legacy.npz>", f"{__import__('os').environ['ROOT']}/datasets_v2",
                 n_samples=45041, k=8, seed=42, T_in=10, gap=4, T_out=1)
build_dataset_v2("<legacy.npz>", f"{__import__('os').environ['ROOT']}/datasets_v2_h1",
                 n_samples=45041, k=8, seed=42, T_in=10, gap=0, T_out=1)
PY
```

### 4.3 Validate before doing anything else

```bash
export ROOT=/path/to/your/workdir
python scripts/check_dataset.py --data $ROOT/datasets_v2/real_magt_data_v2_clean.npz --expect h5
python scripts/check_dataset.py --data $ROOT/datasets_v2_h1/real_magt_data_v2_clean.npz --expect h1
```

Expected: `[PASS] 数据满足训练契约。` This checks keys, shapes, that the three splits are
disjoint, that there are no non-finite values, and that edge indices are in range — in about a
second, instead of failing three minutes into training.

---

## 5. Data cleaning and preparation

Everything below is implemented in `filmstgat/data/real.py::build_dataset_v2` and is
deterministic given the seed. If you use Route A, this has already been done; read it anyway,
because two of these steps are exactly where leakage used to enter.

### 5.1 Pixel eligibility — training period only

A pixel is kept only if it has at least 30 finite annual values **within 1981–2010**:

```python
head = series[:, :30]
keep = np.isfinite(head).sum(axis=1) >= 30
```

Using the full record (as an earlier version did) lets the test period decide which pixels
exist — a subtle form of selection on the target.

### 5.2 Stratified subsampling

Candidates are binned by their training-period mean temperature (10 quantile bins) and sampled
evenly, so the subset spans the thermal range instead of over-representing warm pixels:

```python
stratified_sample(candidates, train_period_mean, n=45041, n_bins=10, seed=42)
```

All experiments then draw a **fixed 500-pixel subset** from this pool with the same procedure
and the same seed, so every model, baseline and floor is evaluated on identical pixels.

### 5.3 Window construction

A sample is one pixel over `T_in = 10` consecutive years. The target is the value `gap + 1`
years after the last input year:

| Protocol | `gap` | Input window ends | Target |
|---|---|---|---|
| five-year (h=5) | 4 | *y − 5* | *y* |
| one-year (h=1) | 0 | *y − 1* | *y* |

The `gap` exists so that a five-year forecast cannot see the intervening years.

### 5.4 Temporal split by target year

```
train: target ≤ 2010        val: 2011–2014        test: 2015–2018
```

Splits are defined over **window indices**, never over flattened rows, and the three index sets
are disjoint by construction — `check_dataset.py` verifies this.

### 5.5 Descriptors at each window's own origin

For a window whose input ends at year *t₀*, the three descriptors are `[mean, linear trend,
standard deviation]` computed from `series[:, :t₀+1]` only:

```python
origin_descriptors(series, origins = target_idx - gap - T_out)
```

**This is the single most important line in the pipeline.** Sharing one descriptor set across
all windows was the original leakage channel: a window forecasting 2005 would be handed
statistics computed through 2018. §5.4 of the manuscript shows what that channel is worth
(1.140 °C versus 1.340 °C when the input is available).

### 5.6 Graph construction

A symmetric k-nearest-neighbour graph over pixel coordinates, `k = 8`, self-loops added at
training time. Stored as an edge list `(2, E)`, converted to a dense adjacency only inside the
external baselines that require one.

### 5.7 Normalisation

Targets are standardised with the mean and standard deviation of the **training period only**
(1981–2010), stored as `ts_mean` / `ts_std`. Descriptors are standardised with training-window
statistics. Metrics are always reported after inverse transformation, in °C, so that learned
models and physical baselines are commensurable.

### 5.8 What the builder writes

`build_dataset_v2` writes the `.npz` described in `docs/DATA.md` §3, plus:

* `window_manifest_v2.csv` — one row per window: split, input years, target year, origin;
* `dataset_manifest.json` — hashes and configuration of the build.

Keep the manifest: it is the answer to "which data produced this table".

---

## 6. Code layout

```
filmstgat/
  data/            dataset construction, windowing, synthetic generators
  models/          STGAT backbone, FiLM layers, legacy baselines + adapters
  train/           the single training loop, seeding
  eval/            metrics, overlap measurement
  experiments/     one script per experiment (e1 … e5), no shared state
  utils/           provenance (run manifests: code hash, data hash, config)
scripts/           data build, probes, audits, result bundling, release
figures/           figure generation, reads result files only
tests/             offline test suites (no data, no GPU)
docs/              deployment, data statement, audit log, manuscript sources
run_all.sh         the only entry point you need
```

Two conventions worth knowing:

* **No duplicated implementations.** One model family, one loop, one evaluator. If a concept
  exists twice, that is treated as a bug.
* **Every run writes provenance.** Each `runs/<stage>/<run_id>/` contains `run_manifest.json`
  with the code tree hash, the dataset hash and the full configuration.

---

## 7. How to run

Set the output root once:

```bash
export ROOT=/path/to/your/workdir     # must contain datasets_v2/ and datasets_v2_h1/
```

### 7.1 Shape probe (10 s, always run this first)

```bash
bash run_all.sh probe
```

Runs one forward pass per model, on CPU, and prints the input/output shapes it expects. Three
external baselines have three different calling conventions; this catches a mismatch before
any training starts.

### 7.2 Experiment stages

| Stage | What it runs | Typical cost | Writes to |
|---|---|---|---|
| `e1`, `e1h1` | conditioning ablations, 5 configurations × 5 seeds | ~10 min | `runs/e1`, `runs/e1h1` |
| `e3`, `e3h1` | naive baselines (persistence, trend) | seconds | `runs/e3` |
| `e5`, `e5h1` | three external backbones, 5 seeds | ~10 min | `runs/e5` |
| `e2` | five-level descriptor-overlap sweep | ~20 min | `runs/e2` |
| `e2b` | two-endpoint contrast, 24 seeds | ~15 min | `runs/e2b` |
| `stats`, `e4agg` | per-seed tests, pooled MDE | seconds | `runs/e4` |
| `signal` | zero-modelling floors, lag profile | seconds | `datasets_v2/signal_diagnosis.json` |
| `synth` | synthetic benchmark with known heterogeneity | ~10 min | `runs/e1synth` |

All stages write logs to `$ROOT/logs/`. `run_all.sh` launches them with `nohup` by default so
a dropped SSH session does not kill a run:

```bash
bash run_all.sh e1                       # background (default)
BG=0 bash run_all.sh e1                  # foreground, see output live
tail -f $ROOT/logs/e1_*.log              # watch progress
```

### 7.3 Determinism

Every run seeds all random sources **before** the model is constructed. Same seed, same GPU
model and same driver give identical results; across GPU models expect small differences from
non-deterministic cuDNN kernels. All conclusions in the manuscript are stated as mean ± s.d.
over five seeds and as intervals, never as a single run.

---

## 8. How to inspect results

### 8.1 Layout

```
$ROOT/runs/<stage>/<config>_seed<NN>/
    metrics.json          metrics in °C + units field + best epoch
    predictions.npz       pred, true — shape (test_windows, n_pixels), physical units
    run_manifest.json     code hash, data hash, config, seeds
```

### 8.2 Reading a single run

```bash
python - <<'PY'
import json, numpy as np
d = "runs/e1/film_full_seed42"
m = json.load(open(f"{d}/metrics.json"))
print(m["metrics"], m["units"], "best epoch", m["best_epoch"])
z = np.load(f"{d}/predictions.npz")
print(z["pred"].shape, z["true"].shape)
PY
```

Always check the `units` field before comparing anything: a normalised RMSE of 0.9 looks
perfectly plausible and is meaningless.

### 8.3 Stage-level summaries

| File | Produced by | Contents |
|---|---|---|
| `runs/e1/summary_e1.json` | `e1` | mean ± s.d. per configuration |
| `runs/e4/aggregate_*.json` | `e4agg` | per-seed deltas, pooled mean, MDE |
| `datasets_v2/signal_diagnosis.json` | `signal` | floors, lag profile, variance split |
| `runs/e2b/analysis/` | `analyze` | paired tests, difference-in-differences |
| `results/MANIFEST.json` | `bundle` | SHA-256 of every shipped result file |

`docs/RESULTS_MAP.md` maps each manuscript table and figure to the command that produces it
and the file it reads.

---

## 9. How to make figures and tables

Both read **result files only** — no number is written into the plotting code, and a missing
input is reported instead of being silently replaced.

```bash
# 9.1 figures
ROOT=$ROOT bash run_all.sh figures
ls $ROOT/figures/*.pdf

# 9.2 tables
ROOT=$ROOT bash run_all.sh tables
ls $ROOT/tables/*.md
```

Manuscript figures produced:

| Figure | Content |
|---|---|
| `F1_floors_and_lag` | zero-modelling floors; RMSE versus forecast lag |
| `F2_collapse_to_persistence` | model output against the persistence forecast, r ≈ 1 |
| `F3_effect_vs_resolution` | per-seed effects with the design's MDE band |
| `F4_overlap_dose_response` | Δ versus descriptor–input overlap α |
| `fig_synthetic_benchmark`, `fig_conditioning_real`, `fig_overlap_contrast`, `fig_split_timeline` | supporting panels |

**Without a GPU and without the raw data**, rebuild everything from the shipped snapshot:

```bash
ROOT=results bash run_all.sh figures
ROOT=results bash run_all.sh tables
```

Individual figures: `python figures/make_figures.py --root $ROOT --only floors`.

---

## 10. Reproducibility audit

```bash
python scripts/repro_audit.py
```

Six checks, no side effects:

| Section | Checks |
|---|---|
| A structure | required files; every script referenced by `run_all.sh` and by the docs exists |
| B syntax | every `.py` compiles (the most common reason code "does not run" elsewhere) |
| C portability | no machine-specific absolute paths in code |
| D secrets | no passwords, tokens or credentialed URLs |
| E data | no raw observations shipped; `results/` present with a manifest |
| F integrity | `tree_hash` in `VERSION.json` matches the files |

`FAIL` blocks a release; `WARN` needs a human decision. Before uploading, use
`--write-version` to refresh the hash, then run once more to confirm it matches.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `missing inputs (skipped, NOT fabricated)` for everything | `ROOT` points at the wrong directory or another machine | check the `host` / `root: N entries, M npz` lines printed by every stage |
| background log stays empty for a long time | Python block-buffers stdout to a file | already handled by `PYTHONUNBUFFERED=1` in `run_all.sh`; add `python -u` if running by hand |
| `not enough values to unpack (expected 4, got 3)` | model expects `(B,N,T,F)`, received `(N,T,F)` | `bash run_all.sh probe` |
| `target size (8,500) != input size (8,500,1)` | a model returned a trailing singleton dim | the trainer now raises with the model name; squeeze the output |
| `'yerr' must not contain negative values` | a bootstrap interval used as an error bar | plot the estimate and the interval separately |
| FiLM appears to do nothing at initialisation | identity init (`gamma=1, beta=0`) makes the derivative zero at step 0 | not a bug; verify learnability with `scripts/probe_film_learnability.py` |
| `CUDA out of memory` | another process on the GPU | `nvidia-smi`; the experiments need < 2 GB |

---

## 12. Packaging and uploading a release

```bash
# 12.1 bundle result files (no weights, no raw data)
python scripts/collect_results.py --runs $ROOT --out results

# 12.2 refresh and verify the integrity hash
python scripts/repro_audit.py --write-version
python scripts/repro_audit.py          # must print FAIL=0

# 12.3 build the archive
cd ..
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
    -czf filmstgat-release.tar.gz filmstgat-v2.8.4

# 12.4 publish on a branch (main stays untouched)
cd filmstgat-v2.8.4
git init -b repro-v3
git add -A
git status --short | head          # verify no datasets/ or runs/ staged
git commit -m "Reproducible release: leakage-free protocol, competition probe, audit tooling"
git remote add origin <your-fork-url>
git push -u origin repro-v3
```

`.gitignore` already excludes `datasets*/`, `runs/`, `logs/`, weights and caches, while keeping
`results/`.

---

## 13. License and citation

Code is released under the MIT License (`LICENSE`). Citation metadata is in `CITATION.cff`;
please cite both the paper and the archived release. The observations themselves are **not**
redistributed — see `docs/DATA.md` for their provenance and terms.
