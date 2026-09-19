# Geo-FiLM ST-GAT: Overlap-Aware FiLM Conditioning for Spatio-Temporal Graph Networks

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4%2Bcu124-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Official code for the manuscript **"Overlap-Aware FiLM Conditioning for Spatio-Temporal Graph Networks with Overlapping Geological Priors: A Permafrost Thermal Prediction Study"** (PeerJ Computer Science, under review).

---

## Table of Contents

- [Description](#description)
- [Dataset Information](#dataset-information)
- [Code Information](#code-information)
- [Requirements](#requirements)
- [Usage Instructions](#usage-instructions)
- [Methodology](#methodology)
- [Reproducing the Paper](#reproducing-the-paper)
- [Citations](#citations)
- [License & Contribution Guidelines](#license--contribution-guidelines)

---

## Description

Permafrost thermal prediction is a problem in which the *target field* (e.g. mean annual ground temperature, MAGT) and the available *geological priors* (e.g. active-layer thickness, ALT; soil thermal conductivity, `k_s`) often exhibit **different spatial gradients**. Existing approaches inject such priors into neural networks with a single, fixed conditioning form — usually additive or multiplicative Feature-wise Linear Modulation (FiLM) — and apply it uniformly regardless of how well the two fields spatially overlap.

This repository implements an **overlap-degree selection framework** for conditioning spatio-temporal graph networks with geological priors. Instead of committing to a single FiLM form, the framework shows that the *degree of spatial overlap* ρ between the geological and target fields determines which conditioning form is most effective:

- **High-overlap regimes (ρ > 0.5)** — Additive geo-bias `h + σ(α)·β` is sufficient. The two fields share similar spatial gradients, so a gentle shift preserves the learned representation.
- **Low-overlap regimes (ρ < 0.3)** — Full FiLM `γ·h + β` is required. The geological prior must actively *transform* the feature space because the spatial gradients differ fundamentally.

The core model, **Geo-FiLM ST-GAT**, embeds this principle into a spatio-temporal graph attention network with **zero-initialized FiLM generators**, guaranteeing that conditioning degrades gracefully to the unconditioned baseline.

**Headline result.** In a fair, parameter-matched, 5-seed ablation on real MAGT data, Full FiLM achieves **1.594 ± 0.034 °C RMSE (R² = 0.9693)** — a **38.8 % RMSE reduction** over the parameter-matched no-FiLM baseline — confirming that the benefit of FiLM is genuine and not an artifact of a larger parameter count.

**Repository contents.** The Geo-FiLM ST-GAT model; six comparison models (ST-GAT, ST-GCN, DCRNN, PINN-Permafrost, Stefan-only, plus Random Forest and XGBoost baselines); automated data acquisition from TPDC / PANGAEA / NCDC; a physics-based synthetic data generator; and reproduction scripts for every figure and table in the paper.

---

## Dataset Information

This study uses publicly available permafrost and geodetic datasets. All raw data are **not redistributed** in this repository; instead, download scripts retrieve them from their original archives. Sentinel values differ per dataset (see [Methodology](#methodology)).

### Primary datasets used for the main experiments

| ID | Dataset | Provider | DOI / Access | Format | Role in the paper |
|:---|:---|:---|:---|:---|:---|
| **D1** | Qinghai–Tibet Engineering Corridor 2D Surface Deformation (2017-02 – 2022-03) | TPDC | `10.11888/Cryos.tpdc.300400` (Du et al., 2023) | GeoTIFF | InSAR deformation field, spatial structure |
| **D2** | Qinghai–Tibet Plateau Mean Annual Ground Temperature (MAGT), 1 km annual simulation (1981–2018) | TPDC | `10.11888/Cryos.tpdc.300603` (Zhao et al., 2023) | GeoTIFF | **Primary target field** for thermal prediction |
| **D2b** | GTN-P CALM Active Layer Thickness (ALT) | PANGAEA | `10.1594/PANGAEA.972777` | CSV | Geological prior constraint (`y ≤ k_s · ALT`) |

### Additional candidate datasets (multi-region generalization)

| ID | Dataset | Provider | Access | Config |
|:---|:---|:---|:---|:---|
| **A1** | Chaimu Railway Deformation + Damage, Qilian Mountains | TPDC | `f7be4a1d-2506` | `configs/chaimu_railway.yaml` |
| **A4** | Qaidam Basin Corridor InSAR | TPDC | `10.11888/cryos.tpdc.300400` | `configs/qaidam_inSAR.yaml` |
| **A5** | Northeast China Permafrost Engineering Stability | NCDC | `70c33bc7` | `configs/northeast_frost.yaml` |
| **A8** | Yeniugou Permafrost Deformation + ALT | TPDC | `dd24266c` | `configs/yeniugou_alt.yaml` |

### Synthetic dataset (fallback, always available)

A **physics-based synthetic generator** (`data/synthetic_generator.py`) produces physically plausible permafrost thermal fields from the Stefan equation and a thermodynamic model, with a known ground truth. This path requires **no download** and is used for framework validation, debugging, and CI-style reproducibility checks.

### How to obtain the data

> **Access note.** TPDC (National Tibetan Plateau / Third Pole Environment Data Center) and NCDC require free registration. After registering and requesting a dataset, the archive issues **temporary download credentials**. These credentials are personal and expire — they are **not** stored in this repository. Place your own credentials in `data/ftp_download_tpdc.py` locally, or download files manually through the web portal.

**Option A — Automated download (recommended):**

```bash
# Download all registered datasets and preprocess into datasets/processed/*.npz
python download_datasets.py

# Only preprocess data that is already in datasets/raw/
python download_datasets.py --preprocess-only

# Check download status without downloading
python download_datasets.py --status
```

**Option B — Direct TPDC FTP download (large archives):**

```bash
python data/ftp_download_tpdc.py --dataset all      # both InSAR + MAGT
python data/ftp_download_tpdc.py --dataset magt     # MAGT only (primary target)
python data/ftp_download_tpdc.py --dataset insar    # InSAR deformation only
```

**Option C — Manual download:** visit the DOI/URL above, download the archive, and place it under `<DATA_ROOT>/datasets/raw/`. The pipeline is aware of `DATA_ROOT` (see [Usage Instructions](#usage-instructions)).

### Data preprocessing summary

- Raster masking per dataset (e.g. `−50 < value < 50` for MAGT; `np.isfinite()` for InSAR velocity, whose NoData marker is unreliable).
- Reprojection of all layers onto a common 1 km grid and extraction of overlapping pixels between the geological prior and the target field.
- Standardization using statistics computed **only on the training split** (no leakage).
- Conversion to spatio-temporal graph tensors: node features `X ∈ R^{N × T × F}`, geological attributes `G ∈ R^{N × 3}`, and a spatial adjacency matrix `A` built with a 50 km threshold.
- Serialization to compressed `.npz` under `datasets/processed/`.

---

## Code Information

### Repository structure

```
overlap-aware-film-permafrost/
├── run_main.py                     # Main entry point — retrospective prediction experiment
├── download_datasets.py            # Standalone download + preprocessing driver
├── setup_autodl.sh                 # One-shot AutoDL cloud-GPU environment setup
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT (code)
│
├── models/                         # Neural network implementations
│   ├── geo_film_st_gat.py          # ★ Core: Overlap-Aware Geo-FiLM ST-GAT
│   ├── st_gat.py                   #   Standard ST-GAT (ablation baseline)
│   ├── stgcn.py                    #   ST-GCN baseline (Chebyshev)
│   ├── dcrnn.py                    #   DCRNN baseline (diffusion convolution)
│   ├── pinn_permafrost.py          #   PINN-Permafrost baseline
│   ├── stefan_model.py             #   Pure Stefan-equation physics baseline
│   └── _utils.py                   #   Shared model utilities
│
├── trainers/
│   └── trainer.py                  # Training loop, losses, early stopping
│
├── data/
│   ├── download_data.py            # Download with graceful synthetic fallback
│   ├── ftp_download_tpdc.py        # TPDC FTP client (bring your own credentials)
│   ├── preprocess.py               # Feature engineering + graph construction
│   ├── preprocess_real_magt.py     # Real-MAGT-specific preprocessing
│   └── synthetic_generator.py      # Stefan-equation physics-based synthetic data
│
├── evaluation/
│   ├── metrics.py                  # RMSE, MAE, R², ...
│   └── physical_constraint.py      # Stefan constraint violation rate
│
├── experiments/                    # Standalone experiment scripts
│   ├── run_fair_film_experiment.py # ★ Fair, parameter-matched FiLM ablation
│   ├── run_fair_mult_film.py       # Full vs. multiplicative FiLM comparison
│   ├── run_real_magt_experiment.py # ★ Real MAGT retrospective prediction
│   ├── run_multi_seed.py           # Multi-seed statistical validation
│   ├── run_no_geo_input.py         # No-geological-prior ablation
│   ├── validate_alt_magt.py        # ALT–MAGT correlation-shift analysis
│   ├── visualize_geo_bias.py       # Geo-bias visualization
│   └── synthetic_generator_v2.py   # Extended synthetic generator
│
├── baselines/
│   ├── gbrt_baseline.py            # XGBoost gradient-boosted trees
│   └── rf_baseline.py              # Random Forest
│
├── figures/                        # Figure generation scripts + outputs
│   ├── generate_paper_figures.py       # ★ Real-data figures (Fig. 7–10)
│   ├── generate_synthetic_figures.py   #   Synthetic-data figures (Fig. 1–6)
│   └── *.png
│
└── configs/                        # YAML experiment configurations
    ├── simulation_fallback.yaml    # Synthetic data (no download needed)
    ├── autodl_gpu.yaml             # GPU training on real MAGT data
    ├── qaidam_inSAR.yaml           # Qaidam Basin InSAR
    ├── chaimu_railway.yaml         # Chaimu Railway corridor
    ├── northeast_frost.yaml        # Northeast China permafrost
    └── yeniugou_alt.yaml           # Yeniugou ALT validation
```

### Key modules

| Module | Responsibility |
|:---|:---|
| `models/geo_film_st_gat.py` | The proposed model: GAT spatial encoder, GRU temporal encoder, and zero-initialized **FiLM generator** mapping `[ALT, k_s, I_t] → (γ, β)`. |
| `trainers/trainer.py` | End-to-end training: data batching, physics-regularized loss, validation, early stopping, checkpointing. |
| `evaluation/physical_constraint.py` | Computes the Stefan constraint violation rate — a physically interpretable, uncertainty-aware conditional metric. |
| `experiments/run_fair_film_experiment.py` | The central ablation; enforces parameter parity across FiLM variants via hidden-width matching. |
| `figures/generate_*_figures.py` | Reproduces all manuscript figures (300 dpi, Morandi palette). |

### Design decisions worth noting

1. **Dynamic `input_dim`.** Never hard-coded — read from `X_train.shape[2]` at runtime, and `geo_dim` from `geo_features.shape[1]`. This makes the pipeline robust to datasets with different feature counts.
2. **Zero-initialized FiLM generators.** FiLM starts as the identity map, so the unconditioned baseline is always recoverable and training cannot be destabilized by an aggressive initial conditioning signal.
3. **Soft-clamp on the FiLM bias.** `β = β_raw / (1 + |β_raw| / β_max)` — bounded but never saturating, avoiding the vanishing gradients of a `tanh` clamp.
4. **No external GNN libraries.** All graph convolutions (GAT, GCN, DCRNN) are self-contained; there is no PyTorch Geometric dependency.
5. **Data-disk awareness.** `DATA_ROOT` is resolved as `--data-root` > `$DATA_ROOT` > `/root/autodl-tmp` (if writable) > project root, so large datasets never fill a small system disk.

---

## Requirements

### Software

| Component | Minimum | Tested |
|:---|:---|:---|
| Python | 3.10+ | 3.12.4 |
| PyTorch | 2.4.0 | 2.4+cu124 |
| NumPy | 1.24.0 | 2.x |
| pandas | 2.0.0 | 2.2 |
| scikit-learn | 1.3.0 | 1.5 |
| SciPy | 1.11.0 | 1.14 |
| Matplotlib | 3.7.0 | 3.9 |
| PyYAML | 6.0 | 6.0 |
| XGBoost | 1.7.0 | 2.x |
| rasterio / xarray | *(real data only)* | latest |

Install everything at once:

```bash
pip install -r requirements.txt
```

> On managed GPU platforms (e.g. AutoDL) PyTorch + CUDA are usually pre-installed — skip reinstalling `torch`. If you do need PyTorch, install the CUDA build explicitly:
> ```bash
> pip install "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu124
> ```

### Hardware

| Configuration | Minimum | Recommended |
|:---|:---|:---|
| GPU | NVIDIA GTX 1080 (8 GB) | NVIDIA RTX 4090 (24 GB) |
| RAM | 16 GB | 32 GB+ |
| Storage | 10 GB free | 50 GB+ (real datasets) |
| CUDA | 12.0+ | 13.0 driver / 12.4 runtime |

CPU-only execution is supported for the synthetic path (set `device: "cpu"` in the config); GPU is required for the full real-data experiments within a practical time budget.

---

## Usage Instructions

### Step 0 — Get the code and install dependencies

```bash
git clone <repository-url>
cd overlap-aware-film-permafrost
pip install -r requirements.txt
```

*(Optional, cloud GPU)* — one-shot AutoDL environment setup:

```bash
bash setup_autodl.sh
```

### Step 1 — Run the synthetic experiment (no data download required)

This is the fastest way to verify that the installation is correct:

```bash
python run_main.py --config configs/simulation_fallback.yaml
```

### Step 2 — Run the real-data experiment

After obtaining the TPDC datasets (see [Dataset Information](#dataset-information)):

```bash
python run_main.py --config configs/autodl_gpu.yaml

# If your data disk is not the default location:
python run_main.py --config configs/autodl_gpu.yaml --data-root /path/to/data
```

### Step 3 — Run the key paper experiments

```bash
# (a) Fair, parameter-matched FiLM ablation — the central experiment
python experiments/run_fair_film_experiment.py --n_seeds 5

# (b) Real MAGT retrospective prediction
python experiments/run_real_magt_experiment.py --n_seeds 5

# (c) ALT–MAGT correlation-shift analysis
python experiments/validate_alt_magt.py

# (d) Multi-seed statistical validation
python experiments/run_multi_seed.py --seeds 5 --config configs/autodl_gpu.yaml

# (e) No-geological-prior ablation
python experiments/run_no_geo_input.py
```

### Step 4 — Generate the figures

```bash
# Real-data figures (Fig. 7–10)
python figures/generate_paper_figures.py

# Synthetic-data figures (Fig. 1–6)
python figures/generate_synthetic_figures.py
```

> **Note.** The plotting scripts contain absolute output paths near the top of each file (e.g. `outdir = '.../outputs/figures'`). Edit these two lines to match your local directory before running.

### Step 5 — Check the results

Training prints one line per epoch:

```
Epoch 45/200 | Train Loss: 0.0312 | Val RMSE: 1.716°C | Val R²: 0.9644 | PhysViol: 99.0%
```

Outputs are written under `<DATA_ROOT>/outputs/`:

```
outputs/<experiment_name>/
├── model_best.pt        # Best checkpoint (lowest validation RMSE)
├── training_log.csv     # Per-epoch loss / RMSE / R² history
├── predictions.npz      # Predictions vs. ground truth
├── config_used.yaml     # Snapshot of the configuration used
└── summary.json         # Aggregated multi-seed mean ± std
```

### Command-line reference (`run_main.py`)

| Flag | Description |
|:---|:---|
| `--config PATH` | YAML configuration (default `configs/simulation_fallback.yaml`) |
| `--data-root PATH` | Override `DATA_ROOT` for datasets and outputs |
| `--epochs N` | Override the number of training epochs |
| `--device {cuda,cpu}` | Force execution device |
| `--skip-download` | Use already-cached raw data |
| `--skip-train` | Only generate data and run dimension checks |
| `--verbose` / `-v` | Level 1: progress messages (default) |
| `--debug` / `-d` | Level 2: tensor shapes and model internals |
| `--quiet` / `-q` | Level 0: final results only |
| `--log-file [PATH]` | Tee all output to a log file |
| `--no-log` | Disable logging |

### Tip for long-running jobs

If your SSH session may drop (e.g. a laptop going to sleep), run experiments in the background so they survive disconnection:

```bash
tmux new -s exp
python experiments/run_fair_film_experiment.py --n_seeds 5
# detach with Ctrl+B then D; reattach with: tmux attach -t exp

# or, without tmux:
nohup python run_main.py --config configs/autodl_gpu.yaml > train.log 2>&1 &
```

---

## Methodology

This section summarizes the processing and modeling steps implemented in the code; see the manuscript for full details and equations.

### 1. Data processing

1. **Masking & quality control.** Each raster is masked with dataset-specific sentinel rules (MAGT: `−50 < value < 50`; InSAR: finite values only, because the declared NoData marker is unreliable).
2. **Co-registration.** All layers are reprojected to a common 1 km grid; only pixels where both the geological prior and the target field are valid are retained (an "overlap" mask).
3. **Normalization.** Features are standardized with training-split statistics only.
4. **Graph construction.** Nodes are grid cells; the adjacency matrix uses a 50 km distance threshold. Node features and geological attributes are packed into spatio-temporal tensors and saved as `.npz`.

### 2. Overlap-degree quantification

The overlap degree ρ between the geological prior field and the target field is measured from their spatial-gradient agreement. ρ is then used to select the conditioning regime: additive geo-bias for ρ > 0.5, full FiLM for ρ < 0.3.

### 3. Geo-FiLM ST-GAT model

- **Spatial encoder.** Multi-head graph attention (GAT) layers capture spatial dependencies on the sensor/grid graph.
- **Temporal encoder.** A GRU models temporal evolution over the input window `T_in`.
- **FiLM generator.** A small MLP maps the geological attributes `g_i = [ALT_i, k_{s,i}, I_{t,i}]` to modulation parameters `(γ, β)`. Its final layer is **zero-initialized**, so `γ = 1 + γ_raw → 1` and `β = β_raw → 0` at the start of training.
- **Conditioning forms.** Additive (`h + σ(α)·β`), multiplicative (`γ·h`), and full FiLM (`γ·h + β`) are all supported and compared.
- **Soft-clamp.** `β = β_raw / (1 + |β_raw| / β_max)` keeps the bias bounded without saturating the gradient.

### 4. Physics-regularized objective

The training loss combines a data term with a physics term weighted by `λ_phys` (= 0.3):

```
L = L_data + λ_phys · L_phys
```

where `L_phys` penalizes violations of the Stefan relation `y ≤ k_s · ALT`, providing a physically interpretable constraint that is especially valuable in sparse-data regimes.

### 5. Experimental protocol

- **Fair parameter matching.** FiLM variants and the no-FiLM baseline are matched in parameter count by adjusting hidden width, isolating the effect of conditioning *form* from model capacity.
- **Multi-seed evaluation.** Every headline number is reported as mean ± standard deviation over 5 random seeds (42, 123, 456, 789, 1024).
- **Metrics.** RMSE (°C), MAE, R², and the Stefan constraint violation rate.

---

## Reproducing the Paper

| Manuscript item | Command | Output |
|:---|:---|:---|
| Table — Fair FiLM ablation | `python experiments/run_fair_film_experiment.py --n_seeds 5` | `outputs/.../summary.json` |
| Fig. 7 — Real MAGT RMSE/R² | `python figures/generate_paper_figures.py` | `Fig7.RealMAGT.png` |
| Fig. 8 — Per-zone RMSE | *(same script)* | `Fig8.PerZone.png` |
| Fig. 9 — Fair FiLM ablation | *(same script)* | `Fig9.FairFiLM.png` |
| Fig. 10 — ALT–MAGT shift | *(same script)* | `Fig10.AltMagt.png` |
| Fig. 1–6 — Framework & synthetic results | `python figures/generate_synthetic_figures.py` | `fig1_*.png … fig6_*.png` |

---

## Citations

If you use this code or build on these methods, please cite the manuscript:

```bibtex
@article{fan2026overlap,
  title   = {Overlap-Aware {FiLM} Conditioning for Spatio-Temporal Graph Networks
             with Overlapping Geological Priors: A Permafrost Thermal Prediction Study},
  author  = {Fan, Xiaohu},
  journal = {PeerJ Computer Science},
  year    = {2026},
  note    = {Under review}
}
```

Please also cite the original data providers:

```bibtex
@dataset{du2023corridor,
  title     = {Two-dimensional surface deformation dataset of the Qinghai-Tibet
               Engineering Corridor permafrost region (2017-2022)},
  author    = {Du, et al.},
  year      = {2023},
  publisher = {National Tibetan Plateau Data Center},
  doi       = {10.11888/Cryos.tpdc.300400}
}

@dataset{zhao2023magt,
  title     = {1-km annual mean annual ground temperature (MAGT) simulation dataset
               of the Qinghai-Tibet Plateau permafrost region (1981-2018)},
  author    = {Zhao, et al.},
  year      = {2023},
  publisher = {National Tibetan Plateau Data Center},
  doi       = {10.11888/Cryos.tpdc.300603}
}

@dataset{calm_alt,
  title     = {GTN-P CALM Active Layer Thickness},
  year      = {2023},
  publisher = {PANGAEA},
  doi       = {10.1594/PANGAEA.972777}
}
```

---

## License & Contribution Guidelines

### License

This project uses a **split license** appropriate to research software:

- **Source code** (`*.py`, scripts, configuration): **MIT License** — see [LICENSE](LICENSE). Copyright © 2026 Xiaohu Fan.
- **Documentation and derived figures** (`README-EN.md`, `*.md`, generated `*.png`): **Creative Commons Attribution 4.0 International (CC BY 4.0)** — you may share and adapt them with attribution.

The **raw datasets** are governed by their respective providers' terms (TPDC, PANGAEA, NCDC). They are downloaded, not redistributed, by this repository; please observe each provider's license and citation requirements.

### Contribution guidelines

Contributions, bug reports, and questions are welcome:

1. Open an issue describing the bug or proposed change, including a minimal reproduction (command, config, and error trace) where applicable.
2. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
3. Keep changes focused; run `python run_main.py --config configs/simulation_fallback.yaml` to confirm the pipeline still works end-to-end.
4. Submit a pull request with a clear description of the change and its motivation.

Please do not commit datasets, model checkpoints, or download credentials to the repository.

---

*Last updated: September 2026.*
