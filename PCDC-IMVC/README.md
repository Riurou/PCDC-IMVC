# PCDC-IMVC

PCDC-IMVC is a cleaned and industrialized project layout for incomplete multi-view clustering research and engineering delivery.

## Project Layout

```text
PCDC-IMVC/
  src/pcdc_imvc/
    cli/               # train entrypoints
    configs/           # default config dictionaries
    models/            # model definitions and losses
    utils/             # data loading, mask, logging, metrics
  configs/             # runtime config templates (yaml/json can be added)
  data/                # dataset root (place .mat files here)
  docs/                # migration and engineering documents
  outputs/
    logs/              # runtime logs
    checkpoints/       # saved model weights
  scripts/             # launcher scripts
  tests/               # unit or smoke tests
  tools/               # optional helper tools
```

## Environment

This project is validated in the existing conda runtime used for development.

Recommended setup:

- Python 3.8 or newer
- PyTorch 2.4.1+cu121
- NumPy, SciPy, scikit-learn, h5py, matplotlib, and munkres
- CUDA-enabled GPU is recommended for full training runs, but smoke tests can run on CPU

If you are using another machine, install the package inside a compatible Python environment and make sure PyTorch can access the expected CUDA runtime if you plan to train on GPU.

## Quick Start

1. Install package in editable mode:

```bash
pip install -e .
```

2. Put dataset files in `data/`.

Or link all datasets covered by configure from existing A/data:

```bash
./tools/link_config_datasets.sh /data/2025_stu/lr/A/data
```

3. Run training:

```bash
pcdc-train --mode two --dataset 6 --missing_rate 0.5
pcdc-train --mode multi --dataset 11 --missing_rate 0.5
# or use dataset name
pcdc-train --dataset_name ALOI_100 --missing_rate 0.5
```

## Smoke Test

Fast acceptance check without full training:

```bash
pcdc-train --mode two --dataset_name Caltech101-20 --dry_run
pcdc-train --mode multi --dataset_name Mfeat --dry_run
```

Automated smoke suite:

```bash
python -m pytest -q tests/smoke_test.py
```

## Scope of Migration

This migration includes only the complete model code path:

- model definitions
- loss definitions
- data loaders
- default configs
- two-view and multi-view training CLIs

Experimental scripts, generated figures, and historical logs are intentionally excluded.

See `docs/MIGRATION_GUIDE.md` for exact file mapping.