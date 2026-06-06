import subprocess
import sys
from pathlib import Path

from pcdc_imvc.configs.configure import (
    SUPPORTED_MULTI_VIEW_DATASETS,
    SUPPORTED_TWO_VIEW_DATASETS,
    get_default_config,
    get_required_data_files,
    resolve_dataset_input,
)


def _run(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_config_single_source():
    for name in sorted(SUPPORTED_TWO_VIEW_DATASETS):
        cfg = get_default_config(name, mode="two")
        assert int(cfg["view"]) == 2
        assert "training" in cfg and "Autoencoder" in cfg and "Inference" in cfg

    for name in sorted(SUPPORTED_MULTI_VIEW_DATASETS):
        cfg = get_default_config(name, mode="multi")
        assert int(cfg["view"]) >= 2
        assert "training" in cfg and "Autoencoder" in cfg and "Inference" in cfg


def test_unified_cli_help():
    code, out, err = _run([sys.executable, "-m", "pcdc_imvc.cli.train", "--help"])
    assert code == 0, err
    assert "--mode" in out
    assert "--dataset_name" in out


def test_dataset_id_registry_resolution():
    name, mode = resolve_dataset_input(dataset=0, dataset_name=None, mode="two")
    assert name == "Caltech101-20"
    assert mode == "two"

    name, mode = resolve_dataset_input(dataset=11, dataset_name=None, mode="multi")
    assert name == "ALOI_100"
    assert mode == "multi"

    name, mode = resolve_dataset_input(dataset=None, dataset_name="Mfeat", mode="auto")
    assert name == "Mfeat"
    assert mode == "multi"


def test_unified_cli_dry_run_two_and_multi():
    code1, out1, err1 = _run(
        [
            sys.executable,
            "-m",
            "pcdc_imvc.cli.train",
            "--mode",
            "two",
            "--dataset_name",
            "Caltech101-20",
            "--dry_run",
        ]
    )
    assert code1 == 0, err1
    assert "Dry run finished" in out1 or "Dry run finished" in err1

    code2, out2, err2 = _run(
        [
            sys.executable,
            "-m",
            "pcdc_imvc.cli.train",
            "--mode",
            "multi",
            "--dataset_name",
            "Mfeat",
            "--dry_run",
        ]
    )
    assert code2 == 0, err2
    assert "Dry run finished" in out2 or "Dry run finished" in err2


def test_data_files_presence_for_config_datasets():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    missing = [fname for fname in get_required_data_files() if not (data_dir / fname).exists()]
    assert not missing, f"Missing dataset files in data/: {missing}"