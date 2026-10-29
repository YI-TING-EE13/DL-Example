"""Verify content provenance and run records on failure and interruption."""

import json

import pytest

from dl_example.cli import train
from dl_example.config import ExperimentConfig
from dl_example.data import dataset_file_hashes


def test_dataset_digest_detects_changed_content(tmp_path):
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"original")
    before = dataset_file_hashes([path])
    path.write_bytes(b"modified")
    assert dataset_file_hashes([path]) != before


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt, "interrupted"), (OSError, "failed")])
def test_run_records_preserve_unsuccessful_status(tmp_path, monkeypatch, error, status):
    def interrupted_loader(*args):
        raise error("Stopped during dataset preparation")

    monkeypatch.setattr("dl_example.cli.prepare_data", interrupted_loader)
    with pytest.raises(error):
        train(ExperimentConfig(device="cpu", output_dir=str(tmp_path)))
    record_path = next(tmp_path.glob("*/run.json"))
    assert json.loads(record_path.read_text())["status"] == status
