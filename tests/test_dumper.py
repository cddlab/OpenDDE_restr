# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from runner.dumper import DataDumper


def test_is_complete_requires_every_structure_and_confidence(tmp_path):
    dumper = DataDumper(str(tmp_path))
    prediction_dir = tmp_path / "job" / "seed_7" / "predictions"
    prediction_dir.mkdir(parents=True)

    for rank in range(2):
        (prediction_dir / f"job_sample_{rank}.cif").touch()
        (prediction_dir / f"job_summary_confidence_sample_{rank}.json").touch()

    assert dumper.is_complete("", "job", 7, 2)

    (prediction_dir / "job_summary_confidence_sample_1.json").unlink()
    assert not dumper.is_complete("", "job", 7, 2)
