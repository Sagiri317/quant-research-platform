from __future__ import annotations

import csv
import json

from quant_research_platform.runtime.timing import TimingRecorder


def test_timing_recorder_writes_json_and_csv(tmp_path) -> None:
    recorder = TimingRecorder()

    with recorder.record("build_processed", logical_layer="processed") as record:
        record.output_rows = 3
        record.partitions = 1

    json_path = recorder.write_json(tmp_path / "timings.json")
    csv_path = recorder.write_csv(tmp_path / "timings.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert payload["records"][0]["step_name"] == "build_processed"
    assert payload["records"][0]["duration_seconds"] >= 0
    assert payload["records"][0]["output_rows"] == 3
    assert payload["records"][0]["rows_per_second"] is not None
    assert rows[0]["step_name"] == "build_processed"
    assert rows[0]["status"] == "success"
