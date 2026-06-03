from __future__ import annotations

import pytest

from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler


def test_stage_profile_bundle_writes_only_inside_private_report_boundary(tmp_path):
    with StageProfiler(ProfileConfig(enabled=True)) as profiler:
        with profiler.stage("evaluation", "ic", sample_memory=False):
            pass

    output_dir = (
        tmp_path / "reports" / "private_real" / "a_share" / "diagnostics" / "run"
    )
    paths = profiler.write_artifacts(output_dir)

    assert paths["csv"].is_file()
    assert paths["json"].is_file()
    assert paths["report"].is_file()


def test_stage_profile_bundle_rejects_public_output(tmp_path):
    profiler = StageProfiler(ProfileConfig(enabled=True))

    with pytest.raises(ValueError, match="outside_private_boundary"):
        profiler.write_artifacts(tmp_path / "public")
