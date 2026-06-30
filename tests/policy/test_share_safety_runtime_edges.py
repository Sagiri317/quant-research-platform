from __future__ import annotations

import pytest

from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_share_safe_text,
)
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    sanitize_shared_object,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_sanitizer_redacts_windows_unc_and_posix_paths() -> None:
    # Intentional fake provider/source marker for share-safety rejection.
    text = r"C:\Users\DELL\secret.csv \\server\share\raw.xlsx /mnt/data/private.csv /home/user/private.csv /Users/name/private.csv /tmp/x"  # noqa: E501

    sanitized = sanitize_shared_text(text)

    assert "C:" not in sanitized
    assert "\\\\server" not in sanitized
    assert "/mnt/" not in sanitized
    assert "/home/" not in sanitized
    assert "/Users/" not in sanitized
    assert "/tmp/" not in sanitized


def test_sanitizer_redacts_absolute_private_repo_path_as_one_token() -> None:
    text = "/workspace/qrp/data/research/a_share/demo/_manifest/manifest.json"

    sanitized = sanitize_shared_text(text)

    assert sanitized == "[local-only-path]"
    assert "/workspace/qrp/[local-only-path]" not in sanitized


@pytest.mark.parametrize(
    "text",
    [
        # Intentional fake provider/source marker for safety rejection tests.
        "/workspace/qrp/data/research/a_share/demo",
        # intentional legacy-path negative case
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/opt/qrp/data/processed/a_share/factor/demo",
        r"C:\repo\data\research\a_share\demo",
        # intentional legacy-path negative case
        r"D:\repo\reports\private_research\a_share\run",
    ],
)
def test_sanitizer_redacts_repo_private_absolute_paths(text: str) -> None:
    sanitized = sanitize_shared_text(text)

    assert "data/research" not in sanitized
    assert "data/processed" not in sanitized
    assert "reports/private_research" not in sanitized
    assert "workspace" not in sanitized.lower()
    assert "C:" not in sanitized
    assert "D:" not in sanitized


@pytest.mark.parametrize(
    "text",
    [
        # intentional legacy-path negative case
        "reports/private_research/a_share/run",
        # intentional legacy-path negative case
        "/workspace/qrp/reports/private_research/a_share/run",
    ],
)
def test_sanitizer_rejects_local_output_paths(text: str) -> None:
    # intentional legacy-path negative case
    assert text.endswith("reports/private_research/a_share/run")
    assert unsafe_for_share_reason(text) == "local_only_path"


def test_sanitizer_redacts_shared_markers_stock_codes_and_credentials() -> None:
    # Intentional fake provider/source marker for safety rejection tests.
    text = (
        "data/raw/csmar/export.csv stock code 600000.SH CSMAR vendor token=abcdef123456"  # noqa: E501
    )

    sanitized = sanitize_shared_text(text)

    assert "data/raw/csmar" not in sanitized
    assert "600000" not in sanitized
    assert "CSMAR" not in sanitized
    assert "abcdef123456" not in sanitized


def test_stock_code_detector_does_not_flag_hash_payloads() -> None:
    text = '{"sha256": "1ab57ebee6473a9012cb336cc004040e2130408ff5a08bcf"}'

    assert unsafe_for_share_reason(text) is None
    assert sanitize_shared_text(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "830000.BJ",
        "430047.BJ",
        "600000.SH",
    ],
)
def test_stock_code_detector_covers_a_share_exchange_suffixes(text: str) -> None:
    assert unsafe_for_share_reason(text) == "stock_code_sample"
    assert text not in sanitize_shared_text(text)


def test_sanitizer_recurses_public_objects() -> None:
    # Intentional fake provider/source marker for safety rejection tests.
    payload = {
        "path": r"D:\quant-workspace\data\processed\a_share",
        "rows": ["000001.SZ"],
        "nested": {"secret": "password=abcdef123456"},
    }

    sanitized = sanitize_shared_object(payload)

    assert "D:" not in str(sanitized)
    assert "000001" not in str(sanitized)
    assert "abcdef123456" not in str(sanitized)


def test_provider_source_marker_input_is_rejected_before_row_level_read() -> None:
    row_reader_called = False

    def guarded_review_safe_output(input_text: str) -> None:
        nonlocal row_reader_called
        assert_share_safe_text(input_text)
        row_reader_called = True

    with pytest.raises(ProposalSafetyError):
        guarded_review_safe_output(
            r"data\processed\a_share row sample: market_id=asset"
        )

    assert row_reader_called is False


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("row sample: market_id=a_share asset_id=000001", "row_sample"),
        ("source marker from CSMAR export table", "provider_source_marker"),
        ("vendor_marker=raw-provider", "provider_source_marker"),
        ("provider_name=raw-provider", "provider_source_marker"),
        ("source_name=raw-source", "provider_source_marker"),
        ("data_provider=raw-provider", "provider_source_marker"),
        ("vendor_name=raw-provider", "provider_source_marker"),
    ],
)
def test_unsafe_for_share_reason_covers_share_safety_markers(
    text: str,
    reason: str,
) -> None:
    assert unsafe_for_share_reason(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "source_id=csmar",
        "provider_id=choice",
        '{"source_id": "csmar"}',
        '{"provider_id": "choice"}',
    ],
)
def test_provider_source_fields_are_fully_sanitized_and_rejected(text: str) -> None:
    assert unsafe_for_share_reason(text) == "provider_source_marker"
    sanitized = sanitize_shared_text(text)
    assert "csmar" not in sanitized.lower()
    assert "choice" not in sanitized.lower()


def test_row_sample_payload_is_fully_sanitized_and_rejected() -> None:
    text = 'row_samples=[{"asset_id":"000001","value":1.2}]'

    assert unsafe_for_share_reason(text) == "row_sample"
    sanitized = sanitize_shared_text(text)
    assert "000001" not in sanitized
    assert "1.2" not in sanitized


def test_embedded_stock_code_run_token_is_rejected_but_date_id_is_allowed() -> None:
    assert unsafe_for_share_reason("run_000001") == "stock_code_sample"
    assert unsafe_for_share_reason("run_202606") is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "daily__market_bars__export-20240101.csv",
            "raw_export_filename",
        ),
        ("source_asset_id", "source_local_key"),
        ("source_market_id", "source_local_key"),
        ("runtime_payload=private-aggregate", "runtime_payload"),
        ("exact_runtime_payload", "runtime_payload"),
        (".run/private/s1/summary.json", "local_run_path"),
    ],
)
def test_share_safety_owns_report_and_handoff_text_markers(
    text: str,
    reason: str,
) -> None:
    assert unsafe_for_share_reason(text) == reason
    assert text not in sanitize_shared_text(text)


def test_logical_local_only_path_allowance_masks_only_the_path_value() -> None:
    logical_path = "data/interim/csmar/example_dataset/_manifest/manifest.json"

    assert (
        unsafe_for_share_reason(
            logical_path,
            allow_logical_local_only_path=True,
        )
        is None
    )
    assert (
        unsafe_for_share_reason(
            f"/workspace/qrp/{logical_path}",
            allow_logical_local_only_path=True,
        )
        == "local_only_path"
    )
    assert (
        unsafe_for_share_reason(
            f"source_id=csmar path={logical_path}",
            allow_logical_local_only_path=True,
        )
        == "provider_source_marker"
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"path": "data/research/a_share/demo"}, "local_only_path"),
        ({"path": "/workspace/qrp/data/research/a_share/demo"}, "local_only_path"),
        ({"row_samples": [{"value": 1}]}, "row_sample"),
        ({"credential": "token=abcdef123456"}, "credential"),
    ],
)
def test_internal_artifact_metadata_safety_is_owned_by_runtime(
    payload: dict[str, object],
    reason: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"unsafe_internal_artifact_metadata: {reason}",
    ):
        assert_internal_artifact_metadata_safe(payload)
