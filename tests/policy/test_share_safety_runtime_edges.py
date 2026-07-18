from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_share_safe_text,
)
from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    evidence_json_value_unsafe_reason,
    sanitize_shared_object,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_share_safety_failure_path_allows_logical_output_path() -> None:
    assert (
        evidence_json_value_unsafe_reason(
            {"path": "outputs/x/a.md", "reason": "invalid_utf8"}
        )
        is None
    )


@pytest.mark.parametrize(
    "path",
    (
        r"C:\repo\outputs\x\a.md",
        r"\\server\share\outputs\x\a.md",
        "/workspace/qrp/outputs/x/a.md",
    ),
)
def test_share_safety_failure_path_still_rejects_absolute_paths(path: str) -> None:
    assert (
        evidence_json_value_unsafe_reason({"path": path, "reason": "invalid_utf8"})
        == "absolute_path"
    )


@pytest.mark.parametrize(
    "path",
    (
        r"outputs/x/C:\private\a.md",
        r"outputs/x/\\server\share\a.md",
        "outputs/x//workspace/qrp/a.md",
        "outputs/x/anaconda3/envs/private/a.md",
    ),
)
def test_logical_path_prefix_cannot_mask_embedded_absolute_token(path: str) -> None:
    assert (
        evidence_json_value_unsafe_reason({"path": path, "reason": "invalid_utf8"})
        == "absolute_path"
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


@pytest.mark.parametrize("container_kind", ["list", "tuple", "mapping"])
def test_sanitizer_rejects_distinct_values_that_collapse_to_one_safe_value(
    container_kind: str,
) -> None:
    values = [r"C:\private\one.csv", r"D:\private\two.csv"]
    if container_kind == "list":
        payload: object = values
    elif container_kind == "tuple":
        payload = tuple(values)
    else:
        payload = {"first": values[0], "second": values[1]}

    with pytest.raises(ValueError, match="sanitize_value_collision"):
        sanitize_shared_object(payload)


def test_sanitizer_allows_repeated_identical_values() -> None:
    value = r"C:\private\one.csv"

    assert sanitize_shared_object([value, value]) == ["[local-path]", "[local-path]"]


def test_sanitizer_mapping_graph_identity_is_key_order_independent() -> None:
    first = {"path": r"C:\private\one.csv", "label": "same"}
    second = {"label": "same", "path": r"D:\private\two.csv"}

    with pytest.raises(ValueError, match="sanitize_value_collision"):
        sanitize_shared_object([first, second])


def test_sanitizer_preserves_mapping_order_for_identical_reordered_graphs() -> None:
    first = {"path": r"C:\private\one.csv", "label": "same"}
    second = {"label": "same", "path": r"C:\private\one.csv"}

    assert sanitize_shared_object([first, second]) == [
        {"path": "[local-path]", "label": "same"},
        {"label": "same", "path": "[local-path]"},
    ]


def test_sanitizer_rejects_mapping_key_collision_after_redaction() -> None:
    with pytest.raises(ValueError, match="sanitize_key_collision"):
        sanitize_shared_object(
            {
                r"C:\private\one.csv": "first",
                r"D:\private\two.csv": "second",
            }
        )


def test_sanitizer_rejects_direct_and_mutual_cycles() -> None:
    direct: list[object] = []
    direct.append(direct)
    left: list[object] = []
    right: list[object] = [left]
    left.append(right)

    with pytest.raises(ValueError, match="sanitize_cycle"):
        sanitize_shared_object(direct)
    with pytest.raises(ValueError, match="sanitize_cycle"):
        sanitize_shared_object(left)


def test_sanitizer_rejects_excessive_container_depth() -> None:
    payload: object = "leaf"
    for _ in range(66):
        payload = [payload]

    with pytest.raises(ValueError, match="sanitize_recursion_limit"):
        sanitize_shared_object(payload)


def test_sanitizer_orders_heterogeneous_sets_by_safe_identity() -> None:
    assert sanitize_shared_object({2, "1", None}) == [None, 2, "1"]


def test_sanitizer_makes_scanner_rejected_path_scalars_share_safe() -> None:
    first = Path(r"C:\private\one.csv")
    second = Path(r"D:\private\two.csv")

    assert sanitize_shared_object(first) == "[local-path]"
    assert sanitize_shared_object(1) == 1
    with pytest.raises(ValueError, match="sanitize_value_collision"):
        sanitize_shared_object([first, second])


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


def test_sanitizer_rejects_distinct_keys_that_collapse() -> None:
    with pytest.raises(ValueError, match="sanitize_key_collision"):
        sanitize_shared_object(
            {
                "token=abcdefgh": "first",
                "token=ijklmnop": "second",
            }
        )


def test_sanitizer_rejects_non_string_mapping_keys_before_json_write() -> None:
    with pytest.raises(ValueError, match="sanitize_mapping_key_must_be_string"):
        sanitize_shared_object({1: "value"})
