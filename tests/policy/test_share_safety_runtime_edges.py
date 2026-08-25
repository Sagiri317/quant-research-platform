from __future__ import annotations

import codecs
import tracemalloc

import pytest

from quant_research_platform.policy.share_safety import (
    MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS,
    assert_internal_artifact_metadata_safe,
    evidence_json_value_unsafe_reason,
    evidence_text_stream_unsafe_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_bounded_evidence_text_scan_preserves_chunk_and_utf8_semantics() -> None:
    safe = "研究摘要\nreview/summary.md\n".encode()
    unsafe_chunks = (b"safe prefix tok", b"en=abcdef", b"123456\n")

    assert (
        evidence_text_stream_unsafe_reason((safe[:1], safe[1:4], safe[4:9], safe[9:]))
        is None
    )
    assert evidence_text_stream_unsafe_reason(unsafe_chunks) == "credential"


def test_bounded_text_scan_does_not_retain_document() -> None:
    chunks = (f"metric_{index}=safe\n".encode() for index in range(100_000))

    assert evidence_text_stream_unsafe_reason(chunks) is None


@pytest.mark.parametrize(
    "chunks",
    [
        (b"market_id=safe\n", b"asset_id=safe\n"),
        (b"asset_id=safe\n", b"timestamp=safe\n"),
    ],
)
def test_bounded_evidence_text_scan_preserves_cross_line_row_detection(
    chunks: tuple[bytes, bytes],
) -> None:
    assert evidence_text_stream_unsafe_reason(chunks) == "row_sample"


def test_bounded_evidence_text_scan_rejects_split_invalid_utf8() -> None:
    with pytest.raises(UnicodeDecodeError):
        evidence_text_stream_unsafe_reason((b"safe \xe4", b"invalid"))


@pytest.mark.parametrize(
    ("extra_characters", "expected"),
    [(0, None), (1, "evidence_capacity_exceeded")],
)
def test_bounded_evidence_text_scan_enforces_exact_line_contract(
    extra_characters: int,
    expected: str | None,
) -> None:
    total = MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS + extra_characters
    chunks = (b"x" for _index in range(total))

    assert evidence_text_stream_unsafe_reason(chunks) == expected


def test_bounded_evidence_text_scan_slices_arbitrary_giant_input_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_sizes: list[int] = []
    real_factory = codecs.getincrementaldecoder("utf-8")

    class RecordingDecoder:
        def __init__(self, *, errors: str) -> None:
            self._decoder = real_factory(errors=errors)

        def decode(self, value, *, final: bool = False):
            observed_sizes.append(len(value))
            return self._decoder.decode(value, final=final)

    monkeypatch.setattr(
        "quant_research_platform.policy.share_safety.codecs.getincrementaldecoder",
        lambda _encoding: RecordingDecoder,
    )
    payload = ("研究=safe\n" * 100_000).encode()

    assert evidence_text_stream_unsafe_reason((payload,)) is None
    assert max(observed_sizes) <= MAX_REVIEW_SAFE_EVIDENCE_LINE_CHARACTERS


def test_python_allocation_diagnostic_for_many_empty_evidence_fragments() -> None:
    tracemalloc.start()
    try:
        assert evidence_text_stream_unsafe_reason(b"" for _ in range(500_000)) is None
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 2 * 1024 * 1024


def test_json_share_safety_retains_key_context_after_escape_decoding() -> None:
    assert evidence_json_value_unsafe_reason({"path": "outputs/x/a.md"}) is None
    assert (
        evidence_json_value_unsafe_reason({"api_key": "safe-looking"}) == "credential"
    )
    assert (
        evidence_json_value_unsafe_reason({"path": "C:\\private\\a.md"})
        == "absolute_path"
    )


def test_typed_identifiers_are_not_free_text_stock_guesses() -> None:
    assert unsafe_for_share_reason("run_000001") == "stock_code_sample"
    assert evidence_json_value_unsafe_reason({"run_id": "run_000001"}) is None
    assert (
        evidence_json_value_unsafe_reason({"message": "run_000001"})
        == "stock_sample"
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
        r"outputs/x/C:\private\a.md",
        r"outputs/x/\\server\share\a.md",
        "outputs/x//workspace/qrp/a.md",
        "outputs/x/anaconda3/envs/private/a.md",
        "/etc/passwd",
        "/opt/qrp/data",
        "/srv/qrp/run",
    ),
)
def test_failure_path_rejects_absolute_tokens(path: str) -> None:
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


@pytest.mark.parametrize("path", ("/etc/passwd", "/opt/qrp/data", "/srv/qrp/run"))
def test_sanitizer_redacts_general_posix_absolute_paths(path: str) -> None:
    assert unsafe_for_share_reason(path) == "local_path"
    assert path not in sanitize_shared_text(path)


def test_sanitizer_redacts_punctuation_credential_without_url_corruption() -> None:
    for text, secret in (
        (
            "api_key=!AbcdefghXYZ url=https://example.com/api/v1 provider=Acme",
            "!AbcdefghXYZ",
        ),
        (
            'api_key="!AbcdefghXYZ" url=https://example.com/api/v1 provider=Acme',
            "!AbcdefghXYZ",
        ),
        (
            "api_key='!AbcdefghXYZ' url=https://example.com/api/v1 provider=Acme",
            "!AbcdefghXYZ",
        ),
    ):
        assert unsafe_for_share_reason(text) == "credential"
        sanitized = sanitize_shared_text(text)

        assert secret not in sanitized
        assert "https://example.com/api/v1" in sanitized
        assert "provider=Acme" in sanitized


@pytest.mark.parametrize(
    "payload",
    (
        {"Authorization": "Bearer abc123"},
        {"auth": "Basic abc123"},
        {"private_key": "secret-key"},
    ),
)
def test_evidence_rejects_structured_credential_keys(
    payload: dict[str, str],
) -> None:
    assert evidence_json_value_unsafe_reason(payload) == "credential"


@pytest.mark.parametrize(
    "text",
    (
        "Authorization: Bearer abc123",
        "auth=Basic abc123",
        'Authorization="Bearer abc123"',
        "private_key=!AbcdefghXYZ",
    ),
)
def test_sanitizer_redacts_explicit_authorization_and_private_key(
    text: str,
) -> None:
    assert unsafe_for_share_reason(text) == "credential"
    assert sanitize_shared_text(text) == "[credential]"


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
        # intentional unsafe-path negative case
        "/workspaces/qrp/reports/private_research/a_share/run",
        "/opt/qrp/data/processed/a_share/factor/demo",
        r"C:\repo\data\research\a_share\demo",
        # intentional unsafe-path negative case
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
        # intentional unsafe-path negative case
        "reports/private_research/a_share/run",
        # intentional unsafe-path negative case
        "/workspace/qrp/reports/private_research/a_share/run",
    ],
)
def test_sanitizer_rejects_local_output_paths(text: str) -> None:
    # intentional unsafe-path negative case
    assert text.endswith("reports/private_research/a_share/run")
    assert unsafe_for_share_reason(text) == "local_only_path"


def test_sanitizer_redacts_path_and_credential_preserves_business_text() -> None:
    # Sanitization is narrow; strict evidence scanners still own these markers.
    text = (
        "data/raw/csmar/export.csv stock code 600000.SH CSMAR vendor token=abcdef123456"  # noqa: E501
    )

    sanitized = sanitize_shared_text(text)

    assert "data/raw/csmar" not in sanitized
    assert "600000.SH" in sanitized
    assert "CSMAR" in sanitized
    assert "abcdef123456" not in sanitized


def test_sanitizer_preserves_ordinary_identifiers_and_business_text() -> None:
    text = "600000 300001 provider_name=Acme business order 600000"

    assert sanitize_shared_text(text) == text


def test_share_safety_preserves_noncredential_counterexamples() -> None:
    safe_texts = (
        "authorization model",
        "basic factor model",
        "https://example.com/api/v1?mode=basic",
    )
    for text in safe_texts:
        assert unsafe_for_share_reason(text) is None
        assert evidence_json_value_unsafe_reason({"message": text}) is None
        assert sanitize_shared_text(text) == text

    public_key = {"public_key": "ssh-ed25519 AAAAexample"}
    assert unsafe_for_share_reason(str(public_key)) is None
    assert evidence_json_value_unsafe_reason(public_key) is None
    assert sanitize_shared_text(str(public_key)) == str(public_key)


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
    assert sanitize_shared_text(text) == text
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
def test_provider_source_fields_are_rejected_but_not_rewritten(text: str) -> None:
    assert unsafe_for_share_reason(text) == "provider_source_marker"
    assert sanitize_shared_text(text) == text


def test_row_sample_payload_is_rejected_but_not_rewritten() -> None:
    text = 'row_samples=[{"asset_id":"000001","value":1.2}]'

    assert unsafe_for_share_reason(text) == "row_sample"
    assert sanitize_shared_text(text) == text


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
    sanitized = sanitize_shared_text(text)
    if reason == "local_run_path":
        assert sanitized == "[local-run-path]s1/summary.json"
    else:
        assert sanitized == text


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
