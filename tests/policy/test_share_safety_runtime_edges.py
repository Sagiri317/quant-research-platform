from __future__ import annotations

import pytest

from quant_research_platform.policy.share_safety import (
    assert_internal_artifact_metadata_safe,
    evidence_json_value_unsafe_reason,
    sanitize_shared_text,
    unsafe_for_share_reason,
)


def test_json_share_safety_retains_key_context_after_escape_decoding() -> None:
    assert evidence_json_value_unsafe_reason({"path": "outputs/x/a.md"}) is None
    assert (
        evidence_json_value_unsafe_reason({"api_key": "safe-looking"}) == "credential"
    )
    assert (
        evidence_json_value_unsafe_reason({"path": "C:\\private\\a.md"})
        == "absolute_path"
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


@pytest.mark.parametrize(
    "text",
    (
        '{"password": "SYNTHETIC_SECRET"}',
        "{'password': 'SYNTHETIC_SECRET'}",
        '{"nested": {"api_key" : "SYNTHETIC_SECRET"}}',
        '{"access_token": "SYNTHETIC_SECRET"}',
        '{"client_secret": "SYNTHETIC_SECRET"}',
        '{"ssh_private_key": "SYNTHETIC_SECRET"}',
        '{"x-api-key": "SYNTHETIC_SECRET"}',
        '{"Authorization": "Bearer SYNTHETIC_SECRET"}',
        "{'auth': 'Basic SYNTHETIC_SECRET'}",
    ),
)
def test_quoted_credentials_share_the_text_and_metadata_boundary(text: str) -> None:
    assert unsafe_for_share_reason(text) == "credential"
    assert evidence_json_value_unsafe_reason({"message": text}) == "credential"
    with pytest.raises(
        ValueError, match="unsafe_internal_artifact_metadata: credential"
    ):
        assert_internal_artifact_metadata_safe({"message": text})
    assert "SYNTHETIC_SECRET" not in sanitize_shared_text(text)


@pytest.mark.parametrize(
    "text",
    (
        r'{"password": "SYNTHETIC_HEAD\"SYNTHETIC_TAIL"}',
        r"{'password': 'SYNTHETIC_HEAD\'SYNTHETIC_TAIL'}",
        'private_key="SYNTHETIC_HEAD\nSYNTHETIC_TAIL"',
        r'{"Authorization": "Bearer SYNTHETIC_HEAD\"SYNTHETIC_TAIL"}',
        '{"Authorization": "SYNTHETIC_HEAD"}',
        'password="SYNTHETIC_HEAD',
        "password='SYNTHETIC_TAIL",
        '{"token":\n"SYNTHETIC_HEAD"}',
        '{"token"\n \n:\n"SYNTHETIC_HEAD"}',
        "auth:\n'Basic SYNTHETIC_TAIL'",
    ),
)
def test_credential_redaction_consumes_the_entire_quoted_value(text: str) -> None:
    assert unsafe_for_share_reason(text) == "credential"
    sanitized = sanitize_shared_text(text)
    assert "SYNTHETIC_HEAD" not in sanitized
    assert "SYNTHETIC_TAIL" not in sanitized


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


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"path": "/workspace/qrp/data/research/a_share/demo"}, "absolute_path"),
        ({"credential": "token=abcdef123456"}, "credential"),
        ({"/srv/synthetic/report": 1}, "absolute_path"),
        ({r"C:\synthetic\report": None}, "absolute_path"),
        ({r"\\synthetic\share\report": {}}, "absolute_path"),
        ({"nested": [{"/srv/synthetic/report": []}]}, "absolute_path"),
        ({"nested": {r"C:\synthetic\report": [{"value": 1}]}}, "absolute_path"),
        ({"nested": [{r"\\synthetic\share\report": {"value": 1}}]}, "absolute_path"),
        ({"//synthetic/share/report": 1}, "absolute_path"),
        ({"prefix /srv/synthetic/report": 1}, "absolute_path"),
        ({"/": None}, "absolute_path"),
        ({"C:\\": {}}, "absolute_path"),
        ({"nested": [{"//synthetic/share/report": []}]}, "absolute_path"),
        ({"nested": {"/": [{"value": 1}]}}, "absolute_path"),
        ({"nested": [{"C:\\": {"value": 1}}]}, "absolute_path"),
        ({"/srv/synthetic/report_token": {}}, "credential"),
        ({"row_samples": [{"/srv/synthetic/report": 1}]}, "absolute_path"),
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


def test_internal_metadata_preserves_schema_fields_and_typed_identifiers() -> None:
    payload = {
        "schema_id": "provider-schema",
        "source_id": "choice",
        "fields": [{"name": "api_key", "dtype": "string"}],
        "key_columns": ["source_asset_id"],
        "source_asset_id": "string",
        "data_id": "dataset_600000",
        "factor_id": "factor_000001",
        "run_id": "run_000001",
        "manifest_path": "data/research/demo/_manifest/manifest.json",
        "empty": {"rows": []},
        "relative_keys": {
            "synthetic/share/report": 1,
            r"synthetic\share\report": 2,
            "C:relative": 3,
        },
    }
    assert_internal_artifact_metadata_safe(payload)


@pytest.mark.parametrize("key", ["_token", "_api_key", "🔑_token"])
def test_leading_separator_credentials_match_structured_keys(key):
    import json

    value = {key: "SYNTHETIC_CANARY"}
    text = json.dumps(value, ensure_ascii=False)
    assert evidence_json_value_unsafe_reason(value) == "credential"
    assert unsafe_for_share_reason(text) == "credential"
    assert "SYNTHETIC_CANARY" not in sanitize_shared_text(text)


@pytest.mark.parametrize(
    "text",
    (
        "600000.SH",
        "run_000001",
        "source_id=csmar",
        "provider_id=choice",
        "row_samples=[{asset_id: 000001, timestamp: 2026}]",
        "data/raw/csmar/example.csv",
        ".run/private/s1/summary.json",
        "source_asset_id",
        "exact_runtime_payload",
        "x" * 100_000,
    ),
)
def test_research_content_is_allowed_without_redaction(text):
    assert unsafe_for_share_reason(text) is None
    assert evidence_json_value_unsafe_reason({"message": text}) is None
    assert_internal_artifact_metadata_safe({"message": text})
    assert sanitize_shared_text(text) == text
