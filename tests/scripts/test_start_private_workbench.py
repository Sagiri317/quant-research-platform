from __future__ import annotations

from pathlib import Path


def test_start_private_workbench_records_cors_runtime_identity() -> None:
    script = Path("scripts/start_private_workbench.ps1").read_text(encoding="utf-8")
    cors_assignment = (
        '$CorsOrigins = "http://$WebHost`:$WebPort;http://localhost`:$WebPort"'
    )

    assert cors_assignment in script
    assert "$record.webPort -eq $WebPort" in script
    assert "$record.corsOrigins -eq $CorsOrigins" in script
    assert "$env:QRP_CORS_ORIGINS = $CorsOrigins" in script
    assert "corsOrigins = $CorsOrigins" in script
