# Environment Contract

QRP Clean Rewrite `0.1.0` formal verification is bound to this environment
contract.

## Supported Runtime

- Official Python: Python 3.13.x
- Python package contract: `>=3.13,<3.14`
- Unsupported / not guaranteed: Python 3.11 / 3.12 / 3.14+
- Primary verified OS: Windows 11 + conda `quantlab313`
- Conda BLAS provider: OpenBLAS, selected through `environment.yml`
- Cross-platform compatible target: macOS / Linux
- macOS/Linux PASS only after tests run on those systems.

Do not claim PASS for an operating system, Python version, dependency set, or
private-data smoke run unless the matching commands actually ran there.

## Fresh Clone Setup

Windows PowerShell:

```powershell
conda env update -f environment.yml --prune
conda run -n quantlab313 python --version
```

macOS/Linux bash:

```bash
conda env update -f environment.yml --prune
conda run -n quantlab313 python --version
```

## Formal Verification

Windows PowerShell:

```powershell
$env:PYTHONPATH="src"
conda run -n quantlab313 python -m pytest tests/architecture -q
conda run -n quantlab313 python -m pytest tests/correctness/test_price_volume_reference.py -q
conda run -n quantlab313 python -m pytest -q
conda run -n quantlab313 python -m pytest --cov=quant_research_platform --cov-report=term-missing -q
conda run -n quantlab313 python -m ruff check src tests
git diff --check
```

macOS/Linux bash:

```bash
export PYTHONPATH=src
conda run -n quantlab313 python -m pytest tests/architecture -q
conda run -n quantlab313 python -m pytest tests/correctness/test_price_volume_reference.py -q
conda run -n quantlab313 python -m pytest -q
conda run -n quantlab313 python -m pytest --cov=quant_research_platform --cov-report=term-missing -q
conda run -n quantlab313 python -m ruff check src tests
git diff --check
```

The formal tests listed above must not depend on private data, `.run` outputs,
handoff packages, or private reports. Real-data smoke, full-horizon rebuilds,
and generated private reports are outside this environment contract unless a
separate task explicitly runs and compares them.

## PASS Semantics

A formal verification PASS is valid only for the exact Python, OS, and
dependency versions recorded with the run. Record at minimum:

- `python --version`
- `python -c "import sys; print(sys.version)"`
- core dependency versions for `numpy`, `pandas`, `pyarrow`, `scikit-learn`,
  `pydantic`, `fastapi`, `typer`, `uvicorn`, `pytest`, `ruff`, `coverage`,
  `hypothesis`, `httpx2`, and `psutil`
- Conda BLAS provider versions for `libblas`, `liblapack`, and `libopenblas`
- OS name/version
- exact commands run

Do not convert a Windows PASS into a macOS/Linux PASS by implication.
