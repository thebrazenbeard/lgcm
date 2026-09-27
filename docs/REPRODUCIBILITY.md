# Reproducibility

LGCM separates deterministic qualification evidence from environment-dependent resource measurements.

## Supported development baseline

The package requires Python 3.12 or newer. CI exercises Python 3.12, 3.13, and 3.14 on Ubuntu and Windows. Python 3.15 is tested only as a non-blocking prerelease compatibility job until its final release.

Create an isolated environment and install development dependencies:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -U pip
.venv/Scripts/python -m pip install -e ".[dev]"
```

On POSIX systems use `.venv/bin/python` instead.

## Deterministic qualification receipt

Example:

```python
import lgcm

result = lgcm.run_qualification(
    seeds=(0, 1, 2, 3, 4),
    lengths={"A": 80, "B": 80, "A_RETURN": 80, "C": 80},
    gradual_modes=(False, True),
)
print(result.receipt.canonical_json())
```

The canonical receipt includes the seed-level runs and aggregate statistics but excludes elapsed wall-clock timing. Identical source, dependency versions, configuration, seeds, and deterministic numerical execution are expected to produce the same canonical receipt.

Floating-point libraries, BLAS implementations, architectures, or dependency changes can alter low-order numerical results. Preserve the environment when comparing exact receipts.

## Resource measurements

`result.resources` contains wall-clock elapsed time for each model/seed/stream run. Timing is deliberately excluded from `canonical_json()` because it is host-dependent.

## Repository qualification commands

```text
python -m pytest -q --cov=lgcm --cov-report=term-missing
ruff check src tests
python -m mypy src/lgcm
python -m build --sdist --wheel
python -m pip check
```

A package build is separate evidence from source tests. For release-style verification, install the built wheel into a clean environment and run a smoke prediction from the installed package.

## Interpreting results

A qualification receipt supports only the exact learner implementation, simulator, configuration, seed set, and metrics represented by the receipt. Aggregate means should be read together with standard deviation, minimum, maximum, and seed-level routing behavior.
