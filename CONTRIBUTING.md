# Contributing

LGCM is source-visible proprietary research software.

## Current contribution boundary

Issues, reproducible experiment reports, security findings, review notes, and proposed patches are welcome for discussion.

The repository license states that accepted contributions are governed by both this file and `CLA.md`. No `CLA.md` is currently present. Therefore this document does **not** create substitute CLA terms, assign copyright, grant a license, or promise that a code contribution can be accepted or merged.

Until the repository owner publishes or separately agrees to applicable contribution terms, treat code pull requests as proposals for review rather than accepted contributions.

## Engineering expectations

A proposed change should:

- preserve the task-free learner boundary unless the change explicitly changes the research question;
- preserve predict-before-update prequential evaluation;
- add a focused regression test for behavior changes;
- avoid weakening snapshot confinement, digest validation, or restart equivalence;
- include multi-seed evidence for changes that claim learning-quality improvement;
- keep planner results separate from learner qualification.

Before submitting a patch, run:

```text
python -m pytest -q --cov=lgcm --cov-report=term-missing
ruff check src tests
python -m mypy src/lgcm
python -m build --sdist --wheel
python -m pip check
```

The current coverage gate is 90%.

## Research claims

Do not convert a single seed, single metric, simulator-specific result, or internal test pass into a broader claim. State the exact configuration, seed set, baselines, and metric used.
