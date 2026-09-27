# LGCM Qualification Foundation Implementation Plan

> **For agentic workers:** Use the host's available task-by-task implementation workflow. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn LGCM's current working prototype into a trustworthy, reproducible qualification-grade continual-learning research package without inflating its algorithmic scope before evidence supports it.

**Architecture:** Preserve the current `ContextualWorldModel` and evaluator boundary. First harden public immutability and snapshot/update integrity, then add deterministic multi-seed qualification receipts, then add a bounded read-only MPC planner, and finally wire repository automation/documentation around those verified interfaces.

**Tech Stack:** Python 3.12+, NumPy, pytest, Ruff, mypy, setuptools/build, GitHub Actions.

## Global Constraints

- Base all work on `main@0043c60419de9ef0b52363b62aed8c0a52ef63f5`.
- Do not pass evaluator regime or segment metadata into learner prediction/update calls.
- Preserve predict-before-update ordering.
- Do not silently evict experts when capacity is exhausted.
- Keep planner evaluation separate from learner qualification.
- Snapshot loads must not read any payload outside the snapshot directory.
- No direct `main` mutation or merge.
- Every behavior change gets a focused failing test before implementation.
- Full completion requires fresh full-suite, Ruff, mypy, build, and installed-wheel verification.

---

### Task 1: Harden public data and configuration contracts

**Files:**
- Modify: `src/lgcm/types.py`
- Modify: `src/lgcm/model.py`
- Test: `tests/test_types.py`
- Test: `tests/test_model_prequential.py`

**Interfaces:**
- Consumes: `Experience`, `Prediction`, `LGCMConfig`
- Produces: same public signatures with stronger non-mutating validation

- [ ] **Step 1: Add focused failing tests**
  - Caller-owned NumPy inputs remain writeable and unchanged after constructing `Experience` and `Prediction`.
  - Mutating the original input after construction cannot change stored arrays.
  - Invalid finite/range/count/factor configuration values raise `ValueError` at `LGCMConfig` construction.

- [ ] **Step 2: Verify relevant failures**
  Run: `python -m pytest -q tests/test_types.py tests/test_model_prequential.py`
  Expected: new ownership/config tests fail against the base implementation.

- [ ] **Step 3: Implement minimum behavior**
  - Copy vectors before setting read-only flags.
  - Validate regularization, forgetting factors, residual parameters, switch/support parameters, spawn/mismatch values, bootstrap/max-expert counts, and finite bounds.

- [ ] **Step 4: Verify focused pass**
  Run the same focused command; expected all selected tests pass.

- [ ] **Step 5: Run affected integration check**
  Run: `python -m pytest -q tests/test_encoders.py tests/test_baselines.py tests/test_model_context_recall.py tests/test_model_capacity.py`
  Expected: pass.

- [ ] **Step 6: Commit**
  Commit message: `fix: harden public data and config contracts`

### Task 2: Constrain and validate snapshot persistence

**Files:**
- Modify: `src/lgcm/persistence.py`
- Modify: `src/lgcm/model.py`
- Test: `tests/test_persistence.py`
- Test: `tests/test_restart_equivalence.py`

**Interfaces:**
- Consumes: `save_snapshot(model, path, overwrite=False)`, `load_snapshot(path)`
- Produces: same public signatures with confined, manifest-bound, state-consistent snapshots

- [ ] **Step 1: Add focused failing tests**
  - A crafted `../outside.npy` state reference is rejected before load.
  - An absolute payload reference is rejected.
  - A state-referenced payload missing from manifest is rejected.
  - Invalid restored covariance shape/asymmetry and invalid evidence counters are rejected.
  - Snapshot attempts made while `ContextualWorldModel.update` is in progress raise `RuntimeError`.

- [ ] **Step 2: Verify relevant failures**
  Run: `python -m pytest -q tests/test_persistence.py tests/test_restart_equivalence.py`
  Expected: newly added confinement/integrity tests fail against base behavior.

- [ ] **Step 3: Implement minimum behavior**
  - Resolve every referenced payload relative to the snapshot root and reject absolute or escaping paths.
  - Require each referenced payload to exist in the manifest.
  - Validate covariance symmetry and expected shape; validate counters and expert IDs.
  - Set/reset `_update_in_progress` using `try/finally` around the mutable update transaction.
  - Preserve save atomicity and existing digest verification.

- [ ] **Step 4: Verify focused pass**
  Run the same focused command; expected pass.

- [ ] **Step 5: Run affected integration check**
  Run: `python -m pytest -q`
  Expected: entire suite passes.

- [ ] **Step 6: Commit**
  Commit message: `fix: secure snapshot trust boundary`

### Task 3: Add deterministic qualification receipts

**Files:**
- Create: `src/lgcm/qualification.py`
- Create: `tests/test_qualification.py`
- Modify: `src/lgcm/evaluation.py`
- Modify: `src/lgcm/envs/continual_regimes.py`
- Modify: `src/lgcm/__init__.py`
- Create: `docs/EVALUATION_PROTOCOL.md`

**Interfaces:**
- Produce a public `run_qualification(...)` API returning immutable run/aggregate dataclasses and a JSON-serializable receipt.
- Evaluator events carry segment metadata used only by evaluation.
- Learner calls remain `predict(observation, action)` then `update(experience)`.

- [ ] **Step 1: Add focused failing tests**
  - Same seeds/config produce byte-stable canonical receipt data excluding explicitly measured timing fields.
  - A/B/A-return segments remain distinguishable in metrics while learner call signatures contain no segment/regime metadata.
  - All five candidate/baseline models run over abrupt and gradual streams.
  - Aggregate statistics report mean/std/min/max and expert-decision counts.

- [ ] **Step 2: Verify relevant failures**
  Run: `python -m pytest -q tests/test_qualification.py tests/test_evaluator_boundary.py`
  Expected: missing qualification API/segment support failures.

- [ ] **Step 3: Implement minimum behavior**
  - Add explicit segment labels.
  - Add deterministic seed/config/result dataclasses.
  - Calculate predeclared metrics from existing primitives plus switch/spawn/capacity event counts.
  - Serialize receipt with stable key ordering.
  - Keep wall-clock timings in a separate non-deterministic resource section.

- [ ] **Step 4: Verify focused pass**
  Run selected tests; expected pass.

- [ ] **Step 5: Run full qualification smoke**
  Run a fixed small seed set for abrupt and gradual streams and validate receipt schema/invariants.

- [ ] **Step 6: Commit**
  Commit message: `feat: add reproducible qualification receipts`

### Task 4: Add bounded read-only MPC planning

**Files:**
- Create: `src/lgcm/planning.py`
- Create: `tests/test_planning.py`
- Modify: `src/lgcm/__init__.py`
- Modify: `README.md`

**Interfaces:**
- Produce `plan_action(model, observation, objective, *, horizon, candidates, seed, lower_bound, upper_bound)`.
- Return immutable `PlanResult(action, score, candidate_index, seed, horizon, candidates)`.

- [ ] **Step 1: Add focused failing tests**
  - Fixed seed produces deterministic action/result.
  - Planning never changes model generation, active expert, weights, detector state, or evidence counts.
  - Returned action obeys bounds.
  - Invalid horizon/candidate counts or bounds raise `ValueError`.
  - On a simple deterministic control model, chosen action improves objective over a fixed neutral-action baseline.

- [ ] **Step 2: Verify relevant failures**
  Run: `python -m pytest -q tests/test_planning.py`
  Expected: missing planner API.

- [ ] **Step 3: Implement minimum behavior**
  - Uniformly sample bounded candidate action sequences from `numpy.random.default_rng(seed)`.
  - Roll each sequence forward through `model.predict` without calling `update`.
  - Score terminal or accumulated predicted states via the caller objective.
  - Return the first action from the lowest-score candidate.

- [ ] **Step 4: Verify focused pass**
  Run focused tests; expected pass.

- [ ] **Step 5: Run full suite**
  Run: `python -m pytest -q`
  Expected: pass.

- [ ] **Step 6: Commit**
  Commit message: `feat: add bounded model-predictive planner`

### Task 5: Establish repository qualification gates and truthful docs

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `SECURITY.md`
- Create: `CONTRIBUTING.md`
- Create: `docs/REPRODUCIBILITY.md`
- Create: `docs/LGCM_DESIGN_SPEC_V2.md`
- Preserve: `docs/specs/2026-09-27-qualification-first-upgrade-design.md`

**Interfaces:**
- CI commands become the canonical repository qualification gate.

- [ ] **Step 1: Add/adjust tooling configuration**
  Configure Ruff, mypy, coverage, build tooling, package metadata, source-available license expression, and private publication classifier/guard without changing repository licensing terms.

- [ ] **Step 2: Add CI**
  Matrix-test supported Python versions; run pytest with strict warnings/coverage, Ruff, mypy, build, and installed-wheel smoke import.

- [ ] **Step 3: Repair docs**
  - Make README claims correspond exactly to implemented source.
  - Replace nonexistent links with real docs.
  - Document research scope, experiment protocol, snapshot trust model, reproducibility, and planner boundary.
  - `CONTRIBUTING.md` explains technical contribution procedure and explicitly states that no CLA terms are created by that file.

- [ ] **Step 4: Verify gates locally**
  Run: `python -m pytest -q -W error --cov=lgcm --cov-report=term-missing`
  Run: `ruff check src tests`
  Run: `mypy src/lgcm`
  Run: `python -m build --sdist --wheel`
  Install built wheel into a clean venv and import/run a smoke prediction.

- [ ] **Step 5: Verify repository diff**
  Confirm no generated build, environment, coverage, or benchmark scratch artifacts are tracked.

- [ ] **Step 6: Commit**
  Commit message: `chore: establish qualification and repository gates`

## Unresolved externally observable decisions

- Legal CLA terms remain owner-defined; this plan will not invent them.
- Algorithmic expansion beyond the present contextual RLS architecture remains gated on comparative qualification evidence.
