# LGCM Evaluation Protocol

Status: qualification protocol for LGCM-0 research results.

## Question

The primary question is whether LGCM can improve prequential action-conditioned dynamics prediction across changing latent regimes while retaining and reusing useful prior context, without receiving task or regime identifiers as learner input.

## Evaluation boundary

Every event is evaluated in this order:

1. the evaluator calls `predict(observation, action)`;
2. prediction error is fixed from the unseen next observation;
3. the evaluator calls `update(experience)`;
4. evaluator-only metadata records the regime/segment and update receipt.

Regime and segment labels are never attached to `Experience` and are never supplied to learner prediction or update calls.

## Streams

The canonical synthetic stream contains:

- `A_INITIAL`: initial exposure to regime A;
- optional `A_TO_B_DRIFT`: gradual interpolation from A to B;
- `B`: changed dynamics;
- `A_RETURN`: recurrence of the original A dynamics;
- `C`: a novel nonlinear regime.

Both abrupt and gradual variants are run over multiple fixed seeds.

## Compared models

Every qualification receipt compares:

- `lgcm`: contextual shared-plus-residual learner;
- `persistence`: next state equals current state;
- `observation_only`: RLS dynamics baseline with action information suppressed;
- `global_ridge`: cumulative action-conditioned RLS;
- `ewrls`: exponentially weighted action-conditioned RLS.

A result is not interpreted as an LGCM gain unless it is compared against the relevant baseline results from the same stream configuration and seed set.

## Predeclared metrics

Primary prediction metrics:

- overall prequential MSE;
- per-segment prequential MSE;
- per-segment adaptation error AUC;
- samples to a fixed error criterion for a configured number of consecutive samples.

Retention and routing diagnostics:

- A-return minus initial-A tail error (`forgetting_delta`);
- final expert count;
- switch count;
- spawn count;
- capacity-exhaustion count;
- spawns during `A_RETURN`;
- whether the expert established during late `A_INITIAL` is reused during `A_RETURN`.

Aggregate results report mean, population standard deviation, minimum, and maximum across seeds. Seed-level runs remain in the receipt so aggregate values cannot hide unstable routing behavior.

## Determinism and resource measurements

`QualificationReceipt.canonical_json()` excludes wall-clock timing and is intended to be byte-stable for identical code, configuration, and seeds on deterministic numerical execution.

Wall-clock elapsed time is returned separately as `QualificationResource`. It is evidence about the particular execution environment, not deterministic receipt content.

## Claim discipline

A passing receipt supports only the exact implementation, configuration, streams, seeds, and metrics represented by that receipt. It does not establish unrestricted continual learning, general intelligence, semantic understanding, deployment fitness, or Vera integration.

## Research basis

This protocol follows the continual-learning evaluation principle that forgetting alone is insufficient and that transfer, adaptation, memory/resource behavior, and temporal evaluation matter. It also preserves the task-free/online concern emphasized by CALM: task boundaries or latent-domain labels must not be supplied to the learner.

References:

- Díaz-Rodríguez, Lomonaco, Filliat, Maltoni, “Don't forget, there is more than forgetting: new metrics for Continual Learning,” arXiv:1810.13166.
- Kruszewski, Sorodoc, Mikolov, “Evaluating Online Continual Learning with CALM,” arXiv:2004.03340.
