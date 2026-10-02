# LGCM Predictive Latent Compression Experiment Design

Status: FUTURE EXPERIMENT DESIGN / LGCM-0 UNCHANGED  
Date: 2026-10-02  
Source subject: `thebrazenbeard/lgcm@0043c60419de9ef0b52363b62aed8c0a52ef63f5`

## Purpose

Define a later LGCM experiment for learned compression without changing the current LGCM-0 fixed-encoder qualification subject.

LGCM-0 deliberately reserves representation learning for later work. This design respects that boundary.

## Research question

Can a continual predictive learner compress historical/context state into a smaller learned representation while preserving the information needed for future prediction, regime recognition, old-context recall, and bounded planning?

A useful compressor should be judged by future predictive utility under distribution/regime changes, not reconstruction quality alone.

## Candidate experiment

Compare:

1. full retained observation/action history or existing sufficient-state baseline;
2. fixed handcrafted compact state;
3. learned latent bottleneck;
4. learned latent bottleneck with selective access to higher-resolution backing state.

Hold the stream generator, prequential evaluation order, context-capacity limits, and planner budget fixed.

## Required measures

- prediction error before current-outcome update;
- regime-switch detection delay;
- old-context recall accuracy;
- catastrophic forgetting/interference;
- latent-state bytes/dimensions;
- higher-resolution retrieval frequency;
- recovery after returning to an old regime;
- planning outcome reported separately from learner quality.

## Adversarial cases

Include streams where a feature appears low-value for a long period and later becomes predictive. A compressor that permanently discards it should be penalized unless selective backing-state access can recover it.

## Boundary

Do not modify LGCM-0's fixed encoders or promote representation learning into current architecture merely because this design exists. New implementation and empirical claims require a separately frozen experiment subject and qualification evidence.

## Relationship to broader work

Vera Mono is the practical multi-resolution runtime target; Mosaic tests compute/residency composition; SPM tests semantic latent representation; Rezon tests demand-driven resolution upgrades. LGCM's role is narrower: test whether learned compression preserves future predictive usefulness under continual change.
