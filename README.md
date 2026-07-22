# When Does Novelty Beat the Objective?
### Deception as the moderator of novelty search in evolved 3D structure building

**Status: research plan / in progress** -- environment and agent scaffolding exist;
evolution loop, novelty search, and experiments are being built. Progress
tracking, milestones, and the build plan live in [TODO.md](TODO.md).

---

## Abstract

Novelty search selects individuals for being *behaviorally different* from
everything tried before, instead of for scoring well on the objective. Lehman &
Stanley (2011) showed it can outperform direct optimization, but only on
**deceptive** problems, where following the objective's local gradient leads
away from the global optimum. This project tests that claim as a controlled
experiment. Populations of small neural networks build 3D voxel structures
block by block under simple support and balance rules. Four selection
strategies (random, objective, novelty, blend) are run on **two tasks**: a
deceptive *cantilever* task, where maximizing reach requires first building
counterweights that don't improve the score, and a non-deceptive *tower* task,
where greedy stacking is optimal. The prediction is an **interaction**: novelty
search beats objective search on the cantilever task and loses on the tower
task. Results are reported over 10+ seeds per condition with nonparametric
statistics, and every stage of the search is inspectable in an interactive
PyVista/Qt viewer.

## 1. Research Question & Hypotheses

> **Is deception the condition under which novelty search outperforms direct
> objective optimization in a structural design task?**

Stated as testable hypotheses, in the order they must be established:

- **H1 (the task is deceptive).** On the cantilever task, objective-driven
  search plateaus below the best structure found by *any* method (including a
  hand-constructed reference solution). Without H1, H2 is meaningless.
- **H2 (main effect).** On the cantilever task, the novelty arm's final best
  objective value exceeds the objective arm's (Mann-Whitney U over 10+ seeds,
  α = 0.05, with effect size).
- **H3 (the effect reverses without deception).** On the tower task, the
  objective arm performs at least as well as the novelty arm.

H2 alone says "novelty won once." H2 **and** H3 together say *deception is the
moderator* -- that's the actual scientific claim, and it's falsifiable in both
directions. A null result on H2, if H1 holds, is also a reportable finding
(consistent with critiques that novelty search is sensitive to setup).

## 2. The Environment

An `N×N×N` voxel grid (`N = 8` to start). A **base cell** is fixed at the
center of the ground plane. Agents place one block per step, up to a budget of
`B` blocks (`B = 40` to start). A placement is **valid** only if all hold:

1. **Support:** the cell is empty, and is either the base cell or
   face-adjacent to an existing block. (Ground-level cells outside the base are
   *not* buildable: the structure is one connected object growing from the base.)
2. **Balance:** after placement, the structure's center of mass stays over the
   base: with unit-mass blocks at positions `(xi, yi, zi)`,
   `|mean(xi) - x_base| <= 0.5` and `|mean(yi) - y_base| <= 0.5`
   (half a block width, the tipping condition for a one-cell footprint).
   Balance must hold at **every intermediate step**, so build order matters.

Invalid placements are masked out of the agent's action space (never executed).

## 3. The Two Tasks

**Cantilever (deceptive).** Objective: `max(xi) - x_base`, horizontal reach in
the **+x direction only**. Why this is deceptive: every block placed toward +x
shifts the center of mass toward +x, and the balance rule halts pure greedy
extension after a couple of blocks. Reaching further requires placing
counterweight blocks toward -x, moves that are **objective-neutral** (they
never increase reach) but enable future reach. Greedy hill-climbing on the
objective refuses exactly the moves the optimum requires; the optimal shape is
a balanced two-sided arm, built by alternating sides. This is deception in
Goldberg's original sense, realized physically.

**Tower (control, non-deceptive).** Objective: `max(zi)`, height. A straight
column above the base never moves the center of mass, so greedy stacking
straight up is optimal and every objective-improving move stays valid.
Hill-climbing cannot be trapped.

Same environment, same agents, same budget -- only the scoring changes. Any
performance flip between arms is attributable to the task's structure.

## 4. Method

**Agents.** A small MLP (`N³ → 128 → N³`) maps the current grid to logits over
cells; invalid cells are masked to very large negative values; the argmax cell is placed. No gradient
descent anywhere -- networks are trained only by evolution.

**Evolutionary loop** (identical across arms; only the selection score differs):
each generation, every agent builds one structure; agents are scored; the top
fraction survive; the next generation is built from survivors' weights with
Gaussian mutation (`θ' = θ + σε`), plus elitism (best agent copied unchanged).

**The four arms:**

| Arm | Selection score |
|-----|-----------------|
| `random` | none, random valid placements (floor baseline) |
| `objective` | task objective only |
| `novelty` | mean distance to k nearest neighbors in behavior space (population + archive) |
| `blend` | `(1-ρ)·objective + ρ·novelty`, ρ = 0.5 |

**Behavior characterization (BC).** Each structure is summarized as a
5-vector, chosen to describe *what was built* without encoding either
objective's direction:
`[block_count, max_height, max_radial_reach, mean_horizontal_offset, com_height]`,
each dimension normalized to [0, 1] by its physical maximum. Novelty is the
mean Euclidean k-NN distance (k = 15) against the current population plus an
**archive**; each generation, individuals whose novelty exceeds a threshold are
added to the archive. The BC is the load-bearing design choice of novelty
search -- a known critique (Pugh et al., 2016) is that results hinge on it,
which is why it is fixed *before* running experiments and shared across all
arms and both tasks.

## 5. Experimental Protocol

A 2 × 4 factorial: {cantilever, tower} × {random, objective, novelty, blend}.

- **10+ seeds per condition** (80+ runs); everything derives from one logged seed.
- Population 100, ~200 generations, fixed mutation σ, identical across arms;
  hyperparameters are tuned (lightly) on the *tower* task only, never on the
  cantilever task, to avoid favoring any arm on the task that decides H2.
- **Primary metric:** best objective value found so far, per generation.
- **Secondary metrics:** behavior-space coverage (fraction of a fixed BC grid
  visited) and population BC variance -- the *mechanism* evidence that novelty
  explores where objective clusters.
- **Statistics:** Mann-Whitney U on final best objective (novelty vs.
  objective, per task), Holm-corrected, with Cliff's delta effect size. Plots
  show median and interquartile bands over seeds, not single runs.
- **Reproducibility:** each run saves config, seed, per-generation scores,
  behaviors, and best structures as JSON/NPZ under `runs/`; every figure in the
  paper regenerates from `runs/` with one script.

**Planned ablation (one, kept small):** archive on/off for the novelty arm on
the cantilever task -- does the archive matter, or is within-population
diversity enough?

## 6. Visualization (PyVista + Qt)

An interactive desktop explorer (`pyvistaqt.BackgroundPlotter`), the
qualitative half of the evidence:

- Any saved structure rendered as voxels; rotate/zoom freely.
- **Generation slider:** scrub evolutionary history, watch the best cantilever
  design discover counterweights (or fail to).
- **Side-by-side arms** at the same generation on the same task.
- 2D **behavior-space scatter** per generation: novelty spreading vs. objective
  clustering, animated over time.
- Exported GIFs of the above for the README and paper.

## 7. Deliverables

1. **A short paper** (4-6 pages, workshop format) in `paper/`: abstract,
   related work, method, results with statistics, limitations. Written to
   arXiv-quality standards even if never submitted.
2. **This repository**: tested, config-driven, one command per experiment.
3. **The interactive explorer** plus demo GIFs.

## 8. Repository Structure

```
src/
  env.py          # grid, base cell, support + balance rules, valid-move mask
  agent.py        # MLP policy: grid -> masked logits -> placement
  evolution.py    # generation loop: build, score, select, mutate, elitism
  fitness.py      # cantilever & tower objectives; BC vector; novelty score
  archive.py      # novelty archive (add-above-threshold, k-NN queries)
  visualize.py    # PyVista/Qt explorer, BC scatter, GIF export
  main.py         # run one experiment arm from a config file
configs/          # one YAML per condition (task, arm, seeds, hyperparams)
tests/            # pytest: env invariants, BC math, novelty math
runs/             # saved results (gitignored except an example run)
paper/            # the writeup
```

## 9. Scope & Constraints

Deliberately CPU-runnable, solo-buildable: 8³ grid, 40-block budget,
population ~100, ~200 generations. One run = minutes; the full 2×4×10 sweep is
an overnight CPU job. No GPU, no physics engine -- balance is a center-of-mass
rule, which is the point: the deception is analyzable precisely because the
physics is simple.

## Stack

Python 3.9+ · PyTorch (agent networks) · NumPy · PyVista + pyvistaqt ·
Matplotlib · SciPy (stats)

## References

- Lehman, J. & Stanley, K. O. (2011). *Abandoning Objectives: Evolution Through
  the Search for Novelty Alone.* Evolutionary Computation, 19(2).
- Goldberg, D. E. (1987). *Simple genetic algorithms and the minimal, deceptive
  problem.* In Genetic Algorithms and Simulated Annealing.
- Mouret, J.-B. & Clune, J. (2015). *Illuminating search spaces by mapping
  elites.* arXiv:1504.04909.
- Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). *Quality Diversity: A New
  Frontier for Evolutionary Computation.* Frontiers in Robotics and AI.
- Stanley, K. O. & Lehman, J. (2015). *Why Greatness Cannot Be Planned: The
  Myth of the Objective.* Springer.
