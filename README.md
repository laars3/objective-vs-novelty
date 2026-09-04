# When Does Novelty Beat the Objective?
### Deception as the moderator of novelty search in evolved 3D structure building

**Status: in progress.** The environment, agent, and objective-driven evolutionary
loop run end to end, and both task objectives are implemented. Novelty search and
the experiments are not built yet. Build tracking and milestones live in
[TODO.md](TODO.md).

---

## Abstract

Novelty search selects individuals for being *behaviorally different* from
everything tried before, rather than for scoring well on the objective. Lehman
and Stanley (2011) showed it can outperform direct optimization, but argued this
holds specifically on *deceptive* problems, where following the objective's local
gradient leads away from the global optimum. This project tests that claim as a
controlled experiment. Populations of small neural networks build 3D voxel
structures block by block under support and balance rules. Four selection
strategies (random, objective, novelty, blend) are run on *two tasks*: a
deceptive **reach** task, where extending horizontally requires first placing
counterweights that do not improve the score, and a non-deceptive **tower** task,
where greedy stacking is optimal. The prediction is an *interaction*: novelty
search beats objective search on the reach task and loses on the tower task.
Results are reported over 10 or more seeds per condition with nonparametric
statistics, and every stage of the search is inspectable in an interactive
PyVista viewer.

## 1. Research Question and Hypotheses

> **Is deception the condition under which novelty search outperforms direct
> objective optimization in a structural design task?**

Stated as testable hypotheses, in the order they must be established:

- **H1 (the task is deceptive).** On the reach task, objective-driven search
  plateaus below the best structure reachable under the same rules and budget,
  established by a hand-constructed reference solution. Without H1, H2 is
  meaningless: a low score could simply mean the budget ran out.
- **H2 (main effect).** On the reach task, the novelty arm's final best
  objective value exceeds the objective arm's (Mann-Whitney U over 10 or more
  seeds, alpha = 0.05, reported with effect size).
- **H3 (the effect reverses without deception).** On the tower task, the
  objective arm performs at least as well as the novelty arm.

H2 alone says "novelty won once." H2 **and** H3 together say *deception is the
moderator*, which is the actual scientific claim, and it is falsifiable in both
directions. A null result on H2, if H1 holds, is also a reportable finding,
consistent with critiques that novelty search is sensitive to setup.

## 2. The Environment

An `N x N x N` voxel grid. A **base cell** is fixed at the center of the ground
plane, at `(N//2, N//2, 0)`. Agents place one block per step, up to a budget of
`B` blocks. A placement is **valid** only if both hold:

1. **Support:** the cell is empty and is face-adjacent to an existing block (or
   is the base cell). The structure is one connected object grown from the base.
2. **Balance:** after placement, the center of mass stays over the base. With
   unit-mass blocks at positions `(xi, yi, zi)`,
   `|mean(xi) - x_base| <= 0.5` and `|mean(yi) - y_base| <= 0.5`,
   half a block width, the tipping condition for a one-cell footprint.
   Balance must hold at **every intermediate step**, so build order matters.

Invalid placements are masked out of the agent's action space and never executed.

## 3. The Two Tasks

**Reach (deceptive).** Objective: `max(xi) - x_base`, horizontal extent in the
**+x direction only**. This is a cantilever in the structural-engineering sense.
Every block placed toward +x shifts the center of mass toward +x, and the balance
rule halts pure greedy extension after a couple of blocks. Extending further
requires placing counterweight blocks toward -x, moves that are
**objective-neutral** (they never increase the score) but that enable future
gains. Greedy hill-climbing on the objective refuses exactly the moves the
optimum requires. The optimal shape is a balanced two-sided arm, built by
alternating sides. This is deception in Goldberg's original sense, realized
physically.

**Tower (control, non-deceptive).** Objective: `max(zi)`, height. A straight
column above the base never moves the center of mass, so greedy stacking is
optimal and every objective-improving move stays valid. Hill-climbing cannot be
trapped.

Same environment, same agents, same budget. Only the scoring changes, so any
performance flip between arms is attributable to the task's structure.

## 4. Method

**Agents.** A small MLP (`N^3 -> 128 -> N^3`) maps the current grid to logits over
cells. Invalid cells are masked to large negative values and the argmax cell is
placed. The policy is deterministic: identical weights on an identical grid
produce an identical structure. No gradient descent is used anywhere. Networks
are trained only by evolution.

**Evolutionary loop** (identical across arms, only the selection score differs).
Each generation, every agent builds one structure on its own fresh grid, agents
are scored, the top fraction survive, and the next generation is bred from
survivors' weights with Gaussian mutation (`theta' = theta + sigma * epsilon`),
plus elitism (the best agent copied unchanged).

**The four arms:**

| Arm | Selection score |
|-----|-----------------|
| `random` | none, random valid placements (floor baseline) |
| `objective` | task objective only |
| `novelty` | mean distance to k nearest neighbors in behavior space (population plus archive) |
| `blend` | `(1-rho) * objective + rho * novelty`, rho = 0.5 |

**Behavior characterization (BC).** Each structure is summarized as a 5-vector
describing what was built:
`[block_count, max_height, max_radial_reach, mean_horizontal_offset, com_height]`,
each dimension normalized to [0, 1] by its physical maximum. Novelty is the mean
Euclidean k-NN distance (k = 15) against the current population plus an
**archive**; each generation, individuals whose novelty exceeds a threshold are
added to the archive.

Note that the BC partially overlaps the objectives: `max_height` is the tower
objective, and `max_radial_reach` is a direction-agnostic relative of the reach
objective. This is standard rather than a flaw. Lehman and Stanley's original
maze BC was the robot's final position, which fully contained their objective
(distance from that position to the goal). The BC is nonetheless the
load-bearing design choice of novelty search, and a known critique (Pugh et al.,
2016) is that results hinge on it, which is why it is fixed *before* running
experiments and shared across all arms and both tasks.

## 5. Experimental Protocol

A 2 x 4 factorial: {reach, tower} x {random, objective, novelty, blend}.

Every comparison runs **within a task**, across arms. The two tasks are never
compared to each other, because their objectives have different ranges: tower
scores reach `N-1`, while reach scores reach only about `N/2 - 1`, since the base
sits at the grid center and only half the grid lies in +x.

- **10 or more seeds per condition** (80 or more runs), everything derived from
  one logged seed.
- Population 100, ~200 generations, fixed mutation sigma, identical across arms.
  Hyperparameters are tuned (lightly) on the *tower* task only, never on the
  reach task, to avoid favoring any arm on the task that decides H2.
- **Grid size N = 16, budget B = 40.** N is chosen for objective resolution
  rather than realism: at N = 8 the reach objective admits only four distinct
  values (0 to 3) and the ceiling is reachable in six blocks, which is too coarse
  an instrument for the statistics in H2. At N = 16 reach spans 0 to 7.
- **Primary metric:** best objective value found so far, per generation.
- **Secondary metrics:** behavior-space coverage (fraction of a fixed BC grid
  visited) and population BC variance, the *mechanism* evidence that novelty
  explores where objective clusters.
- **Statistics:** Mann-Whitney U on final best objective (novelty versus
  objective, per task), Holm-corrected, with Cliff's delta effect size. Plots
  show median and interquartile bands over seeds, not single runs.
- **Reproducibility:** each run saves config, seed, per-generation scores,
  behaviors, and block placements under `runs/`. Placements are logged rather
  than final grids, since the ordered placement list reconstructs the grid and
  also drives build playback in the viewer. Every figure regenerates from
  `runs/` with one script.

**Planned ablation (one, kept small):** archive on/off for the novelty arm on the
reach task. Does the archive matter, or is within-population diversity enough?

## 6. Visualization

An interactive desktop explorer built on PyVista, the qualitative half of the
evidence:

- Any saved structure rendered as voxels, freely rotated and zoomed.
- **Build playback:** replay a structure block by block from its logged placement
  order, showing when counterweights appear relative to reach gains.
- **Generation slider:** scrub evolutionary history and watch the best reach
  design discover counterweights, or fail to.
- **Side-by-side arms** at the same generation on the same task.
- 2D **behavior-space scatter** per generation: novelty spreading versus
  objective clustering, animated over time.
- Exported GIFs of the above for the paper.

## 7. Limitations and Scope

Stated up front, since several are design choices rather than oversights.

**No physics engine.** Balance is a center-of-mass rule, not a simulation. This
is deliberate: the deception is analyzable precisely because the physics is
simple enough to reason about by hand, which is what makes the hand-constructed
H1 reference solution possible.

**Coarse, integer-valued objectives.** Both scores are small integers. Even at
N = 16 the reach objective spans only 0 to 7, so ties between runs are common and
the statistics lose power accordingly. Effect sizes are reported alongside
p-values partly for this reason. A ceiling effect is also possible: an arm that
saturates the objective cannot show how much better it was than one that merely
reached the same ceiling.

**Compute is dominated by move generation.** `valid_move()` currently scans all
`N^3` cells per placement, and at target settings this accounts for roughly 98%
of runtime, far exceeding the cost of the network forward passes. The support
rule implies only cells face-adjacent to existing blocks can ever be valid,
bounding the true candidate set by the structure size rather than the grid size.
Implementing that is a prerequisite for the full 80-run sweep to be an overnight
job rather than a multi-day one.

**Deterministic policies.** An agent maps a grid to a placement with no sampling,
so one agent produces exactly one structure. This makes fitness evaluation
noise-free, which helps the statistics, but means there is no within-lifetime
exploration; all exploration happens across generations.

**Weight-space search at scale.** At N = 16 each agent carries roughly 1.05
million parameters, evolved by Gaussian mutation with a population of 100. This
is a demanding regime for neuroevolution, and the `random` arm exists partly to
establish whether any arm meaningfully beats chance.

**Single BC.** Results may depend on the behavior characterization (Pugh et al.,
2016). The BC is fixed before experiments and shared across conditions, but a
BC ablation is listed as a stretch goal rather than a core result.

## Running it

```
python src/main.py
```

Runs a single objective-arm experiment and renders one resulting structure.
Config-driven runs (one YAML per condition) are planned, not yet built. See
[TODO.md](TODO.md) for current state.

## Stack

Python 3.9+, PyTorch (agent networks), NumPy, PyVista with PySide6, Matplotlib,
SciPy (statistics).

## References

- Lehman, J. and Stanley, K. O. (2011). *Abandoning Objectives: Evolution Through
  the Search for Novelty Alone.* Evolutionary Computation, 19(2).
- Goldberg, D. E. (1987). *Simple genetic algorithms and the minimal, deceptive
  problem.* In Genetic Algorithms and Simulated Annealing.
- Mouret, J.-B. and Clune, J. (2015). *Illuminating search spaces by mapping
  elites.* arXiv:1504.04909.
- Pugh, J. K., Soros, L. B., and Stanley, K. O. (2016). *Quality Diversity: A New
  Frontier for Evolutionary Computation.* Frontiers in Robotics and AI.
- Stanley, K. O. and Lehman, J. (2015). *Why Greatness Cannot Be Planned: The
  Myth of the Objective.* Springer.
