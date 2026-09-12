# When Does Novelty Beat the Objective?

Testing whether deception is the thing that decides when novelty search beats
plain objective optimization.

Small neural nets build 3D voxel structures one block at a time under support
and balance rules. No backprop anywhere, the networks are evolved. Four
selection strategies run on two tasks, one deceptive and one not. The prediction
is that novelty wins on the deceptive one and loses on the other.

This file is my working log: what the project is, where I am, what I found, and
what's next.

## The question

Is deception the condition under which novelty search beats direct objective
optimization?

Three hypotheses, in the order they have to be established:

H1. On the bridge task, objective search plateaus below what is actually
achievable. Confirmed on one seed, see Results. Without this the rest is
meaningless, since a low score could just mean the budget ran out.

H2. On the bridge task, novelty search ends up ahead of objective search.
Mann-Whitney U over 10+ seeds, with effect size.

H3. On the tower task, objective search does at least as well as novelty.

H2 alone only says novelty won once. H2 and H3 together say deception is what
decides it, which is the actual claim, and it can fail in either direction. A
null result on H2 is still worth reporting if H1 holds.

## Setup

### Environment

An N x N x N grid with one base cell fixed at the centre of the ground plane, at
`(N//2, N//2, 0)`. Agents place one block per step up to a budget of B blocks. A
placement is legal only if all three hold:

1. Support. The cell is empty and touches an existing block face-on, or it's the
   base. The structure is one connected thing grown out from the base.
2. Balance. The centre of mass stays over the base, so `|mean(x) - x_base| <= 0.5`
   and the same for y. This has to hold at every step, not just at the end, so
   build order matters.
3. Off the ground. `z > 0` unless it's the base. Only the base touches the floor,
   so reaching sideways is a real cantilever and not a line drawn along the
   ground.

Illegal placements get masked out of the action space, so an agent can't make
one. Worth remembering: balance is a guardrail, not a trap. Whatever deception
exists here can't come from agents over-extending and toppling.

### The two tasks

Bridge, the deceptive one. Score is `max(x) - x_base`, reach in +x only. One
block past the base uses up the half-block of slack, and nothing further out is
legal until ballast goes on the -x side. Those ballast blocks never raise the
score themselves.

Tower, the control. Score is `max(z)`. A column straight up doesn't move the
centre of mass at all, so stacking is always legal and greedy works.

Same environment, same agents, same budget. Only the scoring changes.

### Agents

`Agent` is an MLP, `N^3 -> 128 -> N^3`, mapping the grid to a score per cell.
Illegal cells get masked to -1e9 and the argmax gets placed.

`RandomAgent` has no policy at all. It hands out uniform random scores to the
legal cells, which makes the argmax a uniform random legal placement. This is
the floor baseline.

Three of the four arms use the same `Agent` class. What changes between them is
what selection rewards, not what builds.

### The loop

Each generation every agent builds on its own fresh grid, gets scored, the top
half survive, and the next generation is bred from survivors with Gaussian
mutation plus elitism, meaning the best agent is copied over untouched.

### The arms

| arm | agent | selection score |
|-----|-------|-----------------|
| random | `RandomAgent` | none |
| objective | `Agent` | the task objective |
| novelty | `Agent` | mean distance to k nearest neighbours in behaviour space |
| blend | `Agent` | half objective, half novelty |

### Behaviour characterization

Each structure gets summarised as 5 numbers:
`[block_count, max_height, max_radial_reach, mean_horizontal_offset, com_height]`,
each normalised to [0,1]. Novelty is the mean k-NN distance (k=15) against the
current population plus an archive, and anything novel enough gets added to the
archive.

This overlaps the objectives a bit: `max_height` is the tower objective. That's
normal rather than a problem, Lehman and Stanley's maze BC was the robot's final
position which fully contained their objective. But the BC is the load-bearing
choice in novelty search and results are known to hinge on it (Pugh et al.), so
it's fixed before any experiments and shared across every arm and both tasks.

### Metrics

Three series logged per generation:

- `gen_best`, best score that generation
- `best_so_far`, cumulative max
- `gen_mean`, population mean

`gen_mean` is the one to compare arms on. See the notes below for why.

## Where I am

M0 through M3 done. H1 confirmed on one seed.

- `src/env.py`, grid, base, support, balance, ground rule, `valid_move()`. Also
  `best_possible_bridge(steps)`, which builds the optimal alternating cantilever
  using `valid_move()` as the judge, so whatever it returns is provably legal.
  Hits the ceiling: 5 at N=12 using 11 of 20 blocks, 7 at N=16.
- `src/agent.py`, `Agent` and `RandomAgent`, plus `growth(sigma)` for mutation.
- `src/fitness.py`, `tower_score` and `bridge_score`.
- `src/evolution.py`, the loop. Takes `score_func`, `agent_class`, `seed`. Logs
  the three series plus `best_grid`.
- `src/visu.py`, `render(grid)`, draws a structure as outlined cubes.
- `src/main.py`, runs one experiment, renders the best structure.

Not built yet: novelty search, the archive, run logging, configs, tests, paper.

## Results so far

N=12 with the base at x=6, 20-block budget, population 50, 100 generations,
sigma=0.002, seed 0.

The reference solution reaches 5 using 11 of the 20 blocks, so the budget isn't
the constraint and a plateau below 5 can't be blamed on running out of blocks.

| arm | population mean | best over 5,000 structures |
|-----|-----------------|----------------------------|
| random | 1.93 | 5, one single draw |
| objective | 3.96 | 4 |
| reference | | 5 |

The objective arm does learn. Mean goes from 2.26 to about 3.9 by generation 23,
roughly double random's 1.93. Then it doesn't move for the remaining 77
generations, and it never once produces a 5 across 5,000 structures. H1 holds.

Odd and interesting: random did find a 5. Converging early made objective search
worse than chance at locating the optimum. That's the mechanism H2 is supposed to
depend on, showing up before novelty search even exists.

Why it fails is not what I predicted. Looking at the best structure it produced,
12 of its 21 blocks sit at x < 6, so it found counterweighting fine. It just
spends the budget badly:

- mean(x) ends at 5.57 when balance allows 6.5, so it leaves nearly a whole block
  of reach unused
- blocks scatter across y = 4 to 9, where they do nothing for the score
- 21 blocks for a reach of 4, against the reference's 11 blocks for 5

Why it stops: at the plateau about 48 of 50 agents score exactly 4. Selection
sorts by score and keeps the top half, so with everyone tied the ranking is
arbitrary and there's no pressure left. The search just wanders. That's a
consequence of an integer objective with six possible values.

Single seed so far. Seeds 1 and 2 pending.

## Things I learned that cost time

sigma = 0.002. `nn.Linear` inits weights at std ~0.014 and ~0.05 for the two
layers, so sigma=0.1 shifts a weight by 7x its own size and every child is a
brand new random network. If a fitness curve goes flat, check sigma against
weight scale first.

Best-ever is a bad way to compare arms. It mostly measures how much you sampled.
Random hit the optimum once in 5,000 draws while its mean never budged off 1.93.
Use `gen_mean`.

Selection pressure dies at the plateau. Six possible scores means the population
collapses onto one value, `sorted()` then breaks ties by index, and selection
becomes arbitrary.

The deception isn't "refuses the enabling move". Objective search places
counterweights freely. It fails on efficiency.

Masking removes the textbook deception. Agents can't make illegal moves, so if I
ever redesign this, the deception has to live somewhere an agent can actually go
wrong.

## Next

1. Confirm H1 on seeds 1 and 2. Same settings, bridge task, `Agent`. Only need
   the `gen_mean` plateau and final `best_so_far` from each. Use
   `generations=50`, nothing has ever happened after 25.
2. Then the infrastructure below, before novelty search.

Infrastructure, before M4:

- [ ] Run logging. Runs currently vanish on exit. Write seed, arm, task, config,
  the three series, and block placements to `runs/`. Log placements rather than
  final grids, the ordered list rebuilds the grid and also drives build playback
  later. Already lost several runs to wrong arm, wrong task, or an unprinted
  metric.
- [ ] Config-driven runs. One YAML per condition so the 80-run sweep is a script
  instead of 80 hand-edits of `main.py`.
- [ ] Optimize `valid_move()`. It scans all N^3 cells per placement and is about
  98% of runtime, way more than the network. Support means only cells next to an
  existing block can ever be legal, so the real candidate set is bounded by
  structure size, a few hundred at B=40, not grid size, 4,096 at N=16. This is
  what makes the full sweep an overnight job.
- [ ] Some pytest cases. Env invariants, plus "greedy +x becomes illegal without
  a counterweight" as the deception mechanism written as a test.

## Milestones

- [x] M0, environment rules. Tests still missing.
- [x] M1, one agent builds, viewer shows it.
- [x] M2, evolution works. Checked against the random baseline.
- [x] M3, bridge task and the H1 check. Reference built and verified, objective
  arm plateaus at 4 against an achievable 5. Multi-seed pending.
- [ ] M4, novelty search. BC vector, archive, k-NN novelty, the novelty and
  blend arms, coverage metric. Note `score_func(grid)` can't express novelty,
  which needs the whole population plus archive, so the scoring path has to be
  restructured to score a generation at a time.
- [ ] M5, the full 2x4 with 10+ seeds, Mann-Whitney and Cliff's delta,
  Holm-corrected, and the headline figure.
- [ ] M6, explorer. Build playback from logged placements, generation slider,
  side-by-side arms, BC scatter, GIF export.
- [ ] M7, paper in LaTeX plus a runs-to-figures script.

## Decisions

Settled:

- Ground-level extension removed. `valid_move()` needs `z > 0` except for the
  base, so the bridge task builds a real cantilever.
- The deceptive task is called `bridge` in code. It's a cantilever in the
  engineering sense, which the paper should say once.

Still open:

- Rename `Agent`. It's vague, but `ObjectiveAgent` would be wrong since novelty
  and blend use the same class. `NetworkAgent` or `MLPAgent` names what it is
  instead of which arm uses it. Arm names belong in run configs.
- Move to N=16, B=40 for the real experiment. `main.py` is on N=12 with 20 steps.
  Do it after the `valid_move()` optimization makes the bigger grid affordable.
- Tie-breaking in selection. Pressure vanishes once everyone scores the same. A
  secondary tie-break would restore a gradient but changes the experiment, and
  would have to apply identically to every arm.
- Symmetric ceilings. Tower tops out at N-1, bridge at about N/2 - 1. Not
  required since comparisons are within-task, but a non-cubic grid, wide in x and
  short in z, would even them out. Costs a refactor of `Agent` (`size**3`),
  `env.py`, and `unravel_index`. Deferred.

## Plan for the experiment

2 x 4: {bridge, tower} x {random, objective, novelty, blend}.

Comparisons run within a task, across arms, never between tasks. Tower scores go
up to N-1 and bridge only to about N/2 - 1, so the two aren't on the same scale.

- 10+ seeds per condition, so 80+ runs, everything derived from one logged seed
- population 100, ~200 generations, sigma fixed and identical across arms
- hyperparameters tuned lightly on tower only, never on bridge, so nothing
  favours an arm on the task that decides H2
- primary metric is population mean per generation, best-so-far alongside
- secondary: behaviour-space coverage and population BC variance, the mechanism
  evidence that novelty spreads where objective clusters
- Mann-Whitney U on the final objective, Holm-corrected, with Cliff's delta
- plots show median and IQR over seeds, not single runs
- every run saves config, seed, series and placements under `runs/`, and every
  figure regenerates from there with one script

One small ablation planned: archive on/off for the novelty arm on bridge.

## Known limitations

- No physics. Balance is a centre-of-mass rule. Deliberate, it's what makes the
  reference solution constructible by hand.
- Selection pressure collapses under ties, as above.
- Best-ever is confounded by sampling volume.
- Objectives are coarse integers, so ties between runs are common and the stats
  lose power. Effect sizes reported partly for this reason.
- `valid_move()` dominates runtime.
- `Agent` is deterministic, so one network gives exactly one structure. Makes
  evaluation noise-free but means no exploration within a lifetime.
- At N=16 each agent is ~1.05M parameters evolved by Gaussian mutation with a
  population of 100. That's demanding, and the random arm is partly there to
  check whether any arm beats chance at all.
- One BC, results may hinge on it. A BC ablation is a stretch goal.

## Running it

```
python src/main.py
```

Runs one experiment and renders the best structure it found.

## Stack

Python 3.9+, PyTorch, NumPy, PyVista with PySide6, Matplotlib, SciPy.

## References

- Lehman & Stanley (2011). Abandoning Objectives: Evolution Through the Search
  for Novelty Alone. Evolutionary Computation 19(2).
- Goldberg (1987). Simple genetic algorithms and the minimal, deceptive problem.
- Mouret & Clune (2015). Illuminating search spaces by mapping elites.
  arXiv:1504.04909.
- Pugh, Soros & Stanley (2016). Quality Diversity: A New Frontier for
  Evolutionary Computation. Frontiers in Robotics and AI.
- Stanley & Lehman (2015). Why Greatness Cannot Be Planned. Springer.

## Stretch goals

- MAP-Elites arm, keeping the best structure per BC cell.
- BC ablation, rerunning H2 with a deliberately poor BC, `[block_count]` only.
- Cluster each arm's final structures and characterise build-order patterns.
