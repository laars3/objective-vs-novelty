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
achievable. Not settled yet, see Results. Without this the rest is meaningless,
since a low score could just mean the budget ran out.

H2. On the bridge task, novelty search ends up ahead of objective search.
Mann-Whitney U over 10+ seeds, with effect size.

H3. On the tower task, objective search does at least as well as novelty.

H2 alone only says novelty won once. H2 and H3 together say deception is what
decides it, which is the actual claim, and it can fail in either direction. A
null result on H2 is still worth reporting if H1 holds.

## Setup

### Environment

A voxel grid, currently 24 x 8 x 12. It does not have to be cubic, and it should
not be: the bridge ceiling depends only on the x dimension, while y and z cost
compute without raising it. One base cell is fixed at the centre of the ground
plane, at `(nx//2, ny//2, 0)`. Agents place one block per step up to a budget of
B blocks. A placement is legal only if all three hold:

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
score themselves. Ceiling is `nx - 1 - nx//2`, so 11 on the current grid.

Tower, the control. Score is `max(z)`. A column straight up doesn't move the
centre of mass at all, so stacking is always legal and greedy works. Ceiling is
`nz - 1`, also 11, so both tasks now span the same range.

Same environment, same agents, same budget. Only the scoring changes.

### Agents

`Agent` is an MLP, `cells -> 128 -> cells`, where cells is `nx * ny * nz`. It
maps the grid to a score per cell. Illegal cells get masked to -1e9 and the
argmax gets placed. On the current grid that's 2,304 cells and about 590K
parameters.

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

M0 through M2 done. M3 in progress: the grid was rebuilt and the H1 numbers are
being regenerated on it.

- `src/env.py`, grid, base, support, balance, ground rule, `valid_move()`. Takes
  a shape tuple so the grid can be non-cubic. Also `best_possible_bridge(steps)`,
  which builds the optimal alternating cantilever using `valid_move()` as the
  judge, so whatever it returns is provably legal. On the current grid it reaches
  the ceiling of 11 using 23 of 40 blocks.
- `src/agent.py`, `Agent` and `RandomAgent`, plus `growth(sigma)` for mutation.
- `src/fitness.py`, `tower_score` and `bridge_score`.
- `src/evolution.py`, the loop. Takes `score_func`, `agent_class`, `seed`. Logs
  the three series plus `best_grid`.
- `src/visu.py`, `render(grid)`, draws a structure as outlined cubes.
- `src/main.py`, runs one experiment, renders the best structure.

Not built yet: novelty search, the archive, run logging, configs, tests, paper.

## Results so far

Current grid 24 x 8 x 12, base at x=12, 40-block budget, population 50,
sigma=0.002, seed 0.

The reference reaches the ceiling of 11 using 23 of the 40 blocks, so the budget
isn't the constraint and a plateau below 11 can't be blamed on running out of
blocks.

| arm | population mean | best over 2,500 structures |
|-----|-----------------|----------------------------|
| random | 2.8 | 7 |
| objective (50 gens) | 6.3 and still rising | 8 |
| reference | | 11 |

The objective arm's `gen_best` went 5 at generation 0, then 6, 7, and 8 by
generation 32. That matters more than the number itself: it is the first time in
this project that evolution produced anything better than its own initial random
draw. A 200-generation run is going to find where it actually plateaus, since 50
generations stopped while the mean was still climbing.

The structures look right now too. The best one was 41 blocks with x running 8 to
20: a dense mass of 28 blocks around and behind the base, and a thin 13-block arm
reaching out. That's an actual cantilever. `mean(x)` was 12.27 against a limit of
12.5, so only 0.23 of the balance allowance went unused.

Also worth keeping for the writeup: best-ever separates the two arms by one point
(8 against 7) while the mean separates them by three and a half (6.3 against 2.8).
The random arm found a 7 purely by drawing 2,500 structures. That's the clearest
demonstration I have that best-ever measures sampling volume rather than search
quality.

### What the old 12 x 12 x 12 grid taught me

Worth recording because it nearly sank the project and the diagnosis took a while.

On the cubic grid the bridge ceiling was 5, and a freshly initialized network
could reach it. Seed 0 started with a best of 4 and finished at 4. Seed 1 started
with a best of 5 and finished at 5. In both cases the final best equalled
generation 0's best, so across 5,000 structures per run not one mutated child
ever beat the initial draw.

The sigma = 0 control confirmed it. With mutation switched off entirely the mean
converged to exactly generation 0's best, 4.00 on seed 0 and 5.00 on seed 1.
Turning mutation back on made the mean slightly worse, 3.96 and 4.90, because
children drift off the optimum. So mutation was not just useless, it was mildly
harmful, and the whole thing was a max-finder over the initial population.

The cause was the coarse objective. Six possible scores meant the population
collapsed onto one value within about 25 generations, after which selection was
ranking tied agents by list index. Widening x fixed it by giving the score twelve
possible values and pushing the ceiling well above what a lucky initialization
reaches.

## Things I learned that cost time

sigma = 0.002. `nn.Linear` inits weights at std ~0.014 and ~0.05 for the two
layers, so sigma=0.1 shifts a weight by 7x its own size and every child is a
brand new random network. If a fitness curve goes flat, check sigma against
weight scale first.

Best-ever is a bad way to compare arms. It mostly measures how much you sampled.
Use `gen_mean`.

If the final best equals generation 0's best, the search is doing nothing and
you are looking at selection over the initial draw. The sigma = 0 control is the
cheap way to prove it either way.

Headroom matters more than step size. When the ceiling sits close to what random
initialization reaches, no amount of sigma tuning helps, because the population
ties on one score and selection stops discriminating.

The deception isn't "refuses the enabling move". Objective search places
counterweights freely. It fails on efficiency, spending blocks that buy no reach.

Masking removes the textbook deception. Agents can't make illegal moves, so if I
ever redesign this, the deception has to live somewhere an agent can actually go
wrong.

Cutting the block budget is the wrong way to make the task harder. It makes
agents fail by starvation rather than by deception, and it destroys the reference
solution's argument, since the reference needs slack to prove a better structure
was affordable.

## Next

1. Finish the 200-generation objective run on seed 0 and find where it plateaus.
   That number against the reference's 11 is the H1 test.
2. Repeat on seeds 1 and 2.
3. Then the infrastructure below, before novelty search.

Infrastructure, before M4:

- [ ] Optimize `valid_move()`. It scans all `nx*ny*nz` cells per placement and is
  most of the runtime. Support means only cells next to an existing block can
  ever be legal, so the real candidate set is bounded by structure size, a couple
  of hundred at B=40, not grid size, 2,304 now. Without this the full sweep is
  100 agents x 200 generations x 40 steps x 80 runs, roughly 85 hours. With it,
  closer to 8. This is now the thing gating M5.
- [ ] Run logging. Runs currently vanish on exit. Write seed, arm, task, config,
  the three series, and block placements to `runs/`. Log placements rather than
  final grids, the ordered list rebuilds the grid and also drives build playback
  later. `tee` to a file is the stopgap.
- [ ] Config-driven runs. One YAML per condition so the sweep is a script instead
  of 80 hand-edits of `main.py`.
- [ ] Some pytest cases. Env invariants, plus "greedy +x becomes illegal without
  a counterweight" as the deception mechanism written as a test.

## Milestones

- [x] M0, environment rules. Tests still missing.
- [x] M1, one agent builds, viewer shows it.
- [x] M2, evolution works. Objective arm's mean is more than double the random
  baseline's, and mutation now improves on its initial draw.
- [ ] M3, bridge task and the H1 check. Reference built and verified at 11. The
  objective arm's plateau is being measured on the new grid.
- [ ] M4, novelty search. BC vector, archive, k-NN novelty, the novelty and
  blend arms, coverage metric. Two things to watch: `score_func(grid)` can't
  express novelty, which needs the whole population plus archive, so the scoring
  path has to be restructured to score a generation at a time. And the blend arm
  mixes an integer 0 to 11 with a Euclidean distance around 0 to 2, so both need
  normalising or the blend is just the objective.
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
- Non-cubic grid, 24 x 8 x 12. Bridge ceiling 11, tower ceiling 11, so both
  tasks span the same range. Cheaper than going cubic to N=24, which would have
  cost eight times the compute for the same bridge ceiling.
- Block budget stays generous at 40. Tightening it would make agents fail by
  starvation rather than deception.

Still open:

- Rename `Agent`. It's vague, but `ObjectiveAgent` would be wrong since novelty
  and blend use the same class. `NetworkAgent` or `MLPAgent` names what it is
  instead of which arm uses it. Arm names belong in run configs.
- Tie-breaking in selection. Scores are integers, so mutation gets no partial
  credit for progress between steps. A continuous secondary term, `mean(x)` for
  bridge and centre-of-mass height for tower, would give selection something to
  see. Held back for now because the wider grid may have supplied enough headroom
  on its own. Would have to be applied identically to every arm.
- Population size for the real runs. Currently 50, the plan says 100. Bigger
  populations converge faster mostly by containing more good initial draws, so
  it is not free of the confound above.

## Plan for the experiment

2 x 4: {bridge, tower} x {random, objective, novelty, blend}.

Comparisons run within a task, across arms, never between tasks.

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
- Objectives are coarse integers, so mutation gets no partial credit and ties
  between runs are common. Effect sizes reported partly for this reason.
- Best-ever is confounded by sampling volume.
- Outcomes are sensitive to the initial random population, which on the old grid
  decided the run entirely. The wider grid reduces this but does not remove it,
  which is why seeds are reported rather than single runs.
- `valid_move()` dominates runtime.
- `Agent` is deterministic, so one network gives exactly one structure. Makes
  evaluation noise-free but means no exploration within a lifetime.
- Weight-space evolution on ~590K parameters with a population of 50 to 100 is a
  demanding regime, and the random arm exists partly to check whether any arm
  beats chance at all.
- One BC, results may hinge on it. A BC ablation is a stretch goal.

## Running it

```
python src/main.py
```

Runs one experiment and renders the best structure it found. Pipe through `tee`
to keep the output:

```
mkdir -p runs
python src/main.py | tee runs/objective_seed0.txt
```

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
