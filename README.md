# When Does Novelty Beat the Objective?

Testing whether deception decides when novelty search beats objective
optimization.

Small neural nets build 3D voxel structures one block at a time under support
and balance rules. No backprop, the networks are evolved. Four selection
strategies run on two tasks, one deceptive and one not. The prediction is that
novelty wins on the deceptive one and loses on the other.

This file is my working log.

## The question

Is deception the condition under which novelty search beats direct objective
optimization?

Three hypotheses, in the order they have to be established:

H1. On the bridge task, objective search plateaus below what is achievable.
Tested and failed: objective search reached the ceiling. The rules are being
changed in response, see Results. Without H1 the rest is meaningless, since a
low score could just mean the budget ran out.

H2. On the bridge task, novelty search ends up ahead of objective search.
Mann-Whitney U over 10+ seeds, with effect size.

H3. On the tower task, objective search does at least as well as novelty.

H2 alone only says novelty won once. H2 and H3 together say deception is what
decides it, and that can fail in either direction. A null result on H2 is still
worth reporting if H1 holds.

## Setup

### Environment

A voxel grid, currently 24 x 8 x 12. Non-cubic on purpose: the bridge ceiling
depends only on x, while y and z cost compute without raising it. One base cell
is fixed at the centre of the ground plane, at `(nx//2, ny//2, 0)`. Agents place
one block per step up to a budget of B blocks. A placement is legal only if all
three hold:

1. Support. The cell is empty and touches an existing block face-on, or it is
   the base. The structure is one connected object grown from the base.
2. Balance. The centre of mass stays over the base, so `|mean(x) - x_base| <= 0.5`
   and the same for y. It has to hold at every step, so build order matters.
3. Off the ground. `z > 0` unless it is the base, so reaching sideways is a
   cantilever and not a line along the floor.

Illegal placements are masked out of the action space, so an agent cannot make
one. That turned out to be the reason the task was not deceptive: balance guards
rather than traps.

Rule 2 is being moved out of the mask. Unbalanced placements become legal and
fatal: the structure collapses, the block is undone, the episode ends there, and
the score is whatever stood before that placement. See Results.

### The two tasks

Bridge, the deceptive one. Score is `max(x) - x_base`, reach in +x only. One
block past the base uses the half-block of slack, and nothing further out is
legal until ballast goes on the -x side. Ballast never raises the score itself.
Ceiling is `nx - 1 - nx//2`, so 11 here.

Tower, the control. Score is `max(z)`. A column above the base does not move the
centre of mass, so stacking is always legal and greedy works. Ceiling is
`nz - 1`, also 11, so both tasks span the same range.

Same environment, same agents, same budget. Only the scoring changes.

### Agents

`Agent` is an MLP, `cells -> 128 -> cells`, where cells is `nx * ny * nz`. It
maps the grid to a score per cell, illegal cells are masked to -1e9, and the
argmax gets placed. On the current grid that is 2,304 cells and about 590K
parameters.

`RandomAgent` has no policy. It gives uniform random scores to legal cells, so
the argmax is a uniform random legal placement. Floor baseline.

Three of the four arms use the same `Agent` class. What changes between them is
what selection rewards, not what builds.

### The loop

Each generation every agent builds on its own fresh grid and gets scored, the
top half survive, and the next generation is bred from survivors with Gaussian
mutation plus elitism, meaning the best agent is copied over untouched.

### The arms

| arm | agent | selection score |
|-----|-------|-----------------|
| random | `RandomAgent` | none |
| objective | `Agent` | the task objective |
| novelty | `Agent` | mean distance to k nearest neighbours in behaviour space |
| blend | `Agent` | half objective, half novelty |

### Behaviour characterization

Each structure is summarised as 5 numbers:
`[block_count, max_height, max_radial_reach, mean_horizontal_offset, com_height]`,
each normalised to [0,1]. Novelty is the mean k-NN distance (k=15) against the
population plus an archive, and anything novel enough is added to the archive.

`max_height` is the tower objective, so the BC overlaps the objectives. That is
standard: Lehman and Stanley's maze BC was the robot's final position, which
contained their objective. Results are known to hinge on the BC (Pugh et al.),
so it is fixed before any experiments and shared across arms and tasks.

### Metrics

Three series per generation:

- `gen_best`, best score that generation
- `best_so_far`, cumulative max
- `gen_mean`, population mean

Compare arms on `gen_mean`. Best-ever measures sampling volume, not search
quality, see Results.

## Where I am

M0 to M2 done. M3 failed and the rules are being reworked.

- `src/env.py`, grid, base, support, balance, ground rule, `valid_move()`. Takes
  a shape tuple so the grid can be non-cubic. Also `best_possible_bridge(steps)`,
  which builds the optimal alternating cantilever using `valid_move()` as the
  judge, so what it returns is provably legal. Reaches 11 using 23 of 40 blocks.
- `src/agent.py`, `Agent` and `RandomAgent`, plus `growth(sigma)` for mutation.
- `src/fitness.py`, `tower_score` and `bridge_score`.
- `src/evolution.py`, the loop. Takes `score_func`, `agent_class`, `seed`. Logs
  the three series plus `best_grid`.
- `src/visu.py`, `render(grid)`, structures as outlined cubes.
- `src/main.py`, runs one experiment, renders the best structure.

Not built: novelty search, the archive, run logging, configs, tests, paper.

## Results so far

Grid 24 x 8 x 12, base at x=12, 40-block budget, population 50, sigma=0.002,
seed 0. All of this is under the old rules and will need redoing.

The reference reaches 11 using 23 of the 40 blocks, so the budget is not the
constraint and a plateau below 11 cannot be blamed on running out of blocks.

| arm | population mean | best found | structures drawn |
|-----|-----------------|------------|------------------|
| random (200 gens) | 2.80 | 7 | 10,000 |
| objective (200 gens) | 9.47 | 11 | 10,000 |
| reference | | 11 | built by hand |

Random hit 7 at generation 13 and never beat it across the remaining 9,350
structures, mean pinned at 2.80 throughout (2.82, 2.81, 2.78, 2.80 by
50-generation block). So 7 is a ceiling for chance, not a lucky sample, and
anything above it is beyond what sampling produces.

### The objective arm solves it

Over 200 generations on seed 0 it reached 11, the same as the reference.

```
best_so_far: 5 -> 6 (gen 1) -> 7 (10) -> 8 (32) -> 9 (58) -> 10 (105) -> 11 (107)
gen_mean:    4.80, 5.63, 6.38, 7.06, 7.38, 8.34, 9.30, 9.47   (25-gen blocks)
```

Six improvements over 107 generations, mean climbing behind the leader. The best
structure was 41 blocks spanning x = 5 to 23, 19 behind the base and 15 ahead,
`mean(x)` at 12.12 against a limit of 12.5. It rediscovered the reference.

So the search works. On the old 12-cubed grid it was only sorting its initial
draw; here it climbs from 5 to the optimum.

And H1 fails. Objective search does not plateau below what is achievable, it
reaches it, so the bridge task as built is not deceptive. That makes H2 and H3
unanswerable as written, since novelty cannot beat an arm that is already at the
ceiling.

### Why it is not deceptive, and what I am changing

Illegal placements are masked out, so an agent cannot over-extend and topple.
Balance steers agents toward the counterweights the optimum needs instead of
trapping them, the landscape is smooth, and objective search walks up it.

Deception needs the wrong move to be available and tempting. So balance moves out
of the mask and becomes a failure condition:

- `valid_move()` keeps only support and the ground rule, so extending greedily in
  +x becomes a legal choice
- after each placement the structure is checked, and if the centre of mass has
  left the base it collapses, the block is undone, and the episode ends there
- the score is the structure as it stood before the fatal placement, and the rest
  of the block budget is lost

This is within a single build, not across generations. An agent that topples on
its fourth placement scores whatever three blocks got it and forfeits the other
36 steps of that episode. Nothing carries over to the next generation beyond the
score.

A greedy agent topples on its fourth block at reach 1 and loses its other 36
placements. A careful one reaches 11. Reaching further always means risking a
block that might topple you, and a failed attempt costs the rest of the run, so
timid structures become a local optimum that only risk escapes.

Scoring a collapse as zero was the other option. Rejected because nearly every
agent in an initial population would topple, everyone would score 0, and
selection would have nothing to discriminate on.

### What the old 12 x 12 x 12 grid taught me

On the cubic grid the bridge ceiling was 5, and a freshly initialized network
could reach it. Seed 0 started with a best of 4 and finished at 4. Seed 1 started
at 5 and finished at 5. In both cases the final best equalled generation 0's
best, so across 5,000 structures per run no mutated child ever beat the initial
draw.

The sigma = 0 control confirmed it. With mutation off the mean converged to
exactly generation 0's best, 4.00 on seed 0 and 5.00 on seed 1. Turning mutation
back on made the mean slightly worse, 3.96 and 4.90, since children drift off the
optimum. Mutation was not useless but harmful, and the whole thing was a
max-finder over the initial population.

The cause was the coarse objective. Six possible scores meant the population
collapsed onto one value within about 25 generations, after which selection was
ranking tied agents by list index. Widening x fixed it: twelve possible values,
and a ceiling well above what a lucky initialization reaches.

## Things that cost time

sigma = 0.002. `nn.Linear` inits weights at std ~0.014 and ~0.05 for the two
layers, so sigma=0.1 shifts a weight by 7x its own size and every child is a new
random network. If a fitness curve goes flat, check sigma against weight scale
first.

Best-ever is a bad way to compare arms, it mostly measures how much you sampled.
Use `gen_mean`.

If the final best equals generation 0's best, the search is doing nothing and you
are looking at selection over the initial draw. The sigma = 0 control proves it
either way.

Headroom matters more than step size. When the ceiling sits close to what random
initialization reaches, no amount of sigma tuning helps, because the population
ties on one score and selection stops discriminating.

The deception is not "refuses the enabling move". Objective search places
counterweights freely. It failed on efficiency, and once given headroom it did
not fail at all.

Masking removes deception, and that was measured rather than guessed. Agents
cannot make illegal moves, so balance steers instead of trapping. Deception has
to live somewhere an agent can go wrong.

Cutting the block budget is the wrong way to make a task harder. Agents then fail
by starvation rather than deception, and it destroys the reference solution's
argument, which needs slack to prove a better structure was affordable.

## Next

Reworking balance from a mask into a failure condition:

1. Take balance out of `valid_move()`, leaving support and the ground rule. Also
   makes it cheaper.
2. Add a check on `Environment` for whether the current structure stands. Same
   arithmetic, asked of the grid as it is rather than of a candidate cell.
3. Have `place_block` report whether the structure survived, undoing the block
   and returning False if not, so the grid always holds the last stable state.
4. Break the build loop in `evolution.py` when a placement collapses.
5. Rework `best_possible_bridge`, which leans on `valid_move()` to refuse
   unbalanced cells.
6. Re-measure the reference, the random floor, and the objective arm, then
   retest H1.

Infrastructure, before M4:

- [ ] Optimize `valid_move()`. It scans all `nx*ny*nz` cells per placement and is
  most of the runtime. Only cells next to an existing block can ever be legal, so
  the candidate set is bounded by structure size, a couple of hundred at B=40,
  not grid size, 2,304 now. Without it the full sweep is 100 agents x 200
  generations x 40 steps x 80 runs, roughly 85 hours. With it, about 8. This
  gates M5.
- [ ] Run logging. Runs vanish on exit. Write seed, arm, task, config, the three
  series, and block placements to `runs/`. Log placements rather than final
  grids, the ordered list rebuilds the grid and drives build playback later.
  `tee` is the stopgap.
- [ ] Print the config at the top of every run. Four runs have gone wrong from
  the wrong arm, wrong task, or a missing metric, and none of them said what they
  were on their own output.
- [ ] Config-driven runs. One YAML per condition so the sweep is a script instead
  of 80 hand-edits of `main.py`.
- [ ] Some pytest cases. Env invariants, plus a test that greedy +x extension
  topples the structure without a counterweight.

## Milestones

- [x] M0, environment rules. Tests still missing.
- [x] M1, one agent builds, viewer shows it.
- [x] M2, evolution works. Objective arm climbs 5 to 11 while the random floor
  sits at 2.80, so the search beats chance by a wide margin.
- [ ] M3, bridge task and the H1 check. Reference verified at 11, but H1 failed
  because objective search reached 11 too. Balance is being reworked from a mask
  into a failure condition, then everything needs re-measuring.
- [ ] M4, novelty search. BC vector, archive, k-NN novelty, the novelty and blend
  arms, coverage metric. Two traps: `score_func(grid)` cannot express novelty,
  which needs the whole population plus archive, so the scoring path has to score
  a generation at a time. And the blend arm mixes an integer 0 to 11 with a
  Euclidean distance around 0 to 2, so both need normalising or the blend is just
  the objective.
- [ ] M5, the full 2x4 with 10+ seeds, Mann-Whitney and Cliff's delta,
  Holm-corrected, and the headline figure.
- [ ] M6, explorer. Build playback from logged placements, generation slider,
  side-by-side arms, BC scatter, GIF export.
- [ ] M7, paper in LaTeX plus a runs-to-figures script.

## Decisions

Settled:

- Ground-level extension removed. `valid_move()` needs `z > 0` except for the
  base, so the bridge task builds a cantilever.
- The deceptive task is called `bridge` in code. It is a cantilever in the
  engineering sense, which the paper should say once.
- Non-cubic grid, 24 x 8 x 12. Both ceilings 11. Cheaper than cubic N=24, which
  would cost eight times the compute for the same bridge ceiling.
- Block budget stays at 40. Tightening it makes agents fail by starvation.
- Balance becomes a failure condition rather than a mask. Unbalanced placements
  are legal, but making one collapses the structure: the episode ends there, the
  offending block is undone, and the score is the structure as it stood before
  that placement. The remaining budget is lost. Under the old rules objective
  search reached the ceiling, which proved the task was not deceptive, and the
  reason was that masking made the greedy move impossible rather than costly.

Still open:

- Rename `Agent`. `ObjectiveAgent` would be wrong since novelty and blend use the
  same class. `NetworkAgent` or `MLPAgent` names what it is instead of which arm
  uses it. Arm names belong in run configs.
- Tie-breaking in selection. Scores are integers, so mutation gets no partial
  credit between steps. A continuous secondary term, `mean(x)` for bridge and
  centre-of-mass height for tower, would give selection something to see. Held
  back while the new rules are tested. Would apply identically to every arm.
- Population size for the real runs. Currently 50, the plan says 100. Bigger
  populations converge faster partly by containing more good initial draws.

## Plan for the experiment

2 x 4: {bridge, tower} x {random, objective, novelty, blend}.

Comparisons run within a task, across arms, never between tasks.

- 10+ seeds per condition, so 80+ runs, everything derived from one logged seed
- population 100, ~200 generations, sigma fixed and identical across arms
- hyperparameters tuned on tower only, never on bridge, so nothing favours an arm
  on the task that decides H2
- primary metric is population mean per generation, best-so-far alongside
- secondary: behaviour-space coverage and population BC variance, the mechanism
  evidence that novelty spreads where objective clusters
- Mann-Whitney U on the final objective, Holm-corrected, with Cliff's delta
- plots show median and IQR over seeds, not single runs
- every run saves config, seed, series and placements under `runs/`, and every
  figure regenerates from there with one script

One ablation planned: archive on/off for the novelty arm on bridge.

## Known limitations

- No physics. Balance is a centre-of-mass rule, which is what makes the reference
  solution constructible by hand.
- Objectives are coarse integers, so mutation gets no partial credit and ties
  between runs are common. Effect sizes reported partly for this reason.
- Best-ever is confounded by sampling volume.
- Outcomes are sensitive to the initial population, which on the old grid decided
  the run entirely. The wider grid reduces this but does not remove it, so seeds
  are reported rather than single runs.
- `valid_move()` dominates runtime.
- `Agent` is deterministic, so one network gives one structure. Evaluation is
  noise-free but there is no exploration within a lifetime.
- Weight-space evolution on ~590K parameters with a population of 50 to 100 is
  demanding, and the random arm exists partly to check whether any arm beats
  chance.
- One BC, results may hinge on it. A BC ablation is a stretch goal.

## Running it

```
python src/main.py
```

Runs one experiment and renders the best structure. Pipe through `tee` to keep
the output, and use `-u` so it is written as it goes rather than buffered:

```
mkdir -p runs
python3 -u src/main.py | tee runs/obj_seed0.txt
```

Note that `render()` blocks until the window is closed, so a buffered run will
not flush its output until then.

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
