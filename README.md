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
Holds under the current rules: it stalls at 6 against an achievable 11. It failed
under the old rules, which is why balance was reworked. See Results. Without H1
the rest is meaningless, since a low score could just mean the budget ran out.

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
one block per step up to a budget of B blocks. Two rules decide what is legal:

1. Support. The cell is empty and touches an existing block face-on, or it is
   the base. The structure is one connected object grown from the base.
2. Off the ground. `z > 0` unless it is the base, so reaching sideways is a
   cantilever and not a line along the floor.

Illegal placements are masked out of the action space, so an agent cannot make
one.

Balance is not a legality rule, it is a failure condition. The centre of mass
has to stay over the base, `|mean(x) - x_base| <= 0.5` and the same for y, and
it is checked after every placement. A placement that breaks it is legal and
fatal: the structure collapses, the block is undone, the episode ends there, and
the agent scores whatever stood before that placement, forfeiting the rest of
its budget.

It used to be part of the mask, which is why the task was not deceptive. Balance
guarded instead of trapping. See Results.

### The two tasks

Bridge, the deceptive one. Score is `max(x) - x_base`, reach in +x only. One
block past the base uses the half-block of slack, and nothing further out is
legal until a counterweight goes on the -x side. Counterweights never raise the
score themselves.
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

`bc(grid, budget)` in `fitness.py` summarises a structure as 5 numbers, each on
[0, 1]:

| | what it measures | denominator |
|---|---|---|
| `block_count` | how much it built | `budget + 1`, the base included |
| `max_height` | how tall | `nz - 1` |
| `com_height` | where the mass sits vertically | `nz - 1` |
| `max_reach` | how far the furthest block is from the base | corner distance |
| `com_reach` | how far the average block is | corner distance |

The two reach dimensions come from `np.hypot(x - x_base, y - y_base)`, one
distance per block. Horizontal only, since height is already covered, and
unsigned, so a structure stretching into -x counts as reaching far even though it
scores nothing. `max_reach` and `com_reach` together separate a thin arm poking
out of a compact body from a structure with mass spread all the way out, which is
exactly how the reference differs from the agents.

Normalisation divides by what is achievable, not what is conceivable.
Dividing `block_count` by the 2,304 grid cells would put every structure in
[0, 0.018] and that dimension would contribute nothing to any distance. For the
same reason mean x is not usable: balance pins it to within 0.5 of the base, so
it would be near constant across every structure in the experiment.

`max_height` is the tower objective, so the BC overlaps the objectives. That is
standard: Lehman and Stanley's maze BC was the robot's final position, which
contained their objective. Results are known to hinge on the BC (Pugh et al.),
so it is fixed before any experiments and shared across arms and tasks.

Sanity checks: a column straight up from the base gives `max_reach` and
`com_reach` of 0 and `max_height` of 1. A block in the far grid corner gives
`max_reach` of exactly 1. A tower and the reference cantilever sit 1.41 apart in
a space whose maximum distance is 2.24.

### The archive

Novelty is the mean k-NN distance (k=15) against the current population plus an
archive of BC vectors from earlier structures. The archive stores behaviours,
not scores. It has no idea whether a structure was good. Its job is to mark
territory as visited so that going back there earns nothing, which is what stops
the search cycling: explore A, move to B, forget A, rediscover A.

Entries are added by a dynamic threshold, following Lehman and Stanley. An
individual enters if its novelty exceeds `rho_min`, and `rho_min` adjusts itself:
raise it if too many were added over a window, lower it if none were added for
several generations. Some implementations also add stochastically at around 2% as
a floor.

The threshold being dynamic is the point. A fixed one has to stay right for 200
generations while the population and the archive both change underneath it. Too
low and the archive balloons until everything is near something and novelty
collapses; too high and it stays empty and there is no memory at all.

Because the BC is normalised, `rho_min` is scale-free and interpretable. The
maximum distance in the space is sqrt(5) = 2.24, so a threshold of 0.1 means
roughly a tenth of one dimension's range, and that meaning does not drift.

Gomes, Mariano and Christensen (2015) tested archive strategies systematically
and found novelty search is somewhat robust to how entries are added and removed
and to the choice of k, though both still affect performance. So this is worth
getting reasonable rather than optimal. The planned ablation is archive on/off,
not threshold tuning.

### Metrics

Three series per generation:

- `gen_best`, best score that generation
- `best_so_far`, cumulative max
- `gen_mean`, population mean

Compare arms on `gen_mean`. Best-ever measures sampling volume, not search
quality, see Results.

## Implemented

- `env.py` : grid takes a shape tuple so it can be non-cubic. `valid_move()`
  checks support and the ground rule. `is_stable()` checks the centre of mass.
  `place_block()` undoes the block and returns False if the structure topples.
  `best_possible_bridge()` builds the reference, 11 using 23 of 40 blocks.
- `agent.py` : `Agent` (MLP), `RandomAgent`, `growth(sigma)` for mutation.
- `fitness.py` : `tower_score`, `bridge_score`, `bc(grid, budget)`.
- `evolution.py` : the loop, seeding, `score_func` and `agent_class` parameters,
  logs `gen_best`, `best_so_far`, `gen_mean`, `best_grid`.
- `visu.py` : `render(grid)`, glyph mesh coloured by position relative to the base.
- `main.py` : runs one experiment, renders the result.

## Results

Grid 24 x 8 x 12, base at x=12, 40-block budget, population 50, sigma=0.002,
seed 0, 200 generations.

The reference reaches 11 using 23 of the 40 blocks, so the budget is not the
constraint and a plateau below 11 cannot be blamed on running out of blocks.

| | population mean | best-ever |
|---|---|---|
| fresh agents, no evolution | 0.58 | 3 |
| random arm | 1.05 | 6 |
| objective arm | 4.0 | 6 |
| reference | | 11 |

The objective arm learns and then stops:

```
best_so_far: 3 -> 4 (gen 23) -> 5 (gen 27) -> 6 (gen 87), then flat for 113 gens
gen_mean:    2.39, 3.05, 3.88, 4.17, 4.24, 3.84, 4.02, 3.99   (25-gen blocks)
```

It plateaus at 6 against an achievable 11. H1 holds.

### Both arms stop at 6

The random arm's mean is flat at 1.05 across all 200 generations, and it also
reaches 6, at generation 16, never beaten in the next 184.

So best-ever does not separate the two arms at all. The mean separates them
nearly four to one. Logging only best-ever, which is what I was doing a week ago,
would have said the objective arm learns nothing.

### Seed 1 agrees

| | last improvement | plateau | gen_mean | x span | mean_x |
|---|---|---|---|---|---|
| seed 0 | gen 87 | 6 | 4.0 | 9..18 | 12.50 |
| seed 1 | gen 86 | 5 | 3.51 | 9..17 | 12.50 |
| random | gen 16 | 6 | 1.05 | 9..18 | 12.41 |
| reference | | 11 | | 2..23 | 12.48 |

Both seeds stop improving at generation 86 or 87 and flatten for the next 114.
Both best structures start at x=9 and sit at exactly the stability limit.

### The mechanism: how far back the counterweights go

Leaning to the limit is not what separates them. The reference is at 12.48, also
at the limit. What differs is where the blocks behind the base sit:

```
reference:  blocks behind base spread over x = 2..11,   mean 6.5,   5.5 from base
seed 1:     blocks behind base crammed into x = 9..11,  mean 10.2,  1.8 from base
```

A block at x=2 pulls the centre of mass back three times harder than one at x=11,
because it is three times further from the base. The reference puts its
counterweights far back where each one counts. The agents stack theirs right next
to the base, where each block barely moves the centre of mass, and then run out
of budget.

So the objective-neutral move they never find is not "add a counterweight", it is
"add a counterweight far behind the base". That is the placement that looks least
useful of all: x=2 is as far from raising the score as a block can get, and it is
the most valuable one to place.

Prediction for H2: if novelty search crosses 6, check whether its counterweights
reach back past x=9.

### Earlier setups, and why they failed

Two dead versions, both worth remembering.

A 12 x 12 x 12 cubic grid, bridge ceiling 5. A freshly initialized network could
reach 5, so there was no headroom. In every run the final best equalled
generation 0's best: across 5,000 structures no mutated child ever beat the
initial draw. The sigma = 0 control confirmed it, with mutation off the mean
converged to exactly generation 0's best and turning it back on made the mean
slightly worse. The whole thing was a max-finder over the initial population, and
the cause was six possible scores: the population collapsed onto one value within
25 generations and selection was then ranking ties by list index.

The current grid with balance as part of the action mask. Widening x to 24 fixed
the headroom problem and the search started working, climbing 5 to 11 over 107
generations. But it reached 11, the ceiling, so the task was not deceptive.
Masking meant an agent could not over-extend and topple, so balance steered
agents toward the counterweights the optimum needs instead of trapping them. That
is why balance became a failure condition.

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

## To do

- archive: store BCs above a novelty threshold, k-NN queries
- k-NN novelty scoring
- restructure the scoring path to score a generation at a time, `score_func(grid)`
  cannot express novelty
- novelty and blend arms. blend mixes an integer 0 to 11 with a distance around
  0 to 2, so both need normalising or it is just the objective
- coverage metric
- run logging to `runs/`: seed, arm, task, config, the three series, block
  placements. log placements not grids, the order rebuilds the grid and drives
  playback later
- print the config at the top of every run. four runs have gone wrong from the
  wrong arm, wrong task, or a missing metric
- config files, one per condition
- optimise `valid_move()`. only cells next to a filled one can be legal, so the
  candidate set is bounded by structure size not grid size. the sweep is 85 hours
  without it, about 8 with
- pytest: env invariants, greedy +x topples without a counterweight
- the 2x4 experiment, 10+ seeds, Mann-Whitney and Cliff's delta, Holm-corrected
- figures: H1 plot, build playback gif, offscreen with a fixed camera
- the paper, LaTeX, plus a runs-to-figures script

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
- A collapse is not modelled as one. The offending block is undone and the
  structure that stood is kept and scored, so a toppled agent leaves a neat stub
  rather than rubble. Physically wrong, but scoring collapses as zero would leave
  49% of an initial population at zero with nothing for selection to rank.
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

Python 3.9+, PyTorch, NumPy, PyVista, Matplotlib, SciPy. No Qt: offscreen
rendering and GIF export are plain PyVista plus imageio.

## References

- Lehman & Stanley (2011). Abandoning Objectives: Evolution Through the Search
  for Novelty Alone. Evolutionary Computation 19(2).
- Goldberg (1987). Simple genetic algorithms and the minimal, deceptive problem.
- Mouret & Clune (2015). Illuminating search spaces by mapping elites.
  arXiv:1504.04909.
- Pugh, Soros & Stanley (2016). Quality Diversity: A New Frontier for
  Evolutionary Computation. Frontiers in Robotics and AI.
- Stanley & Lehman (2015). Why Greatness Cannot Be Planned. Springer.
- Gomes, Mariano & Christensen (2015). Devising Effective Novelty Search
  Algorithms: A Comprehensive Empirical Study. GECCO.
- Kistemaker & Whiteson (2011). Critical Factors in the Performance of Novelty
  Search. GECCO.

## Stretch goals

- MAP-Elites arm, keeping the best structure per BC cell.
- BC ablation, rerunning H2 with a deliberately poor BC, `[block_count]` only.
- Cluster each arm's final structures and characterise build-order patterns.
