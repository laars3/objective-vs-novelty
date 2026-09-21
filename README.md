# When Does Novelty Beat the Objective?

Testing whether deception decides when novelty search beats objective
optimization.

Small neural nets build 3D voxel structures one block at a time under support and
balance rules. No backprop, the networks are evolved. Four selection strategies
run on two tasks, one deceptive and one not. The prediction is that novelty wins
on the deceptive one and loses on the other.

Working log.

## Hypotheses

H1. On the bridge task, objective search plateaus below what is achievable.
Holds: it stalls at 6 against an achievable 11.

H2. On the bridge task, novelty search ends up ahead of objective search.

H3. On the tower task, objective search does at least as well as novelty.

H2 alone says novelty won once. H2 and H3 together say deception is what decides
it. A null result on H2 is still worth reporting if H1 holds.

## Setup

Grid 24 x 8 x 12, non-cubic because the bridge ceiling depends only on x. Base
cell fixed at `(nx//2, ny//2, 0)`. Budget 40 blocks.

Legal placements: cell is empty and face-adjacent to an existing block, and
`z > 0` unless it is the base. Illegal ones are masked out of the action space.

Balance is a failure condition, not a legality rule. Centre of mass must stay
within 0.5 of the base in x and y, checked after every placement. Breaking it is
legal and fatal: the block is undone, the episode ends, the agent scores whatever
stood before, and forfeits the rest of its budget.

Bridge, the deceptive one. Score `max(x) - x_base`. Ceiling 11.

Tower, the control. Score `max(z)`. Ceiling 11. A column above the base does not
move the centre of mass, so greedy works.

Same environment and budget for both. Only the scoring changes.

### Agents and arms

`Agent` is an MLP, `cells -> 128 -> cells`. Illegal cells masked to -1e9, argmax
placed. 2,304 cells, ~590K parameters.
`RandomAgent` gives uniform random scores to legal cells. Floor baseline.

| arm | agent | selects on |
|-----|-------|------------|
| random | `RandomAgent` | nothing |
| objective | `Agent` | task objective |
| novelty | `Agent` | k-NN distance in behaviour space |
| blend | `Agent` | half each |

Each generation every agent builds on a fresh grid, top half survive, next
generation bred with Gaussian mutation plus elitism.

### Behaviour characterization

`bc(grid, budget)` in `fitness.py`, 5 numbers each on [0, 1]:

| | measures | denominator |
|---|---|---|
| `block_count` | how much it built | `budget + 1` |
| `max_height` | how tall | `nz - 1` |
| `com_height` | where mass sits vertically | `nz - 1` |
| `max_reach` | furthest block from base | corner distance |
| `com_reach` | average block from base | corner distance |

Reach comes from `np.hypot(x - x_base, y - y_base)`, one per block. Horizontal
only, and unsigned so reaching into -x counts.

Normalise by what is achievable, not conceivable. `block_count` over 2,304 cells
would put everything in [0, 0.018]. Mean x is unusable for the same reason,
balance pins it near the base.

Checks: a column gives `max_reach` and `com_reach` of 0, `max_height` of 1. A
corner block gives `max_reach` of exactly 1. Tower and reference sit 1.41 apart,
out of a maximum 2.24.

`max_height` is the tower objective, so the BC overlaps the objectives. Standard,
Lehman and Stanley's maze BC contained theirs. Fixed before experiments and
shared across arms and tasks.

### Archive

Novelty is the mean k-NN distance (k=15) against the population plus an archive
of past BC vectors. The archive stores behaviours, not scores. It marks territory
as visited so going back earns nothing, which stops the search cycling.

Dynamic threshold, following Lehman and Stanley: enter if novelty exceeds
`rho_min`, raise `rho_min` if too many were added over a window, lower it if none
were added for several generations. A fixed threshold would have to stay right
for 200 generations while the population and archive both change.

Because the BC is normalised, `rho_min` is scale-free. Max distance is 2.24, so
0.1 means about a tenth of one dimension.

Gomes et al. (2015) found novelty search is somewhat robust to archive strategy
and k, though both still matter. Get it reasonable, not optimal.

`novelty()` goes in `fitness.py` (pure function). `Archive` goes in
`evolution.py` (run-scoped state).

### Metrics

`gen_best`, `best_so_far`, `gen_mean`, logged per generation. Compare arms on
`gen_mean`. Best-ever measures sampling volume.

## Implemented

- `env.py` : shape tuple grid, `valid_move()`, `is_stable()`, `place_block()`
  undoes and returns False on collapse, `best_possible_bridge()` reference
- `agent.py` : `Agent`, `RandomAgent`, `growth(sigma)`
- `fitness.py` : `tower_score`, `bridge_score`, `bc(grid, budget)`
- `evolution.py` : the loop, seeding, `score_func` and `agent_class`, logs the
  three series plus `best_grid`
- `visu.py` : `render(grid)`, glyph mesh coloured by position relative to base
- `main.py` : runs one experiment, renders the result

## Results

Grid 24 x 8 x 12, budget 40, population 50, sigma 0.002, 200 generations.

Reference reaches 11 using 23 of 40 blocks, so the budget is not the constraint.

| | population mean | best-ever |
|---|---|---|
| fresh agents | 0.58 | 3 |
| random arm | 1.05 | 6 |
| objective arm | 4.0 | 6 |
| reference | | 11 |

```
seed 0: 3 -> 4 (gen 23) -> 5 (gen 27) -> 6 (gen 87), flat for 113 gens
seed 1: 2 -> 3 (gen 44) -> 4 (gen 79) -> 5 (gen 86), flat for 114 gens
```

H1 holds. Both seeds stop improving at generation 86 or 87.

Best-ever does not separate the arms, both hit 6. The mean separates them four to
one. The random arm hit 6 at generation 16 and never beat it in the next 184.

### The mechanism

| | plateau | gen_mean | x span | mean_x |
|---|---|---|---|---|
| seed 0 | 6 | 4.0 | 9..18 | 12.50 |
| seed 1 | 5 | 3.51 | 9..17 | 12.50 |
| random | 6 | 1.05 | 9..18 | 12.41 |
| reference | 11 | | 2..23 | 12.48 |

Everyone leans to the stability limit, the reference included. What differs is
where the blocks behind the base sit:

```
reference:  x = 2..11,   mean 6.5,   5.5 from base
seed 1:     x = 9..11,   mean 10.2,  1.8 from base
```

A block at x=2 pulls the centre of mass back three times harder than one at x=11.
The agents stack counterweights next to the base where each barely moves the
centre of mass, then run out of budget.

So the move they never find is not "add a counterweight", it is "add one far
behind the base", which is the placement that looks least useful of all.

Prediction for H2: if novelty crosses 6, check whether its counterweights reach
back past x=9.

### Earlier setups

12 x 12 x 12 cubic, bridge ceiling 5. A fresh network could reach 5, so no
headroom. Final best always equalled generation 0's best. The sigma = 0 control
confirmed mutation contributed nothing. Cause: six possible scores, population
collapsed onto one value, selection ranked ties by list index.

Current grid with balance in the mask. Search worked, climbing 5 to 11, but
reaching 11 meant the task was not deceptive. Masking made over-extending
impossible rather than costly, so balance steered instead of trapping.

## Things that cost time

- sigma = 0.002. `nn.Linear` inits at std ~0.014 and ~0.05, so sigma 0.1 is 7x a
  weight's own size and every child is a new random network.
- Best-ever mostly measures how much you sampled. Use `gen_mean`.
- If final best equals generation 0's best, the search is doing nothing. The
  sigma = 0 control proves it either way.
- Headroom matters more than step size.
- Masking removes deception. It has to live somewhere an agent can go wrong.
- Cutting the budget makes agents fail by starvation, and destroys the reference
  solution's argument.

## To do

- `novelty()` in `fitness.py`, `Archive` in `evolution.py`
- restructure the scoring path to score a generation at a time, `score_func(grid)`
  cannot express novelty
- novelty and blend arms. blend mixes an integer 0 to 11 with a distance around
  0 to 2, so both need normalising
- coverage metric
- run logging to `runs/`, placements not grids
- print the config at the top of every run
- config files, one per condition
- optimise `valid_move()`, candidates from neighbours of filled cells. 85 hours
  for the sweep without it, about 8 with
- pytest: env invariants, greedy +x topples without a counterweight
- 2x4 experiment, 10+ seeds, Mann-Whitney and Cliff's delta
- figures: H1 plot, build playback gif
- the paper

## Decisions

Settled:

- Ground-level extension removed, `z > 0` except the base
- Non-cubic grid 24 x 8 x 12, both ceilings 11
- Budget stays at 40
- Balance is a failure condition, not a mask

Open:

- Rename `Agent`. `NetworkAgent` or `MLPAgent`, not `ObjectiveAgent`, since
  novelty and blend use the same class
- Tie-breaking in selection. Integer scores give mutation no partial credit
- Population size, currently 50, plan says 100

## Experiment plan

2 x 4: {bridge, tower} x {random, objective, novelty, blend}. Comparisons within
a task, never between.

- 10+ seeds per condition, everything from one logged seed
- population 100, ~200 generations, sigma identical across arms
- tuned on tower only, never on bridge
- primary metric population mean, best-so-far alongside
- secondary: behaviour-space coverage, population BC variance
- Mann-Whitney U, Holm-corrected, with Cliff's delta
- plots show median and IQR over seeds
- ablation: archive on/off for novelty on bridge

## Limitations

- No physics, balance is a centre-of-mass rule
- A collapse leaves a neat stub rather than rubble. Scoring collapses as zero
  would leave 49% of an initial population tied at zero
- Coarse integer objectives, so ties are common
- Best-ever is confounded by sampling volume
- Outcomes are sensitive to the initial population
- `valid_move()` dominates runtime
- `Agent` is deterministic, no exploration within a lifetime
- ~590K parameters with a population of 50 to 100 is demanding
- One BC, results may hinge on it

## Running it

```
mkdir -p runs
python3 -u src/main.py | tee runs/obj_seed0.txt
```

`-u` because output buffers otherwise, and `render()` blocks until the window is
closed so a buffered run will not flush until then.

## Stack

Python 3.9+, PyTorch, NumPy, PyVista, Matplotlib, SciPy. No Qt, offscreen
rendering and GIF export are plain PyVista plus imageio.

## References

- Lehman & Stanley (2011). Abandoning Objectives. Evolutionary Computation 19(2).
- Goldberg (1987). Simple genetic algorithms and the minimal, deceptive problem.
- Mouret & Clune (2015). Illuminating search spaces by mapping elites.
- Pugh, Soros & Stanley (2016). Quality Diversity. Frontiers in Robotics and AI.
- Stanley & Lehman (2015). Why Greatness Cannot Be Planned.
- Gomes, Mariano & Christensen (2015). Devising Effective Novelty Search
  Algorithms. GECCO.
- Kistemaker & Whiteson (2011). Critical Factors in the Performance of Novelty
  Search. GECCO.

## Stretch

- MAP-Elites arm
- BC ablation with `[block_count]` only
- Cluster each arm's final structures by build-order pattern
