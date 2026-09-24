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

The grid is 24 x 8 x 12. It is not a cube because the bridge ceiling only depends
on the x dimension, so there is no point paying for extra y and z. The base block
sits in the middle of the floor and each agent gets 40 blocks.

A block can only go in an empty cell touching an existing block, and it has to be
off the ground unless it is the base. Anything else is masked out, so the agent
never picks it.

Balance works differently. The centre of mass has to stay within half a block of
the base, checked after every placement. Breaking it is allowed but fatal: the
block is removed, the build stops there, and the agent is scored on what was
standing before, losing the rest of its blocks.

Bridge is the deceptive task, scored on how far the structure reaches in +x past
the base. Tower is the control, scored on height. Both top out at 11. Building
straight up never moves the centre of mass, so on the tower task greedy works.

Same environment and budget for both, only the scoring changes.

### Agents and arms

The agent is a small MLP that takes the whole grid and gives a score to every
cell. Illegal cells are masked out and it places the highest scoring one. There
is also a random agent that places uniformly at random among the legal cells,
which is the floor baseline.

The four arms are random, objective (selects on the task score), novelty
(selects on how different a structure is from the others), and blend (half and
half). The last three all use the same network, only what selection rewards is
different.

Each generation every agent builds on a fresh grid, the top half survive, and the
next generation is bred from them with Gaussian mutation, keeping the best one
unchanged.

### Behaviour characterization

Each structure is summarised as five numbers between 0 and 1: how many blocks it
used, how tall it is, how high its mass sits, how far its furthest block is from
the base, and how far its average block is from the base.

The two distance ones are measured flat across the floor and ignore direction,
so reaching backwards counts as reaching.

Each number is divided by the largest value it can actually reach, not the
largest imaginable. Dividing block count by the 2,304 cells in the grid would
squash every structure into almost nothing. The average x position is not used
at all, because balance keeps it pinned near the base for every structure.

Sanity checks: a straight column scores 0 on both distance numbers and 1 on
height, a block in the far corner scores exactly 1 on furthest distance, and a
tower and the reference bridge come out 1.41 apart out of a possible 2.24.

Height is the tower score, so the summary overlaps the objectives. That is normal,
Lehman and Stanley's maze summary contained their objective too. It is fixed
before any experiments and the same for every arm and task.

### Archive

Novelty is the average distance from a structure to its 15 nearest neighbours,
counting both the current population and an archive of past structures. The
archive only remembers what has been built, not whether it was any good. That way
going back to somewhere already visited earns nothing, and the search cannot keep
circling.

A structure gets into the archive if its novelty is above a threshold, which
Lehman and Stanley call rho min. The threshold adjusts itself: it goes up if too
many get in, and down if nothing has got in for several generations. A fixed one
would have to stay right for 200 generations while everything around it changes.

Because the summary is normalised, the threshold means the same thing throughout.
The largest possible distance is 2.24, so 0.1 is roughly a tenth of one number's
range.

Gomes et al. (2015) found novelty search is fairly robust to how the archive is
managed and to k, though both still matter.

### Scores

Every agent gets two numbers. Its reach score is what always gets recorded, and
it is how arms are compared. Its selection score is what decides who survives.
For the objective arm these are the same. For the novelty arm the selection score
is novelty, but reach is still what gets recorded, so all arms are measured on
the same thing.

Arms are compared on the population mean, not the best ever, because the best
ever mostly measures how many structures were tried.

## Built so far

The environment with the support, ground and balance rules, plus a hand built
reference bridge that reaches 11 using 23 of the 40 blocks. The network agent and
the random agent. Tower and bridge scoring, the behaviour summary, and novelty.
The archive. The evolution loop with seeding and a choice of arm. A renderer that
colours blocks by whether they sit behind the base, over it, or out in front. The
run settings are written once, printed, and then used, so a log always says what
produced it.

## Results

All runs here: 24 x 8 x 12 grid, 40 blocks, population 50, sigma 0.002, 200
generations.

The reference reaches 11 using 23 of the 40 blocks, so the budget is not what
stops anyone.

| | population mean | best ever |
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

The best ever does not separate the arms, both hit 6. The mean separates them
four to one. The random arm hit 6 at generation 16 and never beat it.

### The mechanism

| | plateau | mean | x span | mean x |
|---|---|---|---|---|
| seed 0 | 6 | 4.0 | 9 to 18 | 12.50 |
| seed 1 | 5 | 3.51 | 9 to 17 | 12.50 |
| random | 6 | 1.05 | 9 to 18 | 12.41 |
| reference | 11 | | 2 to 23 | 12.48 |

Everyone leans right up to the balance limit, the reference included. The
difference is where the blocks behind the base go. The reference spreads them
from x = 2 to 11, on average 5.5 blocks behind the base. The agents cram them
into x = 9 to 11, on average 1.8 behind.

A block at x = 2 pulls the centre of mass back three times harder than one at
x = 11. The agents put their counterweights next to the base where each one
barely helps, and run out of blocks.

So the move they never find is not adding a counterweight, it is adding one far
behind the base, which looks like the least useful placement of all.

For H2: if novelty gets past 6, check whether its counterweights reach back past
x = 9.

### Earlier setups

A 12 x 12 x 12 cube with a bridge ceiling of 5. A fresh network could already
reach 5, so there was no room to improve, and the final best always equalled
generation 0's best. Running with no mutation at all gave the same result, which
proved mutation was contributing nothing. With only six possible scores the whole
population ended up tied and selection was just picking by list order.

The current grid, but with balance as part of the mask. The search worked and
climbed from 5 to 11, but reaching the ceiling meant the task was not deceptive.
Agents could not over-extend, so balance guided them to the right answer instead
of trapping them.

## Things that cost time

- sigma is 0.002. The starting weights are around 0.014 and 0.05, so sigma 0.1
  moved each weight by 7 times its own size and every child was a new random
  network.
- The best ever mostly measures how much you sampled. Use the mean.
- If the final best equals generation 0's best, the search is doing nothing. A
  run with no mutation proves it either way.
- Room to improve matters more than step size.
- Masking removes deception. The wrong move has to be possible.
- Cutting the budget makes agents fail from lack of blocks, and breaks the
  argument the reference bridge makes.

## To do

- first novelty run on bridge, seed 0, 200 generations. Does it get past 6
- blend arm. Reach runs 0 to 11 and novelty roughly 0 to 2, so both need scaling
  before mixing
- coverage metric
- a short summary printed at the end of each run
- save each run to a file, including the order blocks were placed
- speed up valid move checking by only looking at cells next to existing blocks.
  The full experiment is about 85 hours without it and about 8 with
- tests for the environment rules, including that reaching straight out topples
  without a counterweight
- the full experiment, 2 tasks x 4 arms x 10+ seeds, Mann-Whitney and Cliff's
  delta
- figures: the H1 plot and a gif of a build
- the paper

## Decisions

Settled: no building along the ground, the 24 x 8 x 12 grid with both ceilings at
11, the budget stays at 40, and balance is a failure rather than a mask.

Still open: renaming the agent class to something like NetworkAgent, since three
arms use it. Whether to add a tie break to selection, since whole number scores
give mutation no credit for small progress. Population size, currently 50, the
plan says 100.

## Experiment plan

Bridge and tower, each with random, objective, novelty and blend. Arms are only
compared within a task.

- 10 or more seeds per condition, all from one logged seed
- population 100, about 200 generations, same sigma for every arm
- settings tuned on tower only, never on bridge
- main measure is the population mean, best ever alongside
- also behaviour space coverage and how spread out the population is
- Mann-Whitney U with Holm correction and Cliff's delta
- plots show the median and spread over seeds
- one ablation: novelty on bridge with and without the archive

## Limitations

- No real physics, balance is just a centre of mass check
- A collapse leaves a neat stub instead of rubble. Scoring collapses as zero would
  leave about half of a starting population tied at zero
- Whole number scores, so ties are common
- The best ever is inflated by how much you sample
- Results depend on the starting population
- Checking valid moves is most of the runtime
- The network always makes the same choice for the same grid, so no exploration
  within a single build
- About 590K weights evolved with a population of 50 to 100 is a lot
- One behaviour summary, results may depend on it

## Running it

```
mkdir -p runs
python3 -u src/main.py | tee runs/obj_seed0.txt
```

The -u stops the output being held back until the end. The render window also
blocks until it is closed.

## Stack

Python 3.9, PyTorch, NumPy, PyVista, Matplotlib, SciPy.

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

## Later

- a MAP-Elites arm
- rerun H2 with a poor behaviour summary, block count only
- group each arm's final structures by how they were built
