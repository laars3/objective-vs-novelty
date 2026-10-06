# When Does Novelty Beat the Objective?

Small neural networks build 3D structures one block at a time. The networks are
evolved, there is no backprop. The question is whether novelty search beats
plain objective search when the task is deceptive, and loses when it is not.

This is my working log for the project.

## Hypotheses

H1. On the bridge task, objective search stalls below what is possible. Holds,
it stalls around 6 when 11 is possible.

H2. On the bridge task, novelty search ends up ahead of objective search. Not
supported on 30 seeds.

H3. On the tower task, objective search does at least as well as novelty.
Holds.

## Setup

The grid is 24 x 8 x 12, with a base block in the middle of the floor. Each
agent gets 40 blocks. A block has to touch an existing block and sit off the
ground. The centre of mass has to stay within half a block of the base. If a
placement breaks that, the block is removed and the build ends there.

Bridge scores how far the structure reaches in +x past the base. Tower scores
height. Both top out at 11. Building straight up never shifts the centre of mass,
so tower is easy and bridge is the deceptive one.

The agent is an MLP that scores every cell of the grid and places the highest
legal one. A random agent that places anywhere legal is the baseline.

There are four arms. Random, objective (selects on reach), novelty (selects on
how different a structure is from past ones), and blend (half each, both scaled
to 0 to 1 first). Each generation the top half survive and are copied with small
random changes to their weights.

Novelty compares structures using five numbers each: block count, height, height
of the centre of mass, distance of the furthest block from the base, and average
distance from the base. An archive keeps structures that were novel enough, so
revisiting old ones earns nothing.

Arms are compared on the task score, mainly the population mean.

## Results

Bridge task, population 50, 200 generations. Objective, blend and novelty have
30 seeds, random has 10. Each run is saved as json in runs/ and analysis.ipynb
computes everything below. The stats are two-sided permutation tests on Cliff's
delta with 10000 shuffles, which is the same test as Mann-Whitney U.

The hand built reference bridge reaches 11 using 23 of the 40 blocks.

Best reach per seed:

objective 6 5 3 6 7 2 7 6 6 8 5 5 6 3 6 8 2 7 6 5 8 5 5 6 5 4 5 2 6 5, median 5.5
blend 6 5 6 5 7 7 9 8 6 7 7 7 7 5 8 8 7 5 6 6 6 7 7 6 6 5 5 6 6 6, median 6
novelty 6 5 5 5 6 6 6 6 6 5 6 7 5 6 6 6 5 6 6 6 7 6 6 7 5 5 8 5 5 5, median 6
random 6 7 5 6 6 6 5 5 7 5, median 6

Mean reach over the last 50 generations, median across seeds: blend 3.60,
objective 3.59, novelty 1.97, random 1.02.

On best reach blend is ahead of objective, delta 0.38 and p = 0.008, and ahead
of novelty, delta 0.34 and p = 0.018. Novelty and objective are level, p = 0.33,
and so are objective and random, p = 0.51.

On mean reach blend and objective are level, p = 0.93. Both are well above
novelty and random, p < 0.001.

Objective collapses on 6 of 30 seeds and ends at 4 or below. Five of them find
their best in the first 10 generations and never improve. Blend never ends below
5, and reaches 7 or more on 13 seeds against 6 for objective. On 10 seeds blend's
lead on best reach was not significant, p = 0.33.

The reason they stall shows in the structures. They all lean right up to the
balance limit, the reference too. The reference puts its counterweights on
average 5.5 blocks behind the base. Every arm keeps them 1.7 to 1.9 behind. A
block further back pulls the centre of mass back harder, so the agents use up
their blocks on weak counterweights. Placing one far behind the base scores
nothing at the time, which is why no arm finds it.

Every arm makes the same mistake, so the limit is probably the network or the
behaviour summary.

Earlier bridge versions failed in two ways. On a 12 x 12 x 12 grid a random
network could already reach the maximum of 5, so evolution had nothing to
improve. With balance built into the mask, agents could never over-extend, so
the task was not deceptive and objective search reached 11.

Tower task, same settings, 10 seeds per arm.

Best height per seed:

objective 11 11 11 10 11 11 11 10 11 11, median 11
blend 11 11 11 11 11 11 11 11 11 11, median 11
novelty 10 11 11 9 11 10 11 11 11 11, median 11
random 8 9 8 8 8 10 9 8 8 8, median 8

Mean height over the last 50 generations, median across seeds: blend 8.93,
objective 8.66, novelty 5.47, random 2.21.

On best height objective and novelty are level, p = 0.72. On mean height
objective is well above novelty, p < 0.001 and Cliff's delta 0.96. Blend and
objective are level on both. Objective beats random on best height, p < 0.001,
which it does not manage on bridge.

Novelty's mean is about half of objective's on bridge and two thirds on tower.
It is behind on both tasks, so making the task deceptive did not change which
one wins.

## Info Collected

1) sigma has to be small. The starting weights are around 0.01 to 0.05, so a
sigma of 0.1 turned every child into a new random network. 0.002 works.
2) Compare the population mean. The best ever mostly reflects how many
structures were tried.
3) If the final best equals the generation 0 best, the search is doing nothing.
A run with no mutation checks this.
4) The wrong move has to be possible, or the task is not deceptive.
5) A 100 generation run takes 45 seconds now that valid moves are only checked
next to existing blocks.

## To do

1) Split reach in the behaviour summary into reach in front of the base and
reach behind it, then rerun novelty and blend on bridge. The current summary
uses one distance from the base for both sides, so a far counterweight does
not stand out as new. If novelty gets past 6, the summary was the limit. If
not, the network probably is.
2) Then try a blend weight that changes during a run.
3) Decide whether to try a different agent.
4) Save the best structure of each generation, for a gif of a run.
5) Figures.
6) The paper.

## Limitations

1) Balance is only a centre of mass check.
2) A collapse keeps what stood before it rather than scoring zero.
3) Scores are whole numbers, so many ties.
4) Results depend a lot on the starting population.
5) About 590K weights is a lot to evolve with 50 agents.
6) One choice of behaviour summary. Results may depend on it.

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
