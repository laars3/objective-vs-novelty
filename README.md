# When Does Novelty Beat the Objective?

Small neural networks build 3D structures one block at a time. The networks are
evolved, there is no backprop. The question is whether novelty search beats
plain objective search when the task is deceptive, and loses when it is not.

This is my working log for the project.

## Hypotheses

H1. On the bridge task, objective search stalls below what is possible. Holds,
it stalls at 6 when 11 is possible.

H2. On the bridge task, novelty search ends up ahead of objective search. Not
supported on 10 seeds.

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

Bridge task, population 50, 200 generations, 10 seeds per arm. Each run is
saved as json in runs/.

The hand built reference bridge reaches 11 using 23 of the 40 blocks.

Best reach per seed:

objective 6 5 3 6 7 2 7 6 6 8, median 6
blend 6 5 6 5 7 7 9 8 6 7, median 6.5
novelty 6 5 5 5 6 6 6 6 6 5, median 6
random 6 7 5 6 6 6 5 5 7 5, median 6

Mean reach over the last 50 generations, median across seeds: blend 3.80,
objective 3.45, novelty 1.82, random 1.02.

Mann-Whitney U across seeds: no two arms differ significantly on best reach.
Blend against objective gives p = 0.33 and Cliff's delta 0.27. On mean reach,
objective and blend are both well above novelty and random, p < 0.001, and
level with each other.

Objective collapses on two seeds, ending at 3 and 2 after finding its best in
generation 0 or 1. Blend's worst seed is 5.

The reason they stall shows in the structures. They all lean right up to the
balance limit, the reference too. The reference puts its counterweights on
average 5.5 blocks behind the base. Every arm keeps them 1.4 to 1.9 behind. A
block further back pulls the centre of mass back harder, so the agents use up
their blocks on weak counterweights. Placing one far behind the base scores
nothing at the time, which is why no arm finds it.

Since every arm makes the same mistake, the limit is probably the network.

Earlier bridge versions failed in two ways. On a 12 x 12 x 12 grid a random
network could already reach the maximum of 5, so evolution had nothing to
improve. With balance built into the mask, agents could never over-extend, so
the task was not deceptive and objective search reached 11.

Tower task, same settings.

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
which no arm manages on bridge.

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

1) Put the 30 seed bridge results in, once the stats in analysis.ipynb are
done. First look: blend and objective are level on the plateau, delta 0.016.
2) Split reach in the behaviour summary into reach in front of the base and
reach behind it, then rerun novelty and blend on bridge. The current summary
uses one distance from the base for both sides, so a far counterweight does
not stand out as new. If novelty gets past 6, the summary was the limit. If
not, the network probably is.
3) Then try a blend weight that changes during a run.
4) Decide whether to try a different agent.
5) Save the best structure of each generation, for a gif of a run.
6) Figures.
7) The paper.

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
