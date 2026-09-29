# When Does Novelty Beat the Objective?

Small neural networks build 3D structures one block at a time. The networks are
evolved, there is no backprop. The question is whether novelty search beats
plain objective search when the task is deceptive, and loses when it is not.

This is my working log for the project.

## Hypotheses

H1. On the bridge task, objective search stalls below what is possible. Holds,
it stalls at 6 when 11 is possible.

H2. On the bridge task, novelty search ends up ahead of objective search. Does
not hold on two seeds so far.

H3. On the tower task, objective search does at least as well as novelty. Not
tested yet.

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

Arms are always compared on reach, using the population mean.

## Results

Population 50, 200 generations, two seeds.

The hand built reference bridge reaches 11 using 23 of the 40 blocks.

Objective reaches 6 on seed 0 and 5 on seed 1, and stops improving around
generation 86 on both. Mean reach 4.0 and 3.5.

Blend reaches 6 and 5, mean 3.5 and 3.3.

Novelty reaches 6 and 5, mean 1.9 and 1.7.

Random reaches 6, mean 1.05.

So every arm reaches the same best within a seed, and the seed matters more than
the arm. The means differ, objective highest and novelty lowest.

The reason they stall shows in the structures. They all lean right up to the
balance limit, the reference too. The reference puts its counterweights on
average 5.5 blocks behind the base. The agents keep theirs about 2 behind. A
block further back pulls the centre of mass back harder, so the agents use up
their blocks on weak counterweights. Placing one far behind the base scores
nothing at the time, which is why no arm finds it.

Novelty builds taller and wider sideways than objective. Those directions rarely
cause a collapse, so they are where it is easy to be different.

Since all four arms stop at the same place, the limit is probably the network
and not the selection method.

Earlier versions failed in two ways. On a 12 x 12 x 12 grid a random network
could already reach the maximum of 5, so evolution had nothing to improve. With
balance built into the mask, agents could never over-extend, so the task was not
deceptive and objective search reached 11.

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

1) Save each run to a file, including the order the blocks were placed.
2) A script that runs 10 seeds of each arm and saves them.
3) Tower runs for H3.
4) Decide whether to try a different agent, or write up all arms hitting the
same limit.
5) Statistics: Mann-Whitney U and Cliff's delta across seeds.
6) Figures, and a gif of a structure being built.
7) The paper.

## Limitations

1) Balance is a centre of mass check, not physics.
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
