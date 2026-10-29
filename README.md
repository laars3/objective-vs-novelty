# When Does Novelty Beat the Objective?

Small neural networks build 3D structures one block at a time and are evolved
without backprop. Random, objective, novelty and blend selection are compared
on a bridge task, which is deceptive, and a tower task, which is not. Two
agents are tested. One is an MLP that reads the whole grid. The other is
LocalAgent, a small network that scores each cell from local inputs.

The full write up is in paper.pdf.

## Results

Novelty search did not beat objective search on either task. Its best scores
were level with objective, and its population mean was lower everywhere.

With the MLP, objective search got a median bridge of 5.5 out of 11 after 200
generations. Blend had the same population mean as objective.

LocalAgent can see how far the structure leans, and it reached a median best of
11 with every arm. Without the two lean inputs it did about as well as the MLP.
What the agent could see mattered more than the search method.

## Gifs

Each gif shows one bridge seed per arm, the one closest to its arm's median.
The left side is a hand built reference bridge. The right side is the best
structure of each generation. Red blocks are in front of the base, blue behind
it and grey above it.

![objectivelocal seed 20](gifs/objectivelocal_bridge_s20.gif)

LocalAgent with objective search, seed 20. The run starts from a column with
reach 3 and ends on the same bridge as the reference.

![objectivenolean seed 2](gifs/objectivenolean_bridge_s2.gif)

LocalAgent without the lean inputs, seed 2. By generation 50 it builds a column
of about 10 blocks on the base with an arm of 6, and it keeps that structure to
the end.

![objective seed 23](gifs/objective_bridge_s23.gif)

MLP with objective search, seed 23. Reach 6 by generation 34, using all 40
blocks, and the structure barely changes after that.

![novelty seed 22](gifs/novelty_bridge_s22.gif)

MLP with novelty search, seed 22. The structure keeps changing shape. Reach is
5 in most generations and drops to 3 or 4 in about a quarter of them.

![blend seed 28](gifs/blend_bridge_s28.gif)

MLP with blend, seed 28. The structure also keeps changing, at reach 5 for most
of the run and 6 in the last generations.

## Running it

```
pip install -r requirements.txt
python src/main.py
```

main.py runs the MLP bridge arms for 30 seeds and saves each run as json in
runs/. Runs that already exist are skipped. The other arms from the paper are
listed in a comment at the top of the file. All runs used in the paper are
already in runs/.

analysis.ipynb computes the table and the stats and makes the curve figures in
stills/. src/record.py reruns chosen seeds and saves the best structure of every
generation to frames/. src/visu.py turns those into the gifs in gifs/ and the
structure figure in stills/.
