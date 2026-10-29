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

One bridge seed per arm, the one closest to its arm's median. Left is a hand
built reference, right is the best structure of each generation. Red is in
front of the base, blue behind it and grey above it.

![objective seed 23](gifs/objective_bridge_s23.gif)

MLP, objective, seed 23. Reach 6 by generation 34, then barely changes.

![novelty seed 22](gifs/novelty_bridge_s22.gif)

MLP, novelty, seed 22. Keeps changing shape, mostly at reach 5.

![blend seed 28](gifs/blend_bridge_s28.gif)

MLP, blend, seed 28. Also keeps changing, at reach 5 and then 6 near the end.

![objectivelocal seed 20](gifs/objectivelocal_bridge_s20.gif)

LocalAgent, objective, seed 20. Starts as a column and ends on the reference
bridge.

![objectivenolean seed 2](gifs/objectivenolean_bridge_s2.gif)

LocalAgent without lean inputs, seed 2. A column with an arm of 6 by
generation 50, kept to the end.

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
