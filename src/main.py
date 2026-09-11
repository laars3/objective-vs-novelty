from evolution import Evolution
from visu import render
from fitness import bridge_score, tower_score
from agent import RandomAgent, Agent
import numpy as np

evolution = Evolution(grid_size=12, population_size=50, score_func=bridge_score, seed=0, agent_class=Agent)
evolution.run(steps=20, sigma=0.002, generations=100)

print("best so far: ",evolution.best_so_far)
print("np.argwhere=", np.argwhere(evolution.best_grid==1))
print("gen mean: ", evolution.gen_mean)
print("gen best: ", evolution.gen_best)

render(evolution.best_grid)