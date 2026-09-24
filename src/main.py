from evolution import Evolution
from visu import render
from fitness import bridge_score, tower_score
from agent import RandomAgent, Agent
import numpy as np

config = evolution = Evolution(grid_shape=(24,8,12), population_size=50, score_func=bridge_score, seed=0, agent_class=Agent, selection="novelty")
run_config = evolution.run(steps=40, sigma=0.002, generations=200)

print("Config: ", {k: getattr(v, "__name__", v) for k,v in {**config, **run_config}.items()})

evolution = Evolution(**config)
evolution.run(**run_config)

print("best so far: ",evolution.best_so_far)
print("np.argwhere=", np.argwhere(evolution.best_grid==1))
print("gen mean: ", evolution.gen_mean)
print("gen best: ", evolution.gen_best)

render(evolution.best_grid)