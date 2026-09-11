from evolution import Evolution
from visu import render
from fitness import bridge_score, tower_score
from agent import RandomAgent, Agent

evolution = Evolution(grid_size=12, population_size=20, score_func=tower_score, seed=0, agent_class=RandomAgent)
evolution.run(steps=20, sigma=0.002, generations=20)
print(evolution.best_so_far)

render(evolution.best_grid)