from evolution import Evolution
from visu import render
from fitness import reach_score, tower_score

evolution = Evolution(grid_size=12, population_size=10, score_func=tower_score, seed=0)
evolution.run(steps=20, sigma=0.002, generations=5)
print(evolution.record)

render(evolution.grids[0])