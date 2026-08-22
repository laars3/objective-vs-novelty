from evolution import Evolution
from visu import render

evolution = Evolution(grid_size=8, population_size=10)
evolution.run(steps=20, sigma=0.1, generations=100)
print(evolution.record)

render(evolution.grids[0])