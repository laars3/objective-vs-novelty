from evolution import Evolution
from visu import render

evolution = Evolution(grid_size=8, population_size=10)
evolution.run(steps=20)

render(evolution.grids[0])