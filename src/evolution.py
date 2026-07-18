from env import Environment
from agent import Agent
import numpy as np
import torch

class Evolution:
    def __init__(self, population_size, grid_size): # builds population list
        self.population=[Agent(grid_size) for _ in range(population_size)]
        scores=[0]*population_size
        self.scores=scores

    def run(self, grid_size, steps=20):
        for i, current_agent in enumerate(self.population):
            env=Environment(grid_size) # fresh grid

            for _ in range(steps):
                output=current_agent.forward(env.grid)
                index=torch.argmax(output).item()
                x,y,z=np.unravel_index(index,(grid_size, grid_size, grid_size))
                env.place_block((x,y,z))
            self.scores[i]=0.0
