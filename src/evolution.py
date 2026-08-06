from env import Environment
from agent import Agent
import numpy as np
import torch

class Evolution:
    def __init__(self, population_size, grid_size): # creates population of agents all starting with random weights
        self.grid_size = grid_size
        self.grids=[None]*population_size
        self.population=[Agent(grid_size) for _ in range(population_size)]
        self.scores=[0]*population_size # scores start at 0, filled in after each agent builds

    def run(self, steps=20): # runs one full generation: each agent builds a structure and gets scored
        for i, current_agent in enumerate(self.population):
            env=Environment(self.grid_size) # fresh grid per agent so they dont share state

            for _ in range(steps):
                valid = env.valid_move() # recompute valid positions after every placement
                if not valid: break # no valid moves left, structure is done early
                output=current_agent.forward(env.grid, valid)
                index=torch.argmax(output).item() # pick highest scoring valid position
                x,y,z=np.unravel_index(index,(self.grid_size, self.grid_size, self.grid_size)) # flat index back to 3D coords
                env.place_block((x,y,z))
            self.scores[i]=0.0 # placeholder, real scoring comes later for now this ok
            self.grids[i]=env.grid
