from env import Environment
from agent import Agent
import numpy as np, torch, random, copy
from fitness import tower_score

class Evolution:
    def __init__(self, population_size, grid_size): # creates population of agents all starting with random weights
        self.grid_size = grid_size
        self.grids=[None]*population_size
        self.population=[Agent(grid_size) for _ in range(population_size)]
        self.scores=[0]*population_size # scores start at 0, filled in after each agent builds
        self.record= []

    def run(self, steps=20, sigma=0.1, generations=100): # runs one full generation: each agent builds a structure and gets scored
        for generation in range(generations):
            for i, current_agent in enumerate(self.population):
                env=Environment(self.grid_size) # fresh grid per agent so they dont share state

                for _ in range(steps):
                    valid = env.valid_move() # recompute valid positions after every placement
                    if not valid: break # no valid moves left, structure is done early
                    output=current_agent.forward(env.grid, valid)
                    index=torch.argmax(output).item() # pick highest scoring valid position
                    x,y,z=np.unravel_index(index,(self.grid_size, self.grid_size, self.grid_size)) # flat index back to 3D coords
                    env.place_block((x,y,z))
                self.scores[i]=tower_score(env.grid)
                self.grids[i]=env.grid

            self.record.append(max(self.scores)) # appends max score

            ranked=sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True) # ranks scores assigned to index
            keep=len(self.population)//2 # drops lower half of scores
            survivors=[self.population[i] for i in ranked[:keep]]

            next_gen= [survivors[0]]
            while len(next_gen) < len(self.population): # repopulates the next generation list with random survivor and applies slight deviation
                parent=random.choice(survivors)
                child=copy.deepcopy(parent)
                child.growth(sigma)
                next_gen.append(child)
            self.population=next_gen
