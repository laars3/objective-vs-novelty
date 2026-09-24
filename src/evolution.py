from env import Environment
import numpy as np, torch, random, copy
from fitness import bc, novelty

class Archive:
    def __init__(self, rho_min=0.1, max_add=4, patience=5, raise_by=1.05, lower_by=0.95):
        self.rho_min = rho_min
        self.max_add = max_add
        self.patience = patience
        self.raise_by = raise_by
        self.lower_by = lower_by
        self.vectors = np.empty((0,5))
        self.gens_since_add =0


    def update(self, bcs,novelties):

        target = novelties > self.rho_min
        self.vectors = np.vstack([self.vectors, bcs[target]])

        # rho adjustment
        n = target.sum()
        if n > self.max_add:
            self.rho_min = (self.rho_min * self.raise_by)
            self.gens_since_add = 0
        elif n == 0:
            self.gens_since_add += 1
            if (self.gens_since_add) >= self.patience:
                self.rho_min = (self.rho_min * self.lower_by)
                self.gens_since_add = 0
        else:
            self.gens_since_add =0

class Evolution:
    def __init__(self, population_size, grid_shape, score_func, seed, agent_class, selection): # creates population of agents all starting with random weights
        torch.manual_seed(seed)
        random.seed(seed)
        self.best_grid=None
        self.agent_class = agent_class
        self.score_func = score_func
        self.seed = seed
        self.grid_shape = grid_shape
        self.grids=[None]*population_size
        self.population=[agent_class(grid_shape) for _ in range(population_size)]
        self.scores=[0]*population_size # scores start at 0, filled in after each agent builds
        self.gen_best=[]
        self.best_so_far=[]
        self.best =-1
        self.gen_mean=[]
        self.selection = selection
        self.archive = Archive()

    def run(self, steps=20, sigma=0.1, generations=100): # runs one full generation: each agent builds a structure and gets scored
        for generation in range(generations):
            for i, current_agent in enumerate(self.population):
                env=Environment(self.grid_shape) # fresh grid per agent so they dont share state

                for _ in range(steps):
                    valid = env.valid_move() # recompute valid positions after every placement
                    if not valid: break # no valid moves left, structure is done early
                    output=current_agent.forward(env.grid, valid)
                    index=torch.argmax(output).item() # pick highest scoring valid position
                    x,y,z=np.unravel_index(index, self.grid_shape) # flat index back to 3D coords
                    if not env.place_block((x,y,z)):
                        break

                self.scores[i] = self.score_func(env.grid)
                self.grids[i] = env.grid

            if self.selection == "novelty": # checks if scores to rank on should be novelty or based on distance like objective
                rows=[]
                for g in self.grids:
                    rows.append(bc(g))
                bcs = np.array(rows)
                rank_scores = novelty(bcs, self.archive.vectors)
                self.archive.update(bcs, rank_scores)
            else:
                rank_scores = self.scores

            maximum=max(self.scores)
            self.gen_best.append(maximum)
            if maximum > self.best:
                self.best=maximum
                self.best_grid=self.grids[self.scores.index(maximum)]
            self.best_so_far.append(self.best)

            self.gen_mean.append((sum(self.scores)/len(self.population)))

            ranked=sorted(range(len(self.scores)), key=lambda i: rank_scores[i], reverse=True) # ranks scores assigned to index
            keep=len(self.population)//2 # drops lower half of scores
            survivors=[self.population[i] for i in ranked[:keep]]

            next_gen= [survivors[0]]
            while len(next_gen) < len(self.population): # repopulates the next generation list with random survivor and applies slight deviation
                parent=random.choice(survivors)
                child=copy.deepcopy(parent)
                child.growth(sigma)
                next_gen.append(child)
            self.population=next_gen
