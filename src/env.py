import numpy as np

class Environment:
    def __init__(self, shape): # takes in a shape
        self.shape=shape
        self.base=(shape[0]//2, shape[1]//2, 0) # base cell is fixed at center of ground plane
        _grid = np.zeros(self.shape, dtype=np.int8)
        self.grid=_grid
        self.place_block(self.base) # every structure starts from the base

    def reset(self): # wipes grid back to all zeros and replaces base block for a new episode
        self.grid=np.zeros(self.shape, dtype=np.int8)
        self.place_block(self.base)

    def place_block(self,coordinates): # checks if after cell placement the structure is stable, if not set back to 0, else continue
        self.grid[coordinates]=1
        stable = self.is_stable()
        if not stable:
            self.grid[coordinates]=0
        return stable

    def valid_move(self): # returns list of positions the agent is allowed to build on this step
        valid_blocks=[]

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                for z in range(self.shape[2]):
                    if self.grid[x,y,z]==1: # skip cells that are already filled
                        continue

                    is_base = (x,y,z)==self.base

                    # all six face-adjacent neighbors of this cell
                    neighbors = [(x-1,y,z), (x+1,y,z), (x, y-1,z), (x, y+1, z), (x,y,z-1), (x,y,z+1)]
                    # support rule: at least one neighbor must be inside the grid and already filled
                    supported = any(0 <= nx < self.shape[0] and 0 <= ny < self.shape[1] and 0 <= nz < self.shape[2] and self.grid[nx, ny,nz]==1 for nx,ny,nz in neighbors)

                    if (supported or is_base) and (z >0 or is_base):
                        valid_blocks.append((x,y,z))

        return valid_blocks

    def best_possible_bridge(self,steps): # find best possible bridge reach to compare
        grid=self.grid
        max_x_grid = self.shape[0]-1
        min_x_grid =0
        y=self.base[1]
        z=1

        self.place_block((self.base[0], self.base[1], 1))
        steps=steps-1
        while steps:
            filled = np.argwhere(grid == 1)

            valid = self.valid_move()
            max_x = filled[:, 0].max()
            min_x = filled[:, 0].min()

            if max_x_grid == max_x:
                break
            if ((max_x + 1, y, z)) in valid and self.place_block((max_x + 1, y, z)):
                pass
            elif ((min_x - 1, y, z)) in valid and self.place_block((min_x - 1, y, z)):
                pass
            else:
                break
            steps=steps-1
        return grid

    def is_stable(self):
        filled = np.argwhere(self.grid == 1)

        x_mean = filled[:, 0].mean()
        y_mean = filled[:, 1].mean()

        # balance rule checks if structure collapses if the mean deviates more than 0.5 off base
        balanced = abs(x_mean - self.base[0]) <= 0.5 and abs(y_mean - self.base[1]) <= 0.5

        return balanced