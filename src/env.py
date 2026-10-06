import numpy as np
from networkx.classes import neighbors
from sympy.codegen.ast import continue_


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

    def valid_move_new(self): # new valid_move func, instead of checking every single position in the grid, just check neighbors of already filled blocks

        valid_blocks = set()
        filled = np.argwhere(self.grid==1)

        for dx,dy,dz in filled:
            neighbors = [(dx - 1, dy, dz),(dx + 1, dy, dz),(dx, dy - 1, dz),(dx, dy + 1, dz),(dx, dy, dz - 1), (dx, dy, dz + 1)]
            for x,y,z in neighbors:
                if 0<=x<self.shape[0] and 0<=y<self.shape[1] and 0<z<self.shape[2] and self.grid[x,y,z]!=1: # keep if inside grid, empty, and z>0
                    valid_blocks.add((int(x),int(y),int(z)))
        return sorted(valid_blocks)

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

            valid = self.valid_move_new()
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