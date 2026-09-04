import numpy as np

class Environment:
    def __init__(self, size): # creates an empty size x size x size grid and places the base block at center ground
        self.size=size
        self.base=(size // 2, size // 2, 0) # base cell is fixed at center of ground plane
        _grid = np.zeros((self.size,self.size,self.size), dtype=np.int8)
        self.grid=_grid
        self.place_block(self.base) # every structure starts from the base

    def reset(self): # wipes grid back to all zeros and replaces base block for a new episode
        self.grid=np.zeros((self.size,self.size,self.size), dtype=np.int8)
        self.place_block(self.base)

    def place_block(self,coordinates): # sets a single cell to 1 (occupied)
        self.grid[coordinates]=1

    def valid_move(self): # returns list of positions the agent is allowed to build on this step
        valid_blocks=[]

        filled = np.argwhere(self.grid == 1)
        x_sum = filled[:,0].sum()
        y_sum= filled[:,1].sum()

        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if self.grid[x,y,z]==1: # skip cells that are already filled
                        continue

                    # all six face-adjacent neighbors of this cell
                    neighbors = [(x-1,y,z), (x+1,y,z), (x, y-1,z), (x, y+1, z), (x,y,z-1), (x,y,z+1)]
                    # support rule: at least one neighbor must be inside the grid and already filled
                    supported = any(0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size and self.grid[nx, ny,nz]==1 for nx,ny,nz in neighbors)

                    # balance rule: placing here must keep center of mass within 0.5 of base in x and y
                    new_mean_x= (x_sum+x)/(len(filled)+1)
                    new_mean_y= (y_sum+y)/(len(filled)+1)
                    balanced = abs(new_mean_x-self.base[0])<=0.5 and abs(new_mean_y-self.base[1])<=0.5

                    if (supported or (x,y,z) == self.base) and balanced:
                        valid_blocks.append((x,y,z))

        return valid_blocks