import numpy as np

class Environment:
    def __init__(self, size): # creates grid of zeros
        self.size=size
        self.base=(size // 2, size // 2, 0)
        _grid = np.zeros((self.size,self.size,self.size), dtype=np.int8)
        self.grid=_grid
        self.place_block(self.base)

    def reset(self): # resets given grid for new blocks to be placed
        self.grid=np.zeros((self.size,self.size,self.size), dtype=np.int8)
        self.place_block(self.base)

    def place_block(self,coordinates): # places block on coordinate
        self.grid[coordinates]=1