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

    def valid_move(self): # valid move func that checks if either base block or has a neighbor connected where it would make the move valid
        valid_blocks=[]
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    if self.grid[x,y,z]==1:
                        continue

                    # all possible neighbors to a single block
                    neighbors = [(x-1,y,z), (x+1,y,z), (x, y-1,z), (x, y+1, z), (x,y,z-1), (x,y,z+1)]
                    # returns true if any of the neighbors are present making the box a valid place to build
                    supported = any(0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size and self.grid[nx, ny,nz]==1 for nx,ny,nz in neighbors)

                    # balance check
                    filled =np.argwhere(self.grid==1)

                    new_mean_x= (filled[:,0].sum()+x)/(len(filled)+1)
                    new_mean_y=(filled[:,1].sum()+y)/(len(filled)+1)

                    balanced = abs(new_mean_x-self.base[0])<=0.5 and abs(new_mean_y-self.base[1])<=0.5

                    if (supported or (x,y,z) == self.base) and balanced:
                        valid_blocks.append((x,y,z))

        return valid_blocks