import numpy as np

def tower_score(grid):
    filled=np.argwhere(grid==1)
    height=filled[:, 2].max()
    return height

def bridge_score(grid):
    filled=np.argwhere(grid==1)
    max_x=filled[:, 0].max()
    base_x=grid.shape[0]//2
    return max_x-base_x

def bc(grid, budget=40): # 5 num summary of structure, novelty measures against these
    filled = np.argwhere(grid==1)
    nx,ny,nz = grid.shape

    block_count = (len(filled)/(budget+1))

    max_height=(filled[:, 2].max()/(nz-1))
    com_height = filled[:, 2].mean()/(nz-1) # mass center

    corner = np.hypot(max(nx//2, nx-1-nx//2), max(ny//2, ny-1-ny//2)) # furthest block can be from base is grid corner
    horizontal_dist = np.hypot((filled[:, 0]-(nx//2)), filled[:, 1]-(ny//2)) # horizontal dist from base

    max_reach = horizontal_dist.max()/corner
    com_reach = horizontal_dist.mean()/ corner

    return np.array([block_count, max_height,com_height, max_reach, com_reach])