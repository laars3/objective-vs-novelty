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
