import numpy as np

def tower_score(grid):
    filled=np.argwhere(grid==1)
    height=filled[:, 2].max()
    return height