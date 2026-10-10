import torch
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape=shape
        self.cells=shape[0]*shape[1]*shape[2] # one network input and one output per grid cell
        self.layer1=nn.Linear(self.cells, 128) # squash grid into 128 hidden features
        self.layer2=nn.Linear(128, self.cells) # expand back to a score for every grid position
        # layer sizes can be tuned later -- passing as args would make ablations easier

    def forward(self, grid, valid_moves):
        _grid=torch.tensor(grid, dtype=torch.float32)
        _grid=torch.flatten(_grid) # 3D grid -> 1D vector for the network
        _grid=self.layer1(_grid)
        _grid=torch.relu(_grid) # non-linearity so two linear layers dont collapse into one
        _grid=self.layer2(_grid)
        # build a mask: all positions start at -1e9 (invalid), valid positions get 0 (keeps their score)
        mask = torch.full((self.cells,), -1e9)
        ny,nz=self.shape[1],self.shape[2]
        for x,y,z in valid_moves:
            i=x*(ny*nz)+(y*nz)+z # (x,y,z) to flat index, C order so it matches np.unravel_index
            mask[i]=0.0
        _grid=_grid+mask # invalid positions are crushed to near -1e9, argmax will never pick them
        return _grid

    def growth(self, sigma):
        for param in self.parameters():
            change=torch.randn_like(param)*sigma
            param.data += change

class RandomAgent: # random agent
    def __init__(self, shape):
        self.shape = shape
        self.cells = shape[0]*shape[1]*shape[2]

    def forward(self, grid, valid_moves):

        scores = torch.full((self.cells,), -1e9)
        ny,nz=self.shape[1],self.shape[2]
        for x,y,z in valid_moves:
            i=x*(ny*nz)+(y*nz)+z
            scores[i]=torch.rand(()) # random scalar
        return scores

    def growth(self, sigma):
        pass # since random, no real growth happens

class LocalAgent(nn.Module): # scores each valid cell from local nums, one small net shared by every cell
    def __init__(self, shape, n_features=12, hidden=16):
        super().__init__()
        self.shape = shape
        self.cells = shape[0] * shape[1] * shape[2]
        self.base_x, self.base_y = shape[0] // 2, shape[1] // 2 # positions are measured from the base
        self.layer1 = nn.Linear(n_features, hidden) # turns the 12 cell nums into 16 combinations of them
        self.layer2 = nn.Linear(hidden, 1) # turns the 16 combinations into 1 score for that cell
        # 225 weights total, so sigma has to be bigger than for Agent (weights start around 0.3 not 0.02)

    def features(self, grid, valid_moves): # 12 nums per valid cell (neighbors, position, lean, blocks used)
        nz = self.shape[2]
        filled = np.argwhere(grid == 1)

        # same for every cell this step: how far the structure leans and how much budget is gone
        com_x = filled[:, 0].mean() - self.base_x
        com_y = filled[:, 1].mean() - self.base_y
        used = len(filled) / 41
        p = np.pad(grid, 1) # pad grid so edge cells have neighbors, cell (x,y,z) is now at (x+1,y+1,z+1)

        rows = []
        for x, y, z in valid_moves:
            X, Y, Z = x+1, y+1, z+1
            neighbors = [p[X-1,Y,Z], p[X+1,Y,Z], p[X,Y-1,Z], p[X,Y+1,Z], p[X,Y,Z-1], p[X,Y,Z+1]] # 1 if filled, back/front, sides, below/above
            position = [(x-self.base_x)/self.base_x, (y-self.base_y)/self.base_y, z/(nz-1)] # negative x means behind the base
            rows.append(neighbors + position + [com_x, com_y, used]) # one row per valid cell
        return torch.from_numpy(np.array(rows, dtype=np.float32)) # (n_valid, 12)

    def forward(self, grid, valid_moves):
        out = self.layer1(self.features(grid, valid_moves)) # all valid cells go through at once
        out = torch.relu(out) # lets a combination switch on only in some cases (front is good unless leaning forward)
        out = self.layer2(out).squeeze(1) # drop the extra dim so its one score per cell

        # put each score back at its spot in the full grid, invalid cells stay -1e9 so argmax never picks them
        scores = torch.full((self.cells,), -1e9)
        ny, nz = self.shape[1], self.shape[2]
        for (x, y, z), s in zip(valid_moves, out):
            scores[x*(ny*nz)+(y*nz)+z] = s # same flat index as Agent so evolution doesnt need to change
        return scores

    def growth(self, sigma): # same as Agent, small random nudge to every weight
        for param in self.parameters():
            change=torch.randn_like(param)*sigma
            param.data += change
