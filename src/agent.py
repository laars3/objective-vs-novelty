import torch
import torch.nn as nn

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
