import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size=size
        self.layer1=nn.Linear(size**3, 128) # squash grid into 128 hidden features
        self.layer2=nn.Linear(128, size**3) # expand back to a score for every grid position
        # layer sizes can be tuned later -- passing as args would make ablations easier

    def forward(self, grid, valid_moves):
        _grid=torch.tensor(grid, dtype=torch.float32)
        _grid=torch.flatten(_grid) # 3D grid -> 1D vector for the network
        _grid=self.layer1(_grid)
        _grid=torch.relu(_grid) # non-linearity so two linear layers dont collapse into one
        _grid=self.layer2(_grid)
         # build a mask: all positions start at -1e9 (invalid), valid positions get 0 (keeps their score)
        mask = torch.full((self.size**3,), -1e9)
        for x,y,z in valid_moves:
            i=x*(self.size**2)+(y*self.size)+z # convert (x,y,z) back to flat index
            mask[i]=0.0
        _grid=_grid+mask # invalid positions are crushed to near -1e9, argmax will never pick them
        return _grid

    def growth(self, sigma):
        for param in self.parameters():
            change=torch.randn_like(param)*sigma
            param.data += change

class RandomAgent: # random agent
    def __init__(self, size):
        self.size = size

    def forward(self, grid, valid_moves):

        scores = torch.full((self.size**3,), -1e9)
        for x,y,z in valid_moves:
            i=x*(self.size**2)+(y*self.size)+z
            scores[i]=torch.rand(()) # random scalar
        return scores

    def growth(self, sigma):
        pass # since random, no real growth happens