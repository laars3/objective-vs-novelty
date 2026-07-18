import torch
import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size=size
        self.layer1=nn.Linear(size**3, 128) # takes size**3 grid and converts it to 128 value
        self.layer2=nn.Linear(128, size**3) # takes the 128 values and converts it back to a 1000 scoring them
# all of these values can be adjusted layer, I might pass them as args and see how different values create different results
    def forward(self, grid):
        _grid=torch.tensor(grid, dtype=torch.float32)
        _grid=torch.flatten(_grid)
        _grid=self.layer1(_grid)
        _grid=torch.relu(_grid)
        _grid=self.layer2(_grid)
        return _grid
