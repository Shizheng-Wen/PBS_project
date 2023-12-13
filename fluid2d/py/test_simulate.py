from optim import Optimization
from solver import Solver
import torch

s = Solver()
optim = Optimization(s)
optim.model.load_state_dict(torch.load("force.pth"))
optim.simulate()