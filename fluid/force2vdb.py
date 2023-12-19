from solver import Solver
import torch
from models import StreamModel
from ctypes import *
cdll.LoadLibrary("pyopenvdb.cpython-311-darwin.so")
import pyopenvdb as vdb

grid_size = (64, 64, 64)
s = Solver(grid_size)
source = torch.load('data/bunny.pt')*2
source_time = 1
m = StreamModel(s, source, source_time)

fc = torch.load('eth_bunny_force.pth')
m.load_state_dict(fc)

density = torch.zeros((1, 64, 64, 64))
vel = torch.zeros((3, 65, 65, 65))
m.compute_force()
grids = []
for f in range(40):
    density, vel = m(density, vel, f)
    array = density.squeeze(0).detach().numpy()
    grid = vdb.FloatGrid()
    grid.copyFromArray(array)
    grid.name = '{:02d}'.format(f)
    grids.append(grid)
    vdb.write('eth_bunny.vdb', grids=grids)
