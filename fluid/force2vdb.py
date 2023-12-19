from solver import Solver
import torch
from models import StreamModel
from ctypes import *
cdll.LoadLibrary("pyopenvdb.cpython-311-darwin.so")
import pyopenvdb as vdb

grid_size = (64, 64, 64)
s = Solver(grid_size)
source = torch.load('data/bunny.pt')
source_time = 1
m = StreamModel(s, source, source_time)

fc, keyframe = torch.load('data/crl_bunny_res.pt')
# fc = torch.load('data/crl_stream_force.pth')
# keyframe = 40
m.load_state_dict(fc)

density = torch.zeros((1, 64, 64, 64))
vel = torch.zeros((3, 65, 65, 65))
m.compute_force()
grids = []
offset = 0
for f in range(keyframe+40):
    if f == keyframe or f == 1:
        for i in range(10):
            array = density.squeeze(0).detach().numpy()
            grid = vdb.FloatGrid()
            grid.copyFromArray(array)
            grid.name = '{:02d}'.format(f+offset)
            grids.append(grid)
            offset += 1
    density, vel = m(density, vel, f)
    array = density.squeeze(0).detach().numpy()
    grid = vdb.FloatGrid()
    grid.copyFromArray(array)
    grid.name = '{:02d}'.format(f+offset)
    grids.append(grid)
vdb.write('crl_bunny.vdb', grids=grids)
