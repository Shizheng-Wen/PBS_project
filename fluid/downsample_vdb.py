from ctypes import *
cdll.LoadLibrary("pyopenvdb.cpython-311-darwin.so")
import pyopenvdb as vdb
import numpy as np
import torch

bunny = vdb.read('data/bunny_cloud.vdb', 'density')
xyzMin, xyzMax = bunny.evalActiveVoxelBoundingBox()

array = np.zeros((577,572,438))
bunny.copyToArray(array, ijk=xyzMin)

input = torch.from_numpy(array).to(torch.float32)
input = input.unsqueeze(0).unsqueeze(0)
dx = torch.linspace(-1, 1, 40)
dy = torch.linspace(-1, 1, 40)
dz = torch.linspace(-1, 1, 40)
meshx, meshy, meshz = torch.meshgrid((dz, dy, dx))
grid = torch.stack((meshx, meshy, meshz), 3)
grid = grid.unsqueeze(0)
output = torch.nn.functional.grid_sample(input, grid)

src = torch.zeros((1,64,64,64))
src[0,12:52, :40, 12:52] = output[0, 0]
torch.save(src, 'data/src.pt')

grid = vdb.FloatGrid()
grid.copyFromArray(src.squeeze(0).numpy())
grid.name = 'density'
vdb.write('data/src.vdb', grids=[grid])