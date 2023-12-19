"""Used to test fluid solver and generate target shape at keyframe
"""

import torch
from solver import Solver
import matplotlib.pyplot as plt
from ctypes import *
import os
cdll.LoadLibrary("pyopenvdb.cpython-311-darwin.so")
import pyopenvdb as vdb


def test2d(mac_on, dir=None):
    if dir != None:
        os.system('rm -rf '+dir)
        os.system('mkdir '+dir)
    res_x = int(64)
    res_y = int(res_x)
    density = torch.zeros((1, res_y, res_x))
    vel = torch.zeros((2, res_y+1, res_x+1))
    force = torch.zeros(vel.shape)
    # force[0,...] = 5e-2
    source = torch.zeros(density.shape)
    source[0, 4:8, 28:36] = 1.

    m = Solver((res_x, res_y))

    for i in range(60):
        print(i)
        density = m.applySource(density, source)
        density, vel = m.advectValues(density, vel, mac_on)
        vel = m.addBuoyancy(density, vel)
        vel = m.addForce(vel, force)
        vel = m.solvePressure(vel)

        if (dir != None):
            plt.imshow(1-density[0], origin="lower", cmap='Greys',  interpolation='nearest')
            plt.axis('off')
            plt.savefig(dir+'/{:02d}.png'.format(i), dpi=120)
    if dir==None:
        print(density.sum())
        plt.imshow(1-density[0], origin="lower", cmap='Greys',  interpolation='nearest')
        plt.show()
    # torch.save(density, 'data/target.pt')


def test3d(mac_on, filename=None):
    res_x = int(64)
    res_z = int(res_x)
    res_y = int(res_x*1.5)
    density = torch.zeros((1, res_z, res_y, res_x))
    vel = torch.zeros((3, res_z+1, res_y+1, res_x+1))
    force = torch.zeros(vel.shape)
    source = torch.zeros(density.shape)
    source[0, 24:40, 8:16, 24:40] = 1.

    m = Solver((res_x, res_y, res_z))
    grids = []
    for i in range(60):
        print(i)
        density = m.applySource(density, source)
        density, vel = m.advectValues(density, vel, mac_on)
        vel = m.addBuoyancy(density, vel)
        vel = m.addForce(vel, force)
        vel = m.solvePressure(vel)

        if filename != None:
            array = density.squeeze(0).numpy()
            grid = vdb.FloatGrid()
            grid.copyFromArray(array)
            grid.name = '{:02d}'.format(i)
            grids.append(grid)
    if filename != None:
        vdb.write(filename, grids=grids)
    else:
        plt.imshow(density.max()-density.mean(dim=1)[0], origin="lower", cmap='Greys',  interpolation='nearest')
        plt.show()

def main():
    test2d(False)
    test3d(False)

if __name__ == "__main__":
    main()
