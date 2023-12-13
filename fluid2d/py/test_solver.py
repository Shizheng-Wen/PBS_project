import torch
from solver import Solver
import matplotlib.pyplot as plt

def main():
    res_x = torch.tensor(64)
    res_y = res_x
    density = torch.zeros(1, res_y, res_x)
    vel = torch.zeros(2, res_y+1, res_x+1)
    force = torch.zeros(2, res_y+1, res_x+1)
    source = (.45, .55, .1, .15)

    m = Solver()
    for i in range(120):
        # print(i)
        density = m.applySource(density, source)
        density, vel = m.advectValues(density, vel)
        vel = m.addBuoyancy(density, vel)
        vel = m.addForce(vel, force)
        vel = m.solvePressure(vel)

    fig, ax = plt.subplots()
    ax.imshow(density[0], origin="lower")
    plt.show()

if __name__ == "__main__":
    main()
