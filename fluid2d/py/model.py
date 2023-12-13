import torch

class Model(torch.nn.Module):
    def __init__(self, solver, source, source_time):
        super().__init__()
        self.solver = solver

        self.source = source
        self.source_time = source_time

        self.param = torch.nn.Parameter(torch.zeros((2, solver.res_y+1, solver.res_x+1), dtype=torch.float32))

    def compute_force(self):
        # avoid registering self.force as nn.Parameter
        self.force = self.param + 0.0

    def get_force(self):
        return self.force
    
    def forward(self, density, vel, t, force_rescale_factor=1.0):
        force = self.force * force_rescale_factor
        if (t < self.source_time):
            density = self.solver.applySource(density, self.source)
        density, vel = self.solver.advectValues(density, vel)
        vel = self.solver.addBuoyancy(density, vel)
        vel = self.solver.addForce(vel, force)
        vel = self.solver.solvePressure(vel)

        return density, vel
