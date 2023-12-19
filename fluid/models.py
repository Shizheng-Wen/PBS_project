import torch
from torch.functional import F

# force field
class Model(torch.nn.Module):
    def __init__(self, solver, source, source_time, mac_on):
        super().__init__()
        self.solver = solver
        self.mac_on = mac_on
        self.source = source
        self.source_time = source_time

        # wind force
        if solver.dim == 2:
            self.param = torch.nn.Parameter(torch.zeros((2, solver.res_y+1, solver.res_x+1)))
        else:
            self.param = torch.nn.Parameter(torch.zeros((3, solver.res_z+1, solver.res_y+1, solver.res_x+1)))

    def compute_force(self):
        # avoid registering self.force as nn.Parameter
        self.force = self.param + 0.0
    
    def forward(self, density, vel, t, force_rescale_factor=1.0):
        force = self.force * force_rescale_factor
        if (t < self.source_time):
            density = self.solver.applySource(density, self.source)
        density, vel = self.solver.advectValues(density, vel, self.mac_on)
        vel = self.solver.addBuoyancy(density, vel)
        vel = self.solver.addForce(vel, force)
        vel = self.solver.solvePressure(vel)

        return density, vel

# stream function
class StreamModel(Model):
    def __init__(self, solver, sources, source_time, mac_on=False):
        super().__init__(solver, sources, source_time, mac_on)
        if solver.dim == 2:
            self.param = torch.nn.Parameter(torch.zeros((1, solver.res_y+1, solver.res_x+1)))

    def get_curl(self, stream):
        if self.solver.dim == 2:
            stream_x = stream[..., 1:] - stream[..., :-1]
            stream_x = F.pad(stream_x, pad=(0,1,0,0))
            stream_y = stream[..., 1:, :] - stream[..., :-1, :]
            stream_y = F.pad(stream_y, pad=(0,0,0,1))
            curl = torch.cat((stream_y, -stream_x), dim=0)
        else:
            # in 3D case, the values of stream fcn are stored on the edges of the cell
            gridx = stream[0:1, ...]
            gridy = stream[1:2, ...]
            gridz = stream[2:3, ...]
            # compute finite difference gradients
            dgridx_dz = gridx[:, 1:, ...] - gridx[:, :-1, ...]
            dgridx_dy = gridx[..., 1:, :] - gridx[..., :-1, :]
            dgridy_dx = gridy[..., 1:] - gridy[..., :-1]
            dgridy_dz = gridy[:, 1:, ...] - gridy[:, :-1, ...]
            dgridz_dx = gridz[..., 1:] - gridz[..., :-1]
            dgridz_dy = gridz[..., 1:, :] - gridz[..., :-1, :]
            # abandon redundant slices
            curl_x = dgridz_dy[:, :-1, ...] - dgridy_dz[..., :-1, :]
            curl_y = dgridx_dz[..., :-1] - dgridz_dx[:, :-1, ...]
            curl_z = dgridy_dx[..., :-1, :] - dgridx_dy[..., :-1]
            # pad redundant slices back for 3D MAC grid
            curl_x = F.pad(curl_x, pad=(0, 0, 0, 1, 0, 1))
            curl_y = F.pad(curl_y, pad=(0, 1, 0, 0, 0, 1))
            curl_z = F.pad(curl_z, pad=(0, 1, 0, 1, 0, 0))
            curl = torch.cat((curl_x, curl_y, curl_z), dim=0)
        return curl

    def compute_force(self):
        # avoid registering self.force as nn.Parameter
        self.force = self.get_curl(self.param)
    