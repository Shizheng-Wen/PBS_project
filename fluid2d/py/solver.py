import torch
from torch.functional import F
import numpy as np
from torch.autograd import Function

class Solver(object):
    """123
    """
    def __init__(self):
        
        super().__init__()
        self.res_x = torch.tensor(64)
        # self.res_y = (self.res_x*1.5).to(torch.int32)
        self.res_y = self.res_x
        
        self.dt = 0.05 * np.sqrt((self.res_x + self.res_y) * 0.5)
        self.dx = 1.
        self.rho = 1.
        
        self.A = None

    
    def advectValues(self, density, vel):

        density_adv = self.advectDensitySL(density, vel)
        vel_adv = self.advectVelocitySL(vel)
        return density_adv, vel_adv

    def advectDensitySL(self, density, vel):
        _, res_y, res_x = density.shape
        y_pos, x_pos = torch.meshgrid([torch.arange(0, res_y, dtype=torch.float32), torch.arange(0, res_x, dtype=torch.float32)])
        mgrid = torch.stack([x_pos, y_pos], dim=0)

        last_vx = 0.5 * (vel[0, :-1, :-1] + vel[0, :-1, 1:])
        last_vy = 0.5 * (vel[1, :-1, :-1] + vel[1, 1:, :-1])
        backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
        
        backtrace_normed_x = 2 * backtrace[0:1, ...] / (res_x-1) - 1
        backtrace_normed_y = 2 * backtrace[1:2,...] / (res_y-1) - 1
        backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y), dim=0)
        
        permutation = (1,2,0)
        backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
        density = density.unsqueeze(0)
        grid_sampled = F.grid_sample(density, backtrace_normed, padding_mode="border", align_corners=True)
        grid_sampled = grid_sampled.squeeze(0)

        return grid_sampled

    def advectVelocitySL(self, vel):
        _, res_y, res_x = vel.shape
        y_pos, x_pos = torch.meshgrid([torch.arange(0, res_y, dtype=torch.float32), torch.arange(0, res_x, dtype=torch.float32)])
        mgrid = torch.stack([x_pos, y_pos], dim=0)

        # advect vx
        last_vx = vel[0, ...]
        pad_y = (1,0,0,1)
        pad_vy = F.pad(vel[1,...], pad_y)
        last_vy = 0.25 * (pad_vy[:-1, :-1] + pad_vy[1:, :-1] + pad_vy[:-1, 1:] + pad_vy[1:, 1:])
        backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
        backtrace_normed_x = 2 * backtrace[0:1, ...] / (res_x-1) - 1
        backtrace_normed_y = 2 * backtrace[1:2,...] / (res_y-1) - 1
        backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y), dim=0)
        permutation = (1,2,0)
        backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
        vx = vel[0:1, ...]
        vx = vx.unsqueeze(0)
        grid_sampled = F.grid_sample(vx, backtrace_normed, padding_mode="border", align_corners=True)
        vx_adv = grid_sampled.squeeze(0)

        # advect vy
        pad_x = (0,1,1,0)
        pad_vx = F.pad(vel[0,...], pad_x)
        last_vx = 0.25 * (pad_vx[1:, 1:] + pad_vx[:-1, 1:] + pad_vx[1:, :-1] + pad_vx[:-1, :-1])
        last_vy = vel[1, ...]
        backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
        backtrace_normed_x = 2. * backtrace[0:1, ...] / (res_x-1) - 1.
        backtrace_normed_y = 2. * backtrace[1:2,...] / (res_y-1) - 1.
        backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y), dim=0)
        permutation = (1,2,0)
        backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
        vy = vel[1:2, ...]
        vy = vy.unsqueeze(0)
        grid_sampled = F.grid_sample(vy, backtrace_normed, padding_mode="border", align_corners=True)
        vy_adv = grid_sampled.squeeze(0)        

        return torch.cat((vx_adv, vy_adv), dim=0)

    def applySource(self, density, source):
        xmin, xmax, ymin, ymax = source
        _, res_y, res_x = density.shape
        density[0, int(ymin*res_y):int(ymax*res_y), int(xmin*res_x):int(xmax*res_x)] = 1.
        return density

    def addBuoyancy(self, density, vel):
        _, _, res_x = density.shape
        pad = (0,1,1,1)
        density_pad = F.pad(density, pad)
        scaling = 64. / res_x
        f = 0.5 * (density_pad[:, 1:, :] + density_pad[:, :-1, :]) * 0.256 * scaling
        vel_x = vel[0:1, ...]
        vel_y = vel[1:2, ...] + f * self.dt
        return torch.cat((vel_x, vel_y), dim=0)

    def addWind(self):
        self.f[0, :,:] = .005

    def addForce(self, vel, force):
        vel = vel + self.dt * force * 64
        return vel

    def solvePressure(self, vel):
        if (self.A==None):
            _, res_y, res_x = vel.shape
            Adiag = 4 * torch.ones(res_y-1, res_x-1) + 1e-3
            Aoffd = -1 * torch.ones(res_y-1, res_x-1)
            self.A = torch.stack([Adiag, Aoffd])

        div = self.computeDivergence(vel)
        pressure = self.solvePoisson(div)
        vel = self.correctVelocity(vel, pressure)
        return vel

    def computeDivergence(self, vel):
        dudx = (vel[0:1, :-1, 1:] - vel[0:1, :-1, :-1]) / self.dx
        dvdy = (vel[1:2, 1:, :-1] - vel[1:2, :-1, :-1]) / self.dx
        div = dudx + dvdy
        return div

    # matrix-vector mult
    def _Mv(self, A, b):
        Ab = torch.zeros(b.size())
        Adiag = A[0, ...]
        Aoffd = A[1, ...]
        pad_x = (1,1,0,0)
        pad_y = (0,0,1,1)
        b_pad_x = F.pad(b, pad_x)
        b_pad_y = F.pad(b, pad_y)
        Ab = Adiag*b + Aoffd*(b_pad_x[..., 2:] + b_pad_x[..., :-2] + b_pad_y[..., 2:, :] + b_pad_y[..., :-2, :])
        return Ab
    # cg
    def _solve_lse(self, rhs, A, acc=1e-5, it_max=1000):
        _, res_y, res_x = A.shape
        pressure = torch.zeros(res_y, res_x)
        r = rhs.clone()
        if (torch.norm(r, np.inf)) <= acc:
            return pressure
        z = r
        s = z.clone()
        sigma = torch.sum(z * r)
        for t in range(it_max):
            z = self._Mv(A, s)
            alpha = sigma / torch.sum(z*s)
            pressure = pressure + alpha*s
            r = r - alpha * z
            if (torch.norm(r, np.inf)) <= acc:
                return pressure
            z = r
            sigma_new = torch.sum(z*r)
            beta = sigma_new / sigma
            s = z + beta * s
            sigma = sigma_new
        print("solve Poisson failed: ", t, " ", torch.norm(r, np.inf))
        return pressure

    def solvePoisson(self, div, acc = 1e-5, it_max = 6000):
        solve_lse_fcn = self._solve_lse
        class solvePoissonOverwrite(Function):
            @staticmethod
            def forward(ctx, rhs, A):
                p = solve_lse_fcn(rhs, A, acc, it_max)
                ctx.save_for_backward(A, p)
                return p
            
            @staticmethod
            def backward(ctx, grad_output):
                A, p = ctx.saved_tensors
                grad_rhs = solve_lse_fcn(grad_output, A, 1e-10) #chain rule
                grad_Adiag = -grad_rhs * p #adjoint, dg/dx dx/dA, s.t. A dx/dA=-x
                padx = (0, 1, 0, 0)
                pady = (0, 0, 0, 1)
                p_padx = F.pad(p, pad=padx)
                grad_rhs_padx = F.pad(grad_rhs, pad=padx)
                p_pady = F.pad(p, pad=pady)
                grad_rhs_pady = F.pad(grad_rhs, pad=pady)
                # twist due to sparse format
                grad_Aoffd = -grad_rhs * p_padx[..., 1:] - grad_rhs_padx[..., 1:] * p - grad_rhs * p_pady[..., 1:, :] - grad_rhs_pady[..., 1:, :] * p
                grad_A = torch.stack([grad_Adiag, grad_Aoffd])
                return grad_rhs, grad_A

        rhs = -div / self.dt * self.rho
        solve = solvePoissonOverwrite.apply
        p = solve(rhs, self.A)

        return p

    def correctVelocity(self, vel, pressure):

        pad = (1,1,1,1)
        pad_p = F.pad(pressure, pad)
        diff_px = pad_p[:, 1:, 1:] - pad_p[:, 1:, :-1]
        diff_py = pad_p[:, 1:, 1:] - pad_p[:, :-1, 1:]
        diff_p = torch.cat((diff_px, diff_py), dim=0)
        vel_corrected = vel - diff_p * self.dt / self.dx
        return vel_corrected

