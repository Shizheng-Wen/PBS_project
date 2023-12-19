import numpy as np
import torch
from torch.functional import F
from torch.autograd import Function

class Solver(object):
    """ Fluid solver for 2d and 3d
    """
    def __init__(self, grid_size):
        
        super().__init__()
        self.dim = len(grid_size)
        if (self.dim == 2):
            self.res_x, self.res_y = grid_size
            self.dt = 0.05 * np.sqrt((self.res_x + self.res_y) * 0.5)
        else:
            self.res_x, self.res_y, self.res_z = grid_size
            self.dt = 0.05 * np.sqrt((self.res_x+self.res_y+self.res_z)/3)
        self.dx = 1.
        self.rho = 1.
        
        # sparse Poisson matrix
        self.A = None

    
    def advectValues(self, density, vel, mac_on=False):

        if mac_on:
            d_forward = self.advectDensitySL(density, vel)
            v_forward = self.advectVelocitySL(vel, vel)
            d_tmp, v_tmp = self.MacCormackUpdate(density, d_forward, vel, v_forward)
            d_adv, v_adv = self.MacCormackClamp(density, d_forward, d_tmp, vel, v_forward, v_tmp)
        else:
            d_adv = self.advectDensitySL(density, vel)
            v_adv = self.advectVelocitySL(vel, vel)


        return d_adv, v_adv

    def advectDensitySL(self, d_adv, vel):
        if self.dim==2:
            _, res_y, res_x = d_adv.shape
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
            d_adv = d_adv.unsqueeze(0)
            grid_sampled = F.grid_sample(d_adv, backtrace_normed, padding_mode="border", align_corners=True)
            grid_sampled = grid_sampled.squeeze(0)
        else:
            _, res_z, res_y, res_x = d_adv.shape
            z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0., res_z), torch.arange(0., res_y), torch.arange(0., res_x)])
            mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)

            last_vx = 0.5 * (vel[0, :-1, :-1, :-1] + vel[0, :-1, :-1, 1:])
            last_vy = 0.5 * (vel[1, :-1, :-1, :-1] + vel[1, :-1, 1:, :-1])
            last_vz = 0.5 * (vel[2, :-1, :-1, :-1] + vel[2, 1:, :-1, :-1])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            
            backtrace_normed_x = 2 * backtrace[0:1, ...] / (res_x-1) - 1
            backtrace_normed_y = 2 * backtrace[1:2,...] / (res_y-1) - 1
            backtrace_normed_z = 2 * backtrace[2:3,...] / (res_z-1) - 1
            backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y, backtrace_normed_z), dim=0)
            
            permutation = (1,2,3,0)
            backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
            d_adv = d_adv.unsqueeze(0)
            grid_sampled = F.grid_sample(d_adv, backtrace_normed, padding_mode="border", align_corners=True)
            grid_sampled = grid_sampled.squeeze(0)

        return grid_sampled

    def advectVelocitySL(self, v_adv, vel):
        if self.dim==2:
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
            vx = v_adv[0:1, ...]
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
            vy = v_adv[1:2, ...]
            vy = vy.unsqueeze(0)
            grid_sampled = F.grid_sample(vy, backtrace_normed, padding_mode="border", align_corners=True)
            vy_adv = grid_sampled.squeeze(0)

            return torch.cat((vx_adv, vy_adv), dim=0)
        else:
            _, D, H, W = vel.shape
            z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0., D), torch.arange(0., H), torch.arange(0., W)])
            mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)

            # advect vx
            # mesh grid
            last_vx = vel[0, ...]
            pad_y = (1,0,0,1,0,0)
            pad_vy = F.pad(vel[1,...], pad_y)
            last_vy = 0.25 * (pad_vy[:, :-1, :-1] + pad_vy[:, 1:, :-1] + pad_vy[:, :-1, 1:] + pad_vy[:, 1:, 1:])
            pad_z = (1,0,0,0,0,1)
            pad_vz = F.pad(vel[2,...], pad_z)
            last_vz = 0.25 * (pad_vz[:-1, :, :-1] + pad_vz[1:, :, :-1] + pad_vz[:-1, :, 1:] + pad_vz[1:, :, 1:])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            # normalize
            backtrace_normed_x = 2 * backtrace[0:1, ...] / (W-1) - 1
            backtrace_normed_y = 2 * backtrace[1:2, ...] / (H-1) - 1
            backtrace_normed_z = 2 * backtrace[2:3, ...] / (D-1) - 1
            backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y, backtrace_normed_z), dim=0)
            permutation = (1,2,3,0)
            backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
            vx = v_adv[0:1, ...]
            vx = vx.unsqueeze(0)
            grid_sampled = F.grid_sample(vx, backtrace_normed, padding_mode="border", align_corners=True)
            vx_adv = grid_sampled.squeeze(0)

            # advect vy
            pad_x = (0,1,1,0,0,0)
            pad_vx = F.pad(vel[0,...], pad_x)
            last_vx = 0.25 * (pad_vx[:, 1:, 1:] + pad_vx[:, :-1, 1:] + pad_vx[:, 1:, :-1] + pad_vx[:, :-1, :-1])
            last_vy = vel[1, ...]
            pad_z = (0,0,1,0,0,1)
            pad_vz = F.pad(vel[2,...], pad_z)
            last_vz = 0.25 * (pad_vz[1:, 1:, :] + pad_vz[:-1, 1:, :] + pad_vz[1:, :-1, :] + pad_vz[:-1, :-1, :])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_normed_x = 2. * backtrace[0:1, ...] / (W-1) - 1.
            backtrace_normed_y = 2. * backtrace[1:2, ...] / (H-1) - 1.
            backtrace_normed_z = 2. * backtrace[2:3, ...] / (D-1) - 1.
            backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y, backtrace_normed_z), dim=0)
            permutation = (1,2,3,0)
            backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
            vy = v_adv[1:2, ...]
            vy = vy.unsqueeze(0)
            grid_sampled = F.grid_sample(vy, backtrace_normed, padding_mode="border", align_corners=True)
            vy_adv = grid_sampled.squeeze(0)        

            # advect vz
            pad_x = (0,1,0,0,1,0)
            pad_vx = F.pad(vel[0,...], pad_x)
            last_vx = 0.25 * (pad_vx[1:, :, 1:] + pad_vx[:-1, :, 1:] + pad_vx[1:, :, :-1] + pad_vx[:-1, :, :-1])
            pad_y = (0,0,0,1,1,0)
            pad_vy = F.pad(vel[1,...], pad_y)
            last_vy = 0.25 * (pad_vy[1:, 1:, :] + pad_vy[:-1, 1:, :] + pad_vy[1:, :-1, :] + pad_vy[:-1, :-1, :])
            last_vz = vel[2, ...]
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_normed_x = 2. * backtrace[0:1, ...] / (W-1) - 1.
            backtrace_normed_y = 2. * backtrace[1:2, ...] / (H-1) - 1.
            backtrace_normed_z = 2. * backtrace[2:3, ...] / (D-1) - 1.
            backtrace_normed = torch.cat((backtrace_normed_x, backtrace_normed_y, backtrace_normed_z), dim=0)
            permutation = (1,2,3,0)
            backtrace_normed = backtrace_normed.permute(permutation).unsqueeze(0)
            vz = v_adv[2:3, ...]
            vz = vz.unsqueeze(0)
            grid_sampled = F.grid_sample(vz, backtrace_normed, padding_mode="border", align_corners=True)
            vz_adv = grid_sampled.squeeze(0)            

            return torch.cat((vx_adv, vy_adv, vz_adv), dim=0)
    
    def MacCormackUpdate(self, d, d_forward, v, v_forward):
        self.dt *= -1
        d_backward = self.advectDensitySL(d_forward, v)
        v_backward = self.advectVelocitySL(v_forward, v)
        self.dt *= -1
        d_tmp = d_forward + 0.5 * (d - d_backward)
        v_tmp = v_forward + 0.5 * (v - v_backward)

        return d_tmp, v_tmp
    
    def MacCormackClamp(self, d, d_forward, d_tmp, v, v_forward, v_tmp):
        if self.dim == 2:
            _, res_y, res_x = d.shape
            y_pos, x_pos = torch.meshgrid([torch.arange(0, res_y, dtype=torch.float32), torch.arange(0, res_x, dtype=torch.float32)])
            mgrid = torch.stack([x_pos, y_pos], dim=0)
            pad_pos = (0,1,0,1)
            max_pool = F.max_pool2d
        else:
            _, res_z, res_y, res_x = d.shape
            z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0., res_z), torch.arange(0., res_y), torch.arange(0., res_x)])
            mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)
            pad_pos = (0,1,0,1,0,1)
            max_pool = F.max_pool3d

        # clamp density
        if self.dim == 2:
            last_vx = 0.5 * (v[0, :-1, :-1] + v[0, :-1, 1:])
            last_vy = 0.5 * (v[1, :-1, :-1] + v[1, 1:, :-1])
            backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()
            
            d_pad_pos = F.pad(d, pad_pos)
            d_max = max_pool(d_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            d_min = -max_pool(-d_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            d_max_backtrace = d_max[:, backtrace_y, backtrace_x]
            d_min_backtrace = d_min[:, backtrace_y, backtrace_x]
            clamp_max_back = d_tmp > d_max_backtrace
            clamp_min_back = d_tmp < d_min_backtrace
            d_adv = torch.where(clamp_max_back+clamp_min_back, d_forward, d_tmp)
        else:
            last_vx = 0.5 * (v[0, :-1, :-1, :-1] + v[0, :-1, :-1, 1:])
            last_vy = 0.5 * (v[1, :-1, :-1, :-1] + v[1, :-1, 1:, :-1])
            last_vz = 0.5 * (v[2, :-1, :-1, :-1] + v[2, 1:, :-1, :-1])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()
            backtrace_z = torch.clamp(backtrace[2,...], 0, res_z-1).long()
            
            d_pad_pos = F.pad(d, pad_pos)
            d_max = max_pool(d_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            d_min = -max_pool(-d_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            d_max_backtrace = d_max[:, backtrace_z, backtrace_y, backtrace_x]
            d_min_backtrace = d_min[:, backtrace_z, backtrace_y, backtrace_x]
            clamp_max_back = d_tmp > d_max_backtrace
            clamp_min_back = d_tmp < d_min_backtrace
            d_adv = torch.where(clamp_max_back+clamp_min_back, d_forward, d_tmp)

        if self.dim == 2:
            _, res_y, res_x = v.shape
            y_pos, x_pos = torch.meshgrid([torch.arange(0, res_y, dtype=torch.float32), torch.arange(0, res_x, dtype=torch.float32)])
            mgrid = torch.stack([x_pos, y_pos], dim=0)
        else:
            _, res_z, res_y, res_x = v.shape
            z_pos, y_pos, x_pos = torch.meshgrid([torch.arange(0., res_z), torch.arange(0., res_y), torch.arange(0., res_x)])
            mgrid = torch.stack([x_pos, y_pos, z_pos], dim=0)

        # clampy vx
        if self.dim == 2:
            last_vx = v[0, ...]
            pad_y = (1,0,0,1)
            pad_vy = F.pad(v[1,...], pad_y)
            last_vy = 0.25 * (pad_vy[:-1, :-1] + pad_vy[1:, :-1] + pad_vy[:-1, 1:] + pad_vy[1:, 1:])
            backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()

            vx_pad_pos = F.pad(v[0:1,...], pad_pos)
            vx_max = max_pool(vx_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vx_min = -max_pool(-vx_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vx_max_backtrace = vx_max[:, backtrace_y, backtrace_x]
            vx_min_backtrace = vx_min[:, backtrace_y, backtrace_x]
            clamp_max_back = v_tmp[0:1,...] > vx_max_backtrace
            clamp_min_back = v_tmp[0:1,...] < vx_min_backtrace
            vx_adv = torch.where(clamp_max_back+clamp_min_back, v_forward[0:1], v_tmp[0:1])
        else:
            last_vx = v[0, ...]
            pad_y = (1,0,0,1,0,0)
            pad_vy = F.pad(v[1,...], pad_y)
            last_vy = 0.25 * (pad_vy[:, :-1, :-1] + pad_vy[:, 1:, :-1] + pad_vy[:, :-1, 1:] + pad_vy[:, 1:, 1:])
            pad_z = (1,0,0,0,0,1)
            pad_vz = F.pad(v[2,...], pad_z)
            last_vz = 0.25 * (pad_vz[:-1, :, :-1] + pad_vz[1:, :, :-1] + pad_vz[:-1, :, 1:] + pad_vz[1:, :, 1:])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()
            backtrace_z = torch.clamp(backtrace[2,...], 0, res_z-1).long()

            vx_pad_pos = F.pad(v[0:1,...], pad_pos)
            vx_max = max_pool(vx_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vx_min = -max_pool(-vx_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vx_max_backtrace = vx_max[:, backtrace_z, backtrace_y, backtrace_x]
            vx_min_backtrace = vx_min[:, backtrace_z, backtrace_y, backtrace_x]
            clamp_max_back = v_tmp[0:1,...] > vx_max_backtrace
            clamp_min_back = v_tmp[0:1,...] < vx_min_backtrace
            vx_adv = torch.where(clamp_max_back+clamp_min_back, v_forward[0:1], v_tmp[0:1])

        # clampy vy
        if self.dim == 2:
            pad_x = (0,1,1,0)
            pad_vx = F.pad(v[0,...], pad_x)
            last_vx = 0.25 * (pad_vx[1:, 1:] + pad_vx[:-1, 1:] + pad_vx[1:, :-1] + pad_vx[:-1, :-1])
            last_vy = v[1, ...]
            backtrace = mgrid - torch.stack((last_vx, last_vy), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()

            vy_pad_pos = F.pad(v[1:2,...], pad_pos)
            vy_max = max_pool(vy_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vy_min = -max_pool(-vy_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vy_max_backtrace = vy_max[:, backtrace_y, backtrace_x]
            vy_min_backtrace = vy_min[:, backtrace_y, backtrace_x]
            clamp_max_back = v_tmp[1:2,...] > vy_max_backtrace
            clamp_min_back = v_tmp[1:2,...] < vy_min_backtrace
            vy_adv = torch.where(clamp_max_back+clamp_min_back, v_forward[1:2], v_tmp[1:2])
        else:
            pad_x = (0,1,1,0,0,0)
            pad_vx = F.pad(v[0,...], pad_x)
            last_vx = 0.25 * (pad_vx[:, 1:, 1:] + pad_vx[:, :-1, 1:] + pad_vx[:, 1:, :-1] + pad_vx[:, :-1, :-1])
            last_vy = v[1, ...]
            pad_z = (0,0,1,0,0,1)
            pad_vz = F.pad(v[2,...], pad_z)
            last_vz = 0.25 * (pad_vz[1:, 1:, :] + pad_vz[:-1, 1:, :] + pad_vz[1:, :-1, :] + pad_vz[:-1, :-1, :])
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()
            backtrace_z = torch.clamp(backtrace[2,...], 0, res_z-1).long()

            vy_pad_pos = F.pad(v[1:2,...], pad_pos)
            vy_max = max_pool(vy_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vy_min = -max_pool(-vy_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vy_max_backtrace = vy_max[:, backtrace_z, backtrace_y, backtrace_x]
            vy_min_backtrace = vy_min[:, backtrace_z, backtrace_y, backtrace_x]
            clamp_max_back = v_tmp[1:2,...] > vy_max_backtrace
            clamp_min_back = v_tmp[1:2,...] < vy_min_backtrace
            vy_adv = torch.where(clamp_max_back+clamp_min_back, v_forward[1:2], v_tmp[1:2])

        if self.dim == 2:
            return d_adv, torch.cat((vx_adv, vy_adv), dim=0)
        else:
            #clamp vz
            pad_x = (0,1,0,0,1,0)
            pad_vx = F.pad(v[0,...], pad_x)
            last_vx = 0.25 * (pad_vx[1:, :, 1:] + pad_vx[:-1, :, 1:] + pad_vx[1:, :, :-1] + pad_vx[:-1, :, :-1])
            pad_y = (0,0,0,1,1,0)
            pad_vy = F.pad(v[1,...], pad_y)
            last_vy = 0.25 * (pad_vy[1:, 1:, :] + pad_vy[:-1, 1:, :] + pad_vy[1:, :-1, :] + pad_vy[:-1, :-1, :])
            last_vz = v[2, ...]
            backtrace = mgrid - torch.stack((last_vx, last_vy, last_vz), dim=0)*self.dt
            backtrace_x = torch.clamp(backtrace[0,...], 0, res_x-1).long()
            backtrace_y = torch.clamp(backtrace[1,...], 0, res_y-1).long()
            backtrace_z = torch.clamp(backtrace[2,...], 0, res_z-1).long()

            vz_pad_pos = F.pad(v[2:3,...], pad_pos)
            vz_max = max_pool(vz_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vz_min = -max_pool(-vz_pad_pos.unsqueeze(0), kernel_size=2, stride=1).squeeze(0)
            vz_max_backtrace = vz_max[:, backtrace_z, backtrace_y, backtrace_x]
            vz_min_backtrace = vz_min[:, backtrace_z, backtrace_y, backtrace_x]
            clamp_max_back = v_tmp[2:3,...] > vz_max_backtrace
            clamp_min_back = v_tmp[2:3,...] < vz_min_backtrace
            vz_adv = torch.where(clamp_max_back+clamp_min_back, v_forward[2:3], v_tmp[2:3])
            return d_adv, torch.cat((vx_adv, vy_adv, vz_adv), dim=0)


    def applySource(self, density, source):
        d = torch.clamp(density + source, 0., source.max())
        return d

    def addBuoyancy(self, density, vel):
        if self.dim==2:
            _, _, res_x = density.shape
            pad = (0,1,1,1)
            density_pad = F.pad(density, pad)
            scaling = 64. / res_x
            f = 0.5 * (density_pad[:, 1:, :] + density_pad[:, :-1, :]) * scaling
            vel_x = vel[0:1, ...]
            vel_y = vel[1:2, ...] + f * self.dt
            return torch.cat((vel_x, vel_y), dim=0)
        else:
            pad = (0,1,1,1,0,1)
            density_pad = F.pad(density, pad)
            scaling = 64. / self.res_x
            f = 0.5 * (density_pad[..., 1:, :] + density_pad[..., :-1, :]) * scaling
            vel_x = vel[0:1, ...]
            vel_y = vel[1:2, ...] + f * self.dt
            vel_z = vel[2:3, ...]
            return torch.cat((vel_x, vel_y, vel_z), dim=0)

    def addForce(self, vel, force):
        vel = vel + self.dt * force
        return vel

    def solvePressure(self, vel):
        if (self.A==None):
            if self.dim == 2:
                _, res_y, res_x = vel.shape
                Adiag = 4 * torch.ones(res_y-1, res_x-1) + 1e-3
                Aoffd = -1 * torch.ones(res_y-1, res_x-1)
            else:
                _, res_z, res_y, res_x = vel.shape
                Adiag = 6 * torch.ones(res_z-1, res_y-1, res_x-1) + 1e-3
                Aoffd = -1 * torch.ones(res_z-1, res_y-1, res_x-1)

            self.A = torch.stack([Adiag, Aoffd])

        div = self.computeDivergence(vel)
        pressure = self.solvePoisson(div)
        vel = self.correctVelocity(vel, pressure)
        return vel

    def computeDivergence(self, vel):
        if self.dim == 2:
            dudx = (vel[0:1, :-1, 1:] - vel[0:1, :-1, :-1]) / self.dx
            dvdy = (vel[1:2, 1:, :-1] - vel[1:2, :-1, :-1]) / self.dx
            div = dudx + dvdy
        else:
            dudx = (vel[0:1, :-1, :-1, 1:] - vel[0:1, :-1, :-1, :-1]) / self.dx
            dvdy = (vel[1:2, :-1, 1:, :-1] - vel[1:2, :-1, :-1, :-1]) / self.dx
            dwdz = (vel[2:3, 1:, :-1, :-1] - vel[2:3, :-1, :-1, :-1]) / self.dx
            div = dudx + dvdy + dwdz

        return div

    # sparse matrix-vector mult
    def Mv(self, A, b):
        if self.dim == 2:
            Ab = torch.zeros(b.size())
            Adiag = A[0, ...]
            Aoffd = A[1, ...]
            pad_x = (1,1,0,0)
            pad_y = (0,0,1,1)
            b_pad_x = F.pad(b, pad_x)
            b_pad_y = F.pad(b, pad_y)
            Ab = Adiag*b + Aoffd*(b_pad_x[..., 2:] + b_pad_x[..., :-2] + b_pad_y[..., 2:, :] + b_pad_y[..., :-2, :])
        else:
            Ab = torch.zeros(b.size())
            Adiag = A[0, ...]
            Aoffd = A[1, ...]
            pad_x = (1,1,0,0,0,0)
            pad_y = (0,0,1,1,0,0)
            pad_z = (0,0,0,0,1,1)
            b_pad_x = F.pad(b, pad_x)
            b_pad_y = F.pad(b, pad_y)
            b_pad_z = F.pad(b, pad_z)
            Ab = Adiag*b + Aoffd*(b_pad_x[..., 2:] + b_pad_x[..., :-2] + b_pad_y[..., 2:, :] + b_pad_y[..., :-2, :] + b_pad_z[..., 2:, :, :] + b_pad_z[..., :-2, :, :])

        return Ab
    # conjugate gradient (NumCSE notation)
    def solve_lse(self, b, A, acc=1e-5, it_max=1000):
        if self.dim == 2:
            _, res_y, res_x = A.shape
            x = torch.zeros(1, res_y, res_x)
        else:
            _, res_z, res_y, res_x = A.shape
            x = torch.zeros(1, res_z, res_y, res_x)

        r = b - self.Mv(A, x)
        r0_norm = torch.norm(r)
        p = r.clone()
        for t in range(it_max):
            beta = torch.sum(r * r)
            h = self.Mv(A, p)
            alpha = beta / torch.sum(p * h)
            x = x + alpha * p
            r = r - alpha * h
            if (torch.norm(r)) <= acc * r0_norm:
                return x
            beta = torch.sum(r * r) / beta
            p = r + beta * p

        print("solve Poisson failed.")
        return x

    def solvePoisson(self, div, acc = 1e-5, it_max = 1000):
        solve_lse_fcn = self.solve_lse
        class solvePoissonOverwrite(Function):
            @staticmethod
            def forward(ctx, b, A):
                p = solve_lse_fcn(b, A, acc, it_max)
                ctx.save_for_backward(A, p)
                return p
            
            @staticmethod
            def backward(ctx, grad_output):
                A, p = ctx.saved_tensors
                grad_b = solve_lse_fcn(grad_output, A, 1e-10) #chain rule
                grad_Adiag = -grad_b * p #adjoint, dg/dx dx/dA, s.t. A dx/dA=-x
                if self.dim == 2:
                    padx = (0, 1, 0, 0)
                    pady = (0, 0, 0, 1)
                    p_padx = F.pad(p, pad=padx)
                    grad_b_padx = F.pad(grad_b, pad=padx)
                    p_pady = F.pad(p, pad=pady)
                    grad_b_pady = F.pad(grad_b, pad=pady)
                    grad_Aoffd = -grad_b * p_padx[..., 1:] - grad_b_padx[..., 1:] * p - grad_b * p_pady[..., 1:, :] - grad_b_pady[..., 1:, :] * p
                else:
                    padx = (0, 1, 0, 0, 0, 0)
                    pady = (0, 0, 0, 1, 0, 0)
                    padz = (0, 0, 0, 0, 0, 1)
                    p_padx = F.pad(p, pad=padx)
                    grad_b_padx = F.pad(grad_b, pad=padx)
                    p_pady = F.pad(p, pad=pady)
                    grad_b_pady = F.pad(grad_b, pad=pady)
                    p_padz = F.pad(p, pad=padz)
                    grad_b_padz = F.pad(grad_b, pad=padz)
                    grad_Aoffd = -grad_b * p_padx[..., 1:] - grad_b_padx[..., 1:] * p - grad_b * p_pady[..., 1:, :] - grad_b_pady[..., 1:, :] * p - grad_b * p_padz[..., 1:, :, :] - grad_b_padz[..., 1:, :, :] * p

                grad_A = torch.stack([grad_Adiag, grad_Aoffd])
                return grad_b, grad_A

        b = -div / self.dt * self.rho
        solve = solvePoissonOverwrite.apply
        p = solve(b, self.A)

        return p

    def correctVelocity(self, vel, pressure):
        if self.dim == 2:
            pad = (1,1,1,1)
            pad_p = F.pad(pressure, pad)
            diff_px = pad_p[:, 1:, 1:] - pad_p[:, 1:, :-1]
            diff_py = pad_p[:, 1:, 1:] - pad_p[:, :-1, 1:]
            diff_p = torch.cat((diff_px, diff_py), dim=0)
        else:
            pad = (1,1,1,1,1,1)
            pad_p = F.pad(pressure, pad)
            diff_px = pad_p[:, 1:, 1:, 1:] - pad_p[:, 1:, 1:, :-1]
            diff_py = pad_p[:, 1:, 1:, 1:] - pad_p[:, 1:, :-1, 1:]
            diff_pz = pad_p[:, 1:, 1:, 1:] - pad_p[:, :-1, 1:, 1:]
            diff_p = torch.cat((diff_px, diff_py, diff_pz), dim=0)

        vel_corrected = vel - diff_p * self.dt / self.dx
        return vel_corrected

