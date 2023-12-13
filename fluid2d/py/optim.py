import torch
from model import Model
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from PyTorch_LBFGS import FullBatchLBFGS

class TotoalVariation(torch.nn.Module):
    def forward(self, input):
        w_variance = torch.sum(torch.pow(input[..., :-1] - input[..., 1:], 2))
        h_variance = torch.sum(torch.pow(input[..., :-1, :] - input[..., 1:, :], 2))
        return w_variance + h_variance
    
class Optimization(object):
    def __init__(self, solver) -> None:
        self.solver = solver
        self.density_init = torch.zeros((1, self.solver.res_y, self.solver.res_x))
        self.vel_init = torch.zeros((2, self.solver.res_y+1, self.solver.res_x+1))
        source = (.3, .7, .1, .3)
        source_time = 1
        self.model = Model(self.solver, source, source_time)
        self.objective_pointwise = torch.nn.MSELoss(reduction="mean")
        self.objective_list = [self.objective_pointwise]
        self.keyframe = 90
        target = torch.load("s.pt")
        self.targets = {}
        self.targets.update({self.keyframe: target})
        self.tv = TotoalVariation()
        self.lambd_tv_reg = 10.0
        self.lr = 1

    def compute_objective(self, phase):
        self.objective = 0.0
        self.tv_objective = 0.0
        density = self.density_init.detach()
        vel = self.vel_init.detach()

        self.model.compute_force()
        for t in range(self.keyframe):
            density, vel = self.model(density, vel, t)
            frame = t + 1
            if frame in list(self.targets.keys()):
                target = self.targets[frame]
                self.objective += self.objective_list[phase](density, target)

        for force in self.model.get_force():
            self.tv_objective += self.tv(force)

        self.objective = (self.objective + self.lambd_tv_reg * self.tv_objective)
        return self.objective

    def get_current_objective(self):
        objective_dict = {"total_objective": self.objective.item()}
        objective_dict.update({"tv_objective": self.tv_objective.item()})
        return objective_dict
    
    def set_optimization(self, phase):
        self.optimizer = FullBatchLBFGS(self.model.parameters(), lr=self.lr, line_search="Wolfe", history_size=10)

    def optimize_parameters(self, phase, is_first_step=False):
        def closure_pytorch_lbfgs():
            self.optimizer.zero_grad()
            self.compute_objective(phase)
            return self.objective
        if is_first_step:
            closure_pytorch_lbfgs()
            self.objective.backward()
        options = {
            "closure": closure_pytorch_lbfgs,
            "current_objective": self.objective,
            "max_ls": 25,
            "inplace": False,
        }
        _, _, t, _, F_eval, G_eval, _, fail = self.optimizer.step(options)
        message = (
            "# fcn eval: {}, # grad eval: {}, "
            "final step size: {}, failure of line search: {}".format(
                F_eval, G_eval, t, fail
            )
        )
        return message, fail
    
    def simulate(self):
        self.model.compute_force()
        density = self.density_init.detach()
        vel = self.vel_init.detach()
        force_rescale_factor = 1.0
        fig, ax = plt.subplots()
        for t in range(self.keyframe):
            frame = t + 1
            density, vel = self.model(density, vel, t, force_rescale_factor)
            ax.imshow(density[0].detach(), origin='lower')
            plt.savefig("imgs/{:03d}.png".format(t), dpi=150)
            if frame in [self.keyframe]:
                ax.imshow(density[0].detach(), origin='lower')
                plt.savefig("keyframe.png", dpi=150)

    
    
