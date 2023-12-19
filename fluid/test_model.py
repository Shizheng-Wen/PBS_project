"""Used to test model based on static force field
"""

import torch
from solver import Solver
from models import Model
import matplotlib.pyplot as plt
import time

def get_loss(model, target, keyframe, loss_fn):
    """get loss for keyframe matching and smoothness
       need to generate target.pt and put it in data/
    """
    model.compute_force()
    density = torch.zeros((1, model.solver.res_y, model.solver.res_x))
    vel = torch.zeros((2, model.solver.res_y+1, model.solver.res_x+1))
    for frame in range(keyframe):
        density, vel = model(density, vel, frame)

    loss_k = loss_fn(density, target)
    loss_f = loss_fn(model.force, torch.zeros(model.force.shape))
    return loss_k, loss_f

def train_loop(model, target, keyframe, loss_fn, alpha, optimizer):
    def closure():
        optimizer.zero_grad()
        loss_k, loss_f = get_loss(model, target, keyframe, loss_fn)
        loss = loss_k + alpha * loss_f
        loss.backward()
        return loss
    optimizer.step(closure)



res_x = int(64)
res_y = int(res_x)
solver = Solver((res_x, res_y))

source = torch.zeros((1, res_y, res_x))
source[0, 4:8, 28:36] = 1.
source_time = 60
mac_on = False
model = Model(solver, source, source_time, mac_on)

keyframe = 60
target = torch.load('data/target.pt')

alpha = 100.
learning_rate = 1.
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size=10, line_search_fn='strong_wolfe')

# train
start = time.time()
epochs = 10
losses = []
for t in range(epochs):
    train_loop(model, target, keyframe, loss_fn, alpha, optimizer)
    loss_k, loss_f = get_loss(model, target, keyframe, loss_fn)
    loss = loss_k + loss_f
    losses.append(loss.item())
    print(t, ', loss: ', loss_k.item(), ', ', loss_f.item())
end = time.time()
print('time: ', end-start, ' s')

# save results
# torch.save((model.state_dict(), keyframe), 'res.pt')
# torch.save(losses, 'loss.pt')

# or load
# res = torch.load('res.pt')
# force, keyframe = res
# model.load_state_dict(force)

density = torch.zeros((1, res_y, res_x))
vel = torch.zeros((2, res_y+1, res_x+1))
model.compute_force()
for f in range(keyframe):
    if f < keyframe:
        density, vel = model(density, vel, f)
plt.imshow(1-density[0].detach().numpy(), origin="lower", cmap='Greys',  interpolation='nearest')
plt.axis('off')
plt.show()
    # plt.savefig('{:02d}.png'.format(f), dpi=120)

