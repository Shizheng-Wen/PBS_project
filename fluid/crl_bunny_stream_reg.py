import torch
from solver import Solver
from models import StreamModel
import time

def get_loss(model, target, keyframe, loss_fn):
        model.compute_force()
        density = torch.zeros((1, model.solver.res_z, model.solver.res_y, model.solver.res_x))
        vel = torch.zeros((3, model.solver.res_z+1, model.solver.res_y+1, model.solver.res_x+1))
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
res_z = int(res_x)
solver = Solver((res_x, res_y, res_z))

source = torch.load('data/bunny.pt')
source_time = 1
mac_on = False
model = StreamModel(solver, source, source_time, mac_on)

keyframe = 40
target = torch.load('data/crl.pt')

alpha = 0.1
learning_rate = 1.
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, history_size=10, line_search_fn='strong_wolfe')

# train
start = time.time()
epochs = 100
losses = []
for t in range(epochs):
    train_loop(model, target, keyframe, loss_fn, alpha, optimizer)
    loss_k, loss_f = get_loss(model, target, keyframe, loss_fn)
    loss = loss_k + loss_f
    losses.append([loss_k.item(), loss_f.item()])
    print(t, ', loss: ', loss_k.item(), ', ', loss_f.item())
    torch.save((model.state_dict(), keyframe), 'crl_bunny_res.pt')
    torch.save(losses, 'crl_bunny_loss.pt')

end = time.time()
print('time: ', end-start, ' s')

