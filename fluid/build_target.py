import torch
import matplotlib.pyplot as plt

res_x = torch.tensor(64)
res_y = torch.tensor(64)
s = plt.imread("data/eth.png")
s = s[..., 3]
s = torch.from_numpy(s)
s = torch.flip(s, [0])
s = s.view(1, 1, s.shape[0], s.shape[1])

dx = torch.linspace(-1, 1, 48)
dy = torch.linspace(-1, 1, 24)
meshx, meshy = torch.meshgrid((dy, dx))
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0)
output = torch.nn.functional.grid_sample(s, grid)

target = torch.zeros((1,res_y,res_x))
target[0,-30:-6, 8:56] = output[0,0]
torch.save(target, 'data/eth.pt')
plt.imshow(target[0], origin="lower")
plt.show()