import torch
import matplotlib.pyplot as plt

res_x = torch.tensor(64)
res_y = res_x
fig, ax = plt.subplots()
s = plt.imread("s.png")
s = s[..., 3]
s = torch.from_numpy(s)
s = torch.flip(s, [0])
s = s.view(1, 1, s.shape[0], s.shape[1])

d = torch.linspace(-1, 1, 24)
meshx, meshy = torch.meshgrid((d, d), indexing="ij")
grid = torch.stack((meshy, meshx), 2)
grid = grid.unsqueeze(0)
output = torch.nn.functional.grid_sample(s, grid)

target = torch.zeros((1,res_y,res_x))
target[0,-24:, 20:44] = output[0,0]
ax.imshow(target[0], origin="lower")
torch.save(target, "s.pt")
plt.show()