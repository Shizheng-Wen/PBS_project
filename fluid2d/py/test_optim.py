from solver import Solver
from optim import Optimization
import torch
import matplotlib.pyplot as plt 

s = Solver()
optimization = Optimization(s)
optimization.set_optimization(0)
    
loss = []
for i in range(100):
    message, fail = optimization.optimize_parameters(0, is_first_step=(i==0))
    objective_dict = optimization.get_current_objective()
    loss.append(objective_dict["total_objective"])
    print(i, ": ", message, ", Loss: ", objective_dict["total_objective"])
torch.save(optimization.model.state_dict(), "force.pth")
plt.plot(loss)
plt.show()