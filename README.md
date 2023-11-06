# PBS_project
Differentiable Physics Solver



## Goal

Compare the adjoint method with automatic differentiation for optimization or control problems.

## Experiments

**2D Burgers' equation**: The 2D Burgers' equation is a fundamental partial differential equation from fluid mechanics. It can model various physical phenomena, including shock waves and turbulence, and it is nonlinear, which makes it interesting for such a study.

The 2D Burgers' equation in its viscous form is:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$




where $u$ and $v$ are the velocity components in the $x$ and $y$ directions, respectively, $\nu$ is the kinematic viscosity, and $t$ is time.



**Methods:**

- Adjoint methods (AJ)

  - Formulate the adjoint problem for the 2D Burgers' equation. This involves defining the adjoint equations, which will be the PDEs solved backward in time.
  - Implement the adjoint solver, ensuring it calculates the gradients with respect to the parameters.

- Automatic Differentiation (AD): Finite element method and finite difference method.

  - Utilize PyTorch's automatic differentiation capabilities to compute gradients of the objective function with respect to the parameters of interest. (**End to End**).

  

**Objective Function**: Define an objective function that you wish to minimize or maximize

-  velocities at a certain point in time and space
- Integrated quantity over the whole domain or a part of it.
- **Optimization of initial conditions that lead to a certain desired velocity profile at a later time.**



**Optimization:**

- Gradient-based method



**Evaluation:**

- Compare the performance of the automatic differentiation and adjoint method implementations in terms of 

  - computation time
  - memory usage
  - accuracy of the gradients
  - The effectiveness of the optimization.

  

## Time Line

| Time Line   | AJ                              | AD                                 |
| ----------- | ------------------------------- | ---------------------------------- |
| 6 Nov - 12  | study the method                | prepare for the weak form residual |
| 13 Nov - 19 | Finish the code (forward)       | Finish the code (forward)          |
| 20 Nov - 26 | Test the experiments (backward) | Test the experiments (backward)    |
| 27 Nov - 30 | Compare the results             | Compare the results                |



