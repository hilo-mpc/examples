"""
In this example a Van der Pol oscillator is driven to the origin
(http://casadi.sourceforge.net/users_guide/html/node8.html)

    minimize    \int_{t=0}^T (x_0^2 + x_1^2 + u^2) dt
      x,u

    s.t.        \dot{x}_0 = (1 - x_1^2)*x_0 - x_1 + u
                \dot{x}_1 = x_0                         for 0<=t<=T
                -1.0<=u<=1.0,   x_1>=-.25
                x_0(0) = 0, x_1(0) = 1
"""

import numpy as np

from hilo_mpc import Model, OCP, SimpleControlLoop, set_plot_backend


# Set plot backend
set_plot_backend('matplotlib')

# Initialize empty model
model = Model()

# Add state and inputs
model.set_dynamical_states('x', 2)
model.set_inputs('u')

# Add dynamical equations
model.set_dynamical_equations(['(1 - x_1^2)*x_0 - x_1 + u', 'x_0'])

# Set up model
model.setup(dt=.5)

# Initial conditions
x0 = [0., 1.]
model.set_initial_conditions(x0)

# Initialize OCP
ocp = OCP(model)

# Quadratic stage cost
ocp.quad_stage_cost.add_states(['x_0', 'x_1'], [1., 1.])
ocp.quad_stage_cost.add_inputs('u', 1.)

# OCP horizon length
ocp.horizon = 20

# OCP boxed constraints
ocp.set_box_constraints(x_lb=[-np.inf, -.25], u_ub=1., u_lb=-1.)

# Initial guess
ocp.set_initial_guess(x_guess=x0)

# Set up NMPC
ocp.setup(options={'print_level': 0})

# Create default control loop
control_loop = SimpleControlLoop(model, ocp)

# Run control loop
control_loop.run(20)

# Plots
control_loop.plot()
