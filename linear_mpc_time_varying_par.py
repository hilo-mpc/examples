from hilo_mpc import Model, LMPC
import casadi as ca
import numpy as np
"""
This is an example of linear discrete-time MPC. 
In this example we use a bike model. The bike model is discretized and then linearized around a steady-state.
Some of the parameters of the model are assumed to be time-varying.  
"""
model = Model(plot_backend='bokeh')

states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
inputs = model.set_inputs(['a', 'delta'])
parameters = model.set_parameters(['lr', 'lf'])

# Unwrap states
v = states[2]
phi = states[3]

# Unwrap states
a = inputs[0]
delta = inputs[1]

# Unwrap Parameters
lr = parameters[0]
lf = parameters[1]

beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

# ODE
dpx = v * ca.cos(phi + beta)
dpy = v * ca.sin(phi + beta)
dv = a
dphi = v / lr * ca.sin(beta)

model.set_dynamical_equations([dpx, dpy, dv, dphi])
model.discretize(method='rk4', inplace=True)
model = model.linearize()
dt = 0.05
lr0 = 1.4  # [m]
lf0 = 1.8  # [m]
model.setup(dt=dt)
model.set_initial_parameter_values(p=[lr0, lf0])
model.set_equilibrium_point(x_eq=[0, 0, 0, 0], u_eq=[0, 0])

mpc = LMPC(model)
mpc.horizon = 10
mpc.Q = np.eye(model.n_x)
mpc.R = np.eye(model.n_u)
mpc.set_time_varying_parameters(names=['lf'])
mpc.setup()
x0 = [1, 1, 0, 0]
lr0 = 1.4  # [m]
lf0 = 1.8  # [m]

mpc.optimize(x0=x0, cp=[lr0], tvp={'lf': [1.8, 1.8, 1.8, 1.8, 1.8, 1.4, 1.4, 1.4, 1.4, 1.4]})
mpc.solution.plot()