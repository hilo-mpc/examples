from hilo_mpc import Model, SimpleControlLoop, LMPC
import numpy as np

"""
This is an example of a simple linear discrete model predictive controller 
"""
x0 = [1, 1]
model = Model(plot_backend='bokeh', discrete=True)

model.A = np.array([[1, 0.5], [0, 1]])
model.B = np.array([[0.5 ** 2 / 2], [0.5]])

model.setup(dt=0.5)
model.set_initial_conditions(x0=x0)

mpc = LMPC(model)
mpc.Q = np.eye(2) # stage cost matrix
mpc.R = 1 # Input cost matrix
mpc.P = np.eye(2) # terminal cost matrix
mpc.horizon = 10
mpc.set_box_constraints(x_lb=[-5, -5], x_ub=[5, 5], u_lb=[-1], u_ub=[1])
mpc.setup()

scl = SimpleControlLoop(model,mpc)
scl.run(200)
scl.plot()