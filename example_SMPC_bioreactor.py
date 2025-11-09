import casadi as ca
from hilo_mpc import SMPC, NMPC, GPArray, Kernel
from hilo_mpc.library.models import ecoli_D1210_conti as conti_plant
import numpy as np

"""
This is the example of a stochastic MPC applied on a bioreactor.
"""


plant = conti_plant(model='complex')

model = plant.copy(setup=False)
model.discretize('erk', order=1, inplace=True)

# Initial conditions
# It is important that the initial conditions are not zero, because otherwise we run into infeasibility
# since the lower bound on the concentrations is also zero.
X0 = 1
S0 = 40.
P0 = 2.
I0 = 5
ISF0 = 1.
IRF0 = 0.1

x0 = [X0, S0, P0, I0, ISF0, IRF0]

# Sampling times
dt = 1

plant.setup(dt=dt)
model.setup(dt=dt)

plant.set_initial_conditions(x0=x0)
model.set_initial_conditions(x0=x0)

Bw = ca.DM.eye(plant.n_x)
features_names = ['S', 'I']

cp = [100, 5]
"""
Generate datasets
To do that I am using an MPC 
"""
nmpc = NMPC(model)
nmpc.horizon = 10
nmpc.quad_terminal_cost.add_states(names='P', weights=10, ref=2)
nmpc.quad_stage_cost.add_states(names='P', weights=10, ref=2)
nmpc.set_box_constraints(x_lb=[0., 0., 0., 0., 0, 0], u_lb=[0, 0])
nmpc.set_initial_guess(x_guess=x0, u_guess=[0, 0])
nmpc.setup()

# Run one batch to collect data with the learning-free model
n_steps = 20
n_gps = Bw.shape[1]
w = ca.DM.zeros((n_gps, n_steps))
for i in range(n_steps):
    u = nmpc.optimize(x0=x0, cp=cp)
    plant.simulate(u=u, p=cp)
    model.reset_solution()
    model.set_initial_conditions(plant.solution['x'][:, -2])
    model.simulate(u=u, p=cp)
    w[:, i] = ca.mtimes(np.linalg.pinv(Bw), (model.solution['x:f'] - plant.solution['x:f']))
    x0 = plant.solution['x:f']

x0 = [X0, S0, P0, I0, ISF0, IRF0]
u0 = [0.1, 0.1]

# Initialize GPs
gps = GPArray(6)
for k, gp in enumerate(gps):
    train_in = ca.vertcat(plant.solution.get_by_name('S')[:, 1:],
                          plant.solution.get_by_name('I')[:, 1:])
    train_out = w[k, :]

    kernel = Kernel.squared_exponential(length_scales=2.)

    # Define the GP
    gp.initialize(features_names, f'w{k}', kernel=kernel, noise_variance=1e-6, solver='Nelder-Mead')
    gp.noise_variance.fixed = True
    gp.set_training_data(np.array(train_in), np.atleast_2d(train_out))
    gp.setup()
    gp.fit_model()

cov_x0 = np.zeros((model.n_x, model.n_x))
Kgain = np.zeros((model.n_u, model.n_x))

smpc = SMPC(model, gps, Bw)
smpc.horizon = 10
smpc.quad_stage_cost.add_states(names='P', weights=10, ref=10)
smpc.quad_terminal_cost.add_states(names='P', weights=10, ref=10)
smpc.set_box_chance_constraints(x_lb=[0, 0, 0, 0, 0, 0],
                                x_ub=[100, 100, 100, 100, 100, 100],
                                u_lb=[0, 0],
                                x_lb_p=[0.9 for _ in range(model.n_x)],
                                x_ub_p=[0.9 for _ in range(model.n_x)])
smpc.setup(options={'ipopt_debugger': True})
smpc.optimize(x0=x0, cov_x0=cov_x0, cp=cp, Kgain=Kgain)
smpc.plot_iterations(plot_last=True)
smpc.plot_prediction()
