import numpy as np

from hilo_mpc import Model, NMPC, SimpleControlLoop, GPArray, Mean, Kernel, set_plot_backend


# Set plot backend
set_plot_backend('bokeh')

# Initialize empty model
model = Model()

# Add equations
equations = """
dx1/dt = u1(k)*cos(x3(t))
dx2/dt = u1(k)*sin(x3(t))
dx3/dt = u2(k)
"""
model.set_equations(equations=equations)

# Sampling time
dt = 0.1

# Set up model
model.setup(dt=dt)

# Initial conditions
x0 = [1, 1, 0]
model.set_initial_conditions(x0)

# Generate data
Tf = 20
N = int(Tf / dt)
time = np.linspace(0, Tf, N)

u1 = 2 + 0.1 * np.sin(time)
u2 = 0.5 + np.sin(time)

for i in range(N):
    model.simulate(u=[u1[i], u2[i]])

# Plot results
model.solution.plot()

# Generate features and labels
features = 3 * [model.solution['t'].full()]
labels = model.solution.get_by_id('x').full()

# Initialize GPs
gps = GPArray(3)
for k, gp in enumerate(gps):
    # Define the kernel
    if k < 2:
        mean = Mean.zero()
        kernel = Kernel.periodic(length_scales=.1)
    else:
        mean = Mean.linear()
        kernel = Kernel.periodic(length_scales=.1)

    # Define kth GP and fit it
    gp.initialize(['t'], [f'x{k + 1}'], mean=mean, kernel=kernel)
    gp.set_training_data(features[k], np.atleast_2d(labels[k, :]))
    gp.setup()
    gp.fit_model()

    # Plots
    gp.plot_1d()

# Set up MPC
model.reset_solution()

nmpc = NMPC(model)
t = nmpc.get_time_variable()

gp_1_mean, _ = gps[0].predict(t)
gp_2_mean, _ = gps[1].predict(t)
gp_3_mean, _ = gps[2].predict(t)

nmpc.horizon = 50
nmpc.quad_stage_cost.add_states(names=['x1', 'x2', 'x3'],
                                ref={'x1': gp_1_mean, 'x2': gp_2_mean, 'x3': gp_3_mean},
                                trajectory_tracking=True,
                                weights=[10, 10, 10])
nmpc.quad_terminal_cost.add_states(names=['x1', 'x2', 'x3'],
                                   ref={'x1': gp_1_mean, 'x2': gp_2_mean, 'x3': gp_3_mean},
                                   weights=[10, 10, 10],
                                   trajectory_tracking=True)
nmpc.setup(options={'objective_function': 'discrete'})

# Run the simulation
sloop = SimpleControlLoop(model, nmpc)
sloop.run(N - nmpc.horizon - 1)

# Plots
sloop.plot()
