#
# HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
# at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
# Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
#
import numpy as np

from hilo_mpc import NMPC, SimpleControlLoop, GP, SquaredExponentialKernel, Model, set_plot_backend
from hilo_mpc.library import cstr_seborg


# === Process / Plant Model =======================================================================================
# Equilibria
x0_plant = [0.6205, 348, 347.5]  # initial point
y0 = x0_plant[0]
u0 = x0_plant[2]
x_ref = [0.4674, 355.9, 356]
y_ref = x_ref[0]
u_ref = x_ref[2]

# Constraints
y_lb = 0.35
y_ub = 0.65
u_lb = 335
u_ub = 372

# Simulation parameters
Ts = 0.5  # sampling time (min)
T = 25  # simulation time (min)
N_step = int(T / Ts)  # simulation time steps

# Set up plant
set_plot_backend('bokeh')
plant = cstr_seborg()

# Discretize model
plant.discretize('erk', inplace=True)

# Initialize model
plant.setup(dt=Ts)

# Plant Parameters
p0 = {
    'C_Af': 1,
    'C_p': 0.3,
    'DeltaH_r': 10000,
    'E': 9750 * 8.314,
    'k_0': 6e10,
    'q_0': 10,
    'rho': 1100,
    'tau': 1.5,
    'UA': 70000,
    'V': 150,
    'T_f': 370
}

# Initial condition
plant.set_initial_conditions(x0=x0_plant)
plant.set_initial_parameter_values(p0)

# === MPC with true dynamics ======================================================================================
nmpc = NMPC(plant)

# Set horizon
nmpc.horizon = 5

# Set cost function ------------------------------
# ...with scaling
# nmpc.set_scaling(x_scaling=[0.5, 350, 350], u_scaling=[350])
# nmpc.quad_stage_cost.add_states(names=['C_A'], weights=[100], trajectory_tracking=True)
# nmpc.quad_stage_cost.add_inputs(names=['T_cr'], weights=[100], trajectory_tracking=True)
# nmpc.quad_terminal_cost.add_states(names=['C_A'], weights=[100], trajectory_tracking=True)

# ...without scaling
nmpc.quad_stage_cost.add_states(names=['C_A'], weights=[100], trajectory_tracking=True)
nmpc.quad_stage_cost.add_inputs(names=['T_cr'], weights=[0.0002], trajectory_tracking=True)
nmpc.quad_terminal_cost.add_states(names=['C_A'], weights=[100], trajectory_tracking=True)
# -------------------------------------------------

# Set constraints
nmpc.set_box_constraints(x_lb=[y_lb, 0, 0], x_ub=[y_ub, 500, 500], u_lb=u_lb, u_ub=u_ub)

# Initial guess
nmpc.set_initial_guess(x_guess=x0_plant, u_guess=u0)

# Set up controller
nmpc.setup(options={'print_level': 0, 'objective_function': 'discrete', 'integration_method': 'discrete'})

# Reference trajectory
ref_trajectory = np.zeros([2, N_step + nmpc.horizon + 1])
ref_trajectory[0, :10] = y0
ref_trajectory[0, 10:] = y_ref
ref_trajectory[1, :10] = u0
ref_trajectory[1, 10:] = u_ref
ref_trajectory_y = ref_trajectory[0, :]
ref_trajectory_u = ref_trajectory[1, :]

# Run the simulation
scl = SimpleControlLoop(plant, nmpc)
scl.run(N_step, p=plant.solution.get_by_id('p:0'), ref_sc={'T_cr': ref_trajectory_u, 'C_A': ref_trajectory_y},
        ref_tc={'C_A': ref_trajectory_y})

# Plots
scl.plot(
    ('t', 'T_cr'),
    ('t', 'C_A'),
    title=('input', 'output'),
    figsize=(900, 300),
    layout=(2, 1),
    background_fill_color='#fafafa',
    output_file='gp_nmpc_cstr_output_feedback/nominal_controller.html'
)

# === GP training data ============================================================================================
# Input sequence composition:
# - 5 small chirps
# - 2 large chirps
chirps = {
    'type': 'linear',
    'amplitude': [6, 6, 6, 6, 6, 18, 18],
    'length': [996, 996, 996, 996, 996, 5 * 996, 5 * 996],
    'mean': [341, 347, 353, 359, 365, 353, 353],
    'chirp_rate': [0.00015, 0.00015, 0.00015, 0.00015, 0.00015, 0.000015, 0.000035]
}
# Standard deviation for noise added to C_A
sigma_n = 5e-3
# Generate data
data_set = plant.generate_data('chirp', shift=2, add_noise={'std': sigma_n, 'seed': 10}, chirps={'T_cr': chirps},
                               skip=('T', 'T_c'))
data_set.plot_raw_data(
    ('t', 'T_cr'),
    ('t', 'C_A_k+1', 'C_A_k', 'C_A_k-1', 'C_A_k-2'),
    title=('input', 'output'),
    figsize=(900, 300),
    layout=(2, 1),
    background_fill_color="#fafafa",
    output_file='gp_nmpc_cstr_output_feedback/raw_data.html'
)

# data_set.add_noise('C_A_k+1', seed=10, std=sigma_n)
data_set.plot_raw_data(
    ('t', 'C_A_k+1_noisy'),
    title="output (noisy)",
    figsize=(900, 300),
    background_fill_color="#fafafa",
    output_file='gp_nmpc_cstr_output_feedback/raw_noisy_data.html'
)

# Training data reduction
data_set.select_train_data('euclidean_distance', distance_threshold=.59)
data_set.plot_train_data(
    ('t', 'T_cr'),
    ('t', 'C_A_k+1_noisy'),
    title=('input', 'output'),
    figsize=(900, 300),
    layout=(2, 1),
    background_fill_color="#fafafa",
    output_file='gp_nmpc_cstr_output_feedback/train_data.html'
)

# Select test data
data_set.select_test_data('downsample', downsample_factor=20)
data_set.plot_test_data(
    ('t', 'T_cr'),
    ('t', 'C_A_k+1_noisy'),
    title=('input', 'output'),
    figsize=(900, 300),
    layout=(2, 1),
    background_fill_color="#fafafa",
    output_file='gp_nmpc_cstr_output_feedback/test_data.html'
)

# Initialize GP
kernel = SquaredExponentialKernel(active_dims=[1, 2, 3], ard=True)
gp = GP(['x_0', 'x_1', 'x_2', 'u'], ['y'], kernel=kernel, noise_variance=sigma_n ** 2)
gp.noise_variance.fixed = True

# Training of GP
train_in, train_out = data_set.train_data
gp.set_training_data(train_in, train_out)
gp.setup()
gp.fit_model()

# Validate GP model
test_in, test_out = data_set.test_data
gp.plot_prediction_error(
    test_in,
    test_out,
    figsize=(900, 300),
    layout=(2, 1),
    background_fill_color="#fafafa",
    output_file='gp_nmpc_cstr_output_feedback/gp_prediction_error.html'
)
gp.plot(
    test_in,
    background_fill_color="#fafafa",
    xlabel=('y(k)', 'y(k-1)', 'y(k-2)', 'u(k)'),
    ylabel=4 * ('y(k+1)', ),
    output_file='gp_nmpc_cstr_output_feedback/gp_train_test_scatter.html'
)

# === GP prediction models ========================================================================================
# Initialize empty model
gp_model = Model(plot_backend='bokeh', name='GP_Model', discrete=True)

# Set states and inputs
gp_model.set_dynamical_states('x', 3)
gp_model.set_inputs('u', 1)

# States
# x_0(k) := y(k)
# x_1(k) := y(k-1)
# x_2(k) := y(k-2)

# Set difference equations
# x_0(k+1) = y(k+1) = GP
# x_1(k+1) = y(k)   = x_0(k)
# x_2(k+1) = y(k-1) = x_1(k)
gp_model.set_dynamical_equations(['0', 'x_0', 'x_1'])

# Add GP to the ODE's model --> y(k+1) = GP
gp_model += [gp, 0, 0]

# Set up the model
gp_model.setup(dt=Ts)

# Set initial conditions
x0_GP = [y0, y0, y0]
gp_model.set_initial_conditions(x0=x0_GP)

# === GP-MPC (model-plant mismatch) ===============================================================================
# Build controller
gp_nmpc = NMPC(gp_model)

# Set horizon
gp_nmpc.horizon = 5

# Set cost function
weight_x0 = 10
weight_u = weight_x0 / 15e3
gp_nmpc.quad_stage_cost.add_states(names=['x_0'], weights=[weight_x0], trajectory_tracking=True)
gp_nmpc.quad_stage_cost.add_inputs(names=['u'], weights=[weight_u], trajectory_tracking=True)
gp_nmpc.quad_terminal_cost.add_states(names=['x_0'], weights=[weight_x0], trajectory_tracking=True)

# Set constraints
gp_nmpc.set_box_constraints(x_lb=[y_lb, y_lb, y_lb], x_ub=[y_ub, y_ub, y_ub], u_lb=u_lb, u_ub=u_ub)

# Initial guess
gp_nmpc.set_initial_guess(x_guess=x0_GP, u_guess=u0)

# Set up controller
gp_nmpc.setup(options={'print_level': 1, 'objective_function': 'discrete', 'integration_method': 'discrete'})

# Copy previous solution for comparison
nominal_solution = plant.solution.copy()

# Run the simulation
# NOTE: SimpleControlLoop won't work here, since the model used in the MPC differs too much from the plant model.
#  Maybe in future version an advanced control loop class will be available.
plant.reset_solution()
xk = x0_GP
for _ in range(N_step):
    u = gp_nmpc.optimize(x0=xk, ref_sc={'u': ref_trajectory[1, :], 'x_0': ref_trajectory[0, :]},
                         ref_tc={'x_0': ref_trajectory[0, :]})
    plant.simulate(u=u)
    xk = [plant.solution.get_by_id('x:f')[0], xk[0], xk[1]]  # xk = [y_k, y_{k-1}, y_{k-2}]

# Plots
plant.solution.plot(
    ('t', 'T_cr'),
    ('t', 'C_A'),
    data=nominal_solution,
    data_suffix='nominal',
    title=('input', 'output'),
    figsize=(900, 300),
    layout=(2, 1),
    line_width=2,
    background_fill_color='#fafafa',
    output_file='gp_nmpc_cstr_output_feedback/gp_mpc_vs_nominal_mpc.html'
)
