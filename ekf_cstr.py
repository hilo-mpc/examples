"""
HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
"""
import numpy as np

from hilo_mpc import EKF, set_plot_backend
from hilo_mpc.library import cstr_schaffner_and_zeitz


# Load CSTR model
set_plot_backend('bokeh')
model = cstr_schaffner_and_zeitz()

# Set up model
model.setup(dt=.1)

# Initialize Kalman filter
ekf = EKF(model)

# Set up Kalman filter
ekf.setup()

# Constants
const = {
    'a_1': .2674,
    'a_2': 1.815,
    'b_1': 1.05e14,
    'b_2': 4.92e13,
    'g': 1.5476,
    'E': 34.2583
}

# Initial conditions of the model
model.set_initial_conditions([.5, 0.])
model.set_initial_parameter_values(const)

# Initial guess for the Kalman filter
ekf.R = 1e-4
ekf.set_initial_guess([0., 0.], P0=[.25, .25])
ekf.set_initial_parameter_values(const)

# Seed for noise
np.random.seed(0)

# Run simulations
for _ in range(200):
    model.simulate(u=-.002)

    # Get noisy measurement and calculate estimates
    yk = model.solution.make_some_noise('y:f', var={'y': 1e-4})
    ekf.estimate(y=yk, u=-.002)

# Plots
model.solution.plot(
    ('t', 'x_1'),
    ('t', 'x_2'),
    ('t', 'y', 'y_noisy'),
    data=ekf.solution,
    data_suffix='pred',
    data_skip=2,
    title=("conversion of chemical reaction", "scaled reactor temperature", "measured temperature"),
    interactive=True,
    output_file='ekf_cstr/actual_vs_predicted.html'
)
ekf.solution.plot(('t', 'P_0', 'P_3'), output_file='ekf_cstr/error_variance.html')
