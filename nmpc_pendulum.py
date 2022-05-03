"""
HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
"""

import numpy as np

from hilo_mpc import Model, NMPC, SimpleControlLoop, set_plot_backend


# matplotlib.use('Qt5Agg')


# Set plot backend
set_plot_backend('matplotlib')

# Initialize empty model
model = Model()

# Define system
equations = """
# Constants
M = 5
m = 1
l = 1
h = 0.5
g = 9.81

# DAE
dx(t)/dt = v(t)
dv(t)/dt = 1/(M + m - 3/4*m*cos(theta(t))^2) * (3/4*m*g*sin(theta(t))*cos(theta(t)) ...
- 1/2*m*l*sin(theta(t))*omega(t)^2 + F(k))
d/dt(theta(t)) = omega(t)
d/dt(omega(t)) = 3/(2*l) * (dv(t)/dt*cos(theta(t)) + g*sin(theta(t)))
0 = h + l*cos(theta(t)) - y(t)
"""
model.set_equations(equations)

# Set up model
model.setup(dt=.1)

# Initial conditions
x0 = [2.5, 0., 1.5, 0.]
z0 = np.sqrt(3.) / 2.
model.set_initial_conditions(x0=x0, z0=z0)

# Initialize NMPC
nmpc = NMPC(model)

# Quadratic stage cost
nmpc.quad_stage_cost.add_states(names=['v', 'theta'], ref=[0, 0], weights=[10, 5])
nmpc.quad_stage_cost.add_inputs(names='F', weights=0.1)

# NMPC horizon length
nmpc.horizon = 25

# NMPC boxed constraints
nmpc.set_box_constraints(x_ub=[5, 10, 10, 10], x_lb=[-5, -10, -10, -10])

# Initial guess
nmpc.set_initial_guess(x_guess=x0, u_guess=0.)

# Set up NMPC
nmpc.setup(options={'print_level': 0})

# Create default control loop
control_loop = SimpleControlLoop(model, nmpc)

# Run control loop
control_loop.run(100, live_animation=True)
