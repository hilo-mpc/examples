#
# HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
# at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
# Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
#
"""
This example solves a path following MPC and a trajectory tracking MPC problem for a simple double integrator system.
In this example we will not use the SimpleControlLoop class, but we will write the control loop manually.
"""

from bokeh.io import show
from bokeh.plotting import figure
import casadi as ca
import numpy as np

from hilo_mpc import NMPC, Model


# Define the model
model = Model(plot_backend='bokeh')

# Constants
M = 5.

# States and algebraic variables
xx = model.set_dynamical_states('x', 'vx', 'y', 'vy')
model.set_measurements('y_x', 'y_vx', 'y_y', 'y_vy')
model.set_measurement_equations([xx[0], xx[1], xx[2], xx[3]])
x = xx[0]
vx = xx[1]
y = xx[2]
vy = xx[3]

# Inputs
F = model.set_inputs('Fx', 'Fy')
Fx = F[0]
Fy = F[1]

# ODEs
dd1 = vx
dd2 = Fx / M
dd3 = vy
dd4 = Fy / M

model.set_dynamical_equations([dd1, dd2, dd3, dd4])

# Initial conditions
x0 = [0, 0, 0, 0]
u0 = [0., 0.]

# time interval
dt = 0.1
model.setup(dt=dt)

# Define path following MPC
nmpc = NMPC(model)
theta = nmpc.create_path_variable(u_pf_lb=1e-6)
nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                   ref=ca.vertcat(ca.sin(theta), ca.sin(2 * theta)), path_following=True)
nmpc.horizon = 10
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
nmpc.setup(options={'print_level': 1})

# Prepare and run closed loop
n_steps = 200
model.set_initial_conditions(x0=x0)
sol = model.solution
xt = x0.copy()
for step in range(n_steps):
    u = nmpc.optimize(xt)
    model.simulate(u=u)
    xt = sol['x:f']

# define path function for plotting
def path(theta):
    return np.sin(theta), np.sin(2 * theta)
x_path = []
y_path = []
for t in range(1000):
    x_p, y_p = path(t / 100)
    x_path.append(x_p)
    y_path.append(y_p)

# Plot using bokeh
p_pf = figure(title='Path following problem')
p_pf.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
p_pf.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
# p = format_figure(p)
p_pf.yaxis.axis_label = "y [m]"
p_pf.xaxis.axis_label = "x [m]"
show(p_pf)

# Define trajectory tracking MPC
nmpc = NMPC(model)
time = nmpc.get_time_variable()
nmpc.quad_stage_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)), trajectory_tracking=True)
nmpc.quad_terminal_cost.add_states(names=['x', 'y'], weights=[10, 10],
                                   ref=ca.vertcat(ca.sin(time), ca.sin(2 * time)), trajectory_tracking=True)
nmpc.horizon = 10
nmpc.set_initial_guess(x_guess=x0, u_guess=u0)
nmpc.setup(options={'print_level': 1})

# Prepare and run closed loop
n_steps = 100
# the solution stored in the model must be reset (initial conditions are kept by default)
model.reset_solution()
sol = model.solution
xt = x0.copy()
for step in range(n_steps):
    u = nmpc.optimize(xt)
    model.simulate(u=u)
    xt = sol['x:f']

# plot using bokeh
p_tt = figure(title='Trajectory tracking problem')
p_tt.line(x=np.array(model.solution['x']).squeeze(), y=np.array(model.solution['y']).squeeze())
p_tt.line(x=x_path, y=y_path, line_color='red', line_dash='dashed')
# p = format_figure(p)
p_tt.yaxis.axis_label = "y [m]"
p_tt.xaxis.axis_label = "x [m]"
show(p_tt)
