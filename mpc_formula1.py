"""
This examples uses a car model
The complex car model was taken from the paper "Optimization-Based Autonomous Racing of 1:43 Scale RC Cars".
Thanks to Alex Liniger to share the parameters of the model
"""

from hilo_mpc import Model, NMPC, UKF
import casadi as ca
import numpy as np

# Necessary for plots
from bokeh.plotting import figure
from bokeh.io import show

""" Initialize simple car model """
model = Model(plot_backend='bokeh')

states = model.set_dynamical_states(['px', 'py', 'v', 'phi'])
inputs = model.set_inputs(['a', 'delta'])
model.set_measurements(['y_px', 'y_py', 'y_v', 'y_phi'])
model.set_measurement_equations(states)

# Unwrap states
px = states[0]
py = states[1]
v = states[2]
phi = states[3]

# Unwrap states
a = inputs[0]
delta = inputs[1]

# Parameters
lr = 1.4  # [m]
lf = 1.8  # [m]
beta = ca.arctan(lr / (lr + lf) * ca.tan(delta))

# ODE
dpx = v * ca.cos(phi + beta)
dpy = v * ca.sin(phi + beta)
dv = a
dphi = v / lr * ca.sin(beta)

model.set_dynamical_equations([dpx, dpy, dv, dphi])

# Initial conditions
x0 = [15, 30, 0, 0]
u0 = [0., 0.]

# Create model and run simulation
dt = 0.1
model.setup(dt=dt)
model.set_initial_conditions(x0=x0)

""" Setup Observer """
observer = UKF(model, plot_backend='bokeh', alpha=1, beta=0, kappa=1)
# Set up Kalman filter
observer.setup()
observer.R = [1e-4, 1e-4, 1e-4, 1e-4]
observer.set_initial_guess(x0, P0=[.01, .01, 0.02, 0.01])

""" Setup NMPC """
nmpc = NMPC(model)

# Obstacle position and geometry
obs_x = 30
obs_y = 15
obs_rad = 2  # [m]

theta = nmpc.create_path_variable(u_pf_lb=0.2, u_pf_ub=1)
path_x = 30 - 14 * ca.cos(theta)
path_y = 30 - 16 * ca.sin(theta)

nmpc.horizon = 20
nmpc.quad_stage_cost.add_states(names=['px', 'py'], ref=ca.vertcat(path_x, path_y), weights=[1, 1], path_following=True)
nmpc.quad_stage_cost.add_inputs(names=['a', 'delta'], weights=[1, 1])
nmpc.quad_terminal_cost.add_states(names=['px', 'py'], ref=ca.vertcat(path_x, path_y), weights=[1, 1],
                                   path_following=True)
nmpc.set_box_constraints(x_lb=[-100, -100, -10, -100], x_ub=[100, 100, 10, 100], u_lb=[-1, -1], u_ub=[1, 1])

nmpc.stage_constraint.constraint = (px - obs_x) ** 2 + (py - obs_y) ** 2
nmpc.stage_constraint.ub = ca.inf
nmpc.stage_constraint.lb = obs_rad ** 2
# nmpc.stage_constraint.max_violation = 0.5
# nmpc.stage_constraint.weight = 10

nmpc.terminal_constraint.constraint = (px - obs_x) ** 2 + (py - obs_y) ** 2
nmpc.terminal_constraint.ub = ca.inf
nmpc.terminal_constraint.lb = obs_rad ** 2
# nmpc.terminal_constraint.max_violation = 0.5
# nmpc.terminal_constraint.weight = 10

nmpc.set_initial_guess(x_guess=x0, u_guess=[0, 0])
nmpc.setup(options={'print_level': 1})

""" Run simulation """
n_steps = 300
solution = model.solution

for _ in range(n_steps):
    u = nmpc.optimize(x0)
    model.simulate(u=u)
    yf = solution.make_some_noise('y:f', var={'y': [1e-1, 1e-1, 1e-3, 1e-3]})
    observer.estimate(y=yf, u=u)
    x0 = observer.solution['x:f']

""" Plots """

# Create function to plot the path
pp = ca.SX.sym('pp')
path_x = 30 - (14 * pp) * ca.cos(theta)
path_y = 30 - (16 * pp) * ca.sin(theta)

path_fun = ca.Function('path', [theta, pp], [path_x, path_y])

x_path = []
y_path = []
for t in range(800):
    x_p, y_p = path_fun(t / 100, 1)
    x_path.append(float(x_p))
    y_path.append(float(y_p))

plot = figure(y_range=(10, 50), x_range=(10, 50), y_axis_label='y', x_axis_label='x', title="F1 racing with obstacle.")

plot.line(x=x_path, y=y_path, alpha=0.8, color='grey', line_width=30)
plot.line(x=x_path, y=y_path, line_dash='dashed', color='white')

plot.line(x=np.array(model.solution['px']).squeeze(), y=np.array(model.solution['py']).squeeze(), color='red',
          line_width=3, legend_label='position')
plot.line(x=np.array(observer.solution['px']).squeeze(), y=np.array(observer.solution['py']).squeeze(), color='blue',
          line_dash='dashed',
          line_width=3, legend_label='obs. pos.')
r = plot.circle(x=obs_x, y=obs_y)
glyph = r.glyph
glyph.radius = obs_rad / 2
glyph.fill_alpha = 0.9
glyph.fill_color = 'black'
glyph.line_color = "firebrick"
glyph.line_width = 2

show(plot)
