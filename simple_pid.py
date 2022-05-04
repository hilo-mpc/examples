"""
HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
"""
from hilo_mpc import Model, PID, SimpleControlLoop


# Initialize empty model
model = Model(plot_backend='bokeh')

# Set model states
model.set_dynamical_states('x')

# Set model inputs
model.set_inputs('u')

# Set model ODE's
model.set_dynamical_equations('2*x+u')

# Set up model
model.setup(dt=.01)

# Initialize model states
x0 = 0.
model.set_initial_conditions(x0)

# Initialize PID controller
pid = PID(plot_backend='bokeh')

# Set up PID controller
pid.setup(dt=.01)

# Set PID tuning parameters
pid.k_p = 8.
pid.t_i = 1.

# Unit step
pid.set_point = 1.

# Closed loop
scl = SimpleControlLoop(model, pid)
scl.run(1000)
scl.plot()
