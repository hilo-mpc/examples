"""
HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
"""
import numpy as np

from hilo_mpc import Model, LQR, SimpleControlLoop


# Initialize empty model
model = Model(discrete=True, time_unit='', plot_backend='bokeh')

# Set model matrices
model.A = np.array([[1., 1.], [0., 1.]])
model.B = np.array([[0.], [1.]])

# Set up model
model.setup()

# Initialize LQR
lqr = LQR(model, plot_backend='bokeh')

# Set LQR horizon for finite horizon formulation
lqr.horizon = 5

# Set up LQR
lqr.setup()

# Initial conditions of the model
model.set_initial_conditions([2, 1])

# Set LQR matrices
lqr.Q = 2 * np.ones((2, 2))
lqr.R = 2

# Run simulation
scl = SimpleControlLoop(model, lqr)
scl.run(20)
scl.plot()
