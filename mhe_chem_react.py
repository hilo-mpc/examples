#
# HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
# at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
# Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
#
"""
Example using the Moving Horizon Estimator in a chemical reaction. The example is taken from the book
Model Predictive Control - Theory and Design. J. Rawlings and D. Mayne, doi: 10.1002/9781119941446.ch3
"""

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np

from hilo_mpc import Model, MHE


# Create model
model = Model(plot_backend='bokeh')
x = model.set_dynamical_states(['Ca', 'Cb', 'Cc'], units=['mol/l', 'mol/l', 'mol/l'],
                               short_description=['Ca', 'Cb', 'Cc'])
model.set_measurements(['P'], units=['atm'], short_description=['Pressure'])

# Unwrap states
Ca = x[0]
Cb = x[1]
Cc = x[2]

# Known Parameters
k1 = 0.5
k_1 = 0.05
k2 = 0.2
k_2 = 0.01
dt = 0.25
RT = 32.84  # L atm/ (mol)

dCa = - (k1 * Ca - k_1 * Cb * Cc)
dCb = k1 * Ca - k_1 * Cb * Cc - 2 * (k2 * Cb ** 2 - k_2 * Cc)
dCc = k1 * Ca - k_1 * Cb * Cc + 1 * (k2 * Cb ** 2 - k_2 * Cc)

model.set_measurement_equations(RT * (Ca + Cb + Cc))
model.set_dynamical_equations([dCa, dCb, dCc])

model.setup(dt=dt)

# Initial conditions
x0_real = [0.5, 0.05, 0]
x0_est = [1, 0, 4]

model.set_initial_conditions(x0=x0_real)

n_steps = 120

# Setup the MHE
mhe = MHE(model)
mhe.quad_arrival_cost.add_states(weights=[1 / (0.5 ** 2), 1 / (0.5 ** 2), 1 / (0.5 ** 2)], guess=x0_est)
mhe.quad_stage_cost.add_measurements(weights=[1 / (0.25 ** 2)])
mhe.quad_stage_cost.add_state_noise(weights=[1 / (0.001 ** 2), 1 / (0.001 ** 2), 1 / (0.001 ** 2)])
mhe.set_box_constraints(x_lb=[0, 0, 0])
mhe.horizon = 20
mhe.setup(options={'print_level': 0})

# Run the simulation
for i in range(n_steps):
    model.simulate()
    mhe.add_measurements(y_meas=model.solution['y'][:, -2])
    x_est, _ = mhe.estimate()

# If you want you can save the data to mat file. For that decomment the following lines. Remember to pass a valid path.
# mhe.solution.to_mat('t', 'x', file_name='../Results/mhe_chem_reactor/mhe.mat')
# model.solution.to_mat('t', 'x', file_name='../Results/mhe_chem_reactor/model.mat')

p_tot = []
for name in model.dynamical_state_names:
    p = figure(background_fill_color="#fafafa")
    p.scatter(x=np.array(mhe.solution['t']).squeeze(), y=np.array(mhe.solution[name]).squeeze(),
              legend_label='Estimated')
    p.line(x=np.array(model.solution['t']).squeeze(), y=np.array(model.solution[name]).squeeze(), legend_label='Real')
    p.yaxis.axis_label = name + ' mol/L'
    p.xaxis.axis_label = 'Time [s]'
    p.yaxis.axis_label_text_font_size = "18pt"
    p.yaxis.major_label_text_font_size = "18pt"
    p.xaxis.major_label_text_font_size = "18pt"
    p.xaxis.axis_label_text_font_size = "18pt"

    p_tot.append(p)

show(gridplot(p_tot, ncols=3))
