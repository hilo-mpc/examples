"""
HILO-MPC is developed by Johannes Pohlodek and Bruno Morabito under the supervision of Prof. Rolf Findeisen
at the  Control and cyber-physical systems laboratory, TU Darmstadt (https://www.ccps.tu-darmstadt.de/ccp) and at the
Laboratory for Systems Theory and Control, Otto von Guericke University (http://ifatwww.et.uni-magdeburg.de/syst/).
"""
import numpy as np

from hilo_mpc import Model, NMPC, SimpleControlLoop, ANN, Layer


# Initialize empty model
system = Model(plot_backend='bokeh', name='Linear SMD')

# Set states and inputs
system.set_dynamical_states('x', 2)
system.set_inputs('u')

# Add dynamics equations to model
system.set_dynamical_equations(['x_1', 'u - 2 * x_0 - 0.8 * x_1'])

# Sampling time
Ts = 0.015  # Ts = 15 ms

# Set up model
system.setup(dt=Ts)

# Initialize system
x_0 = [12.5, 0]
system.set_initial_conditions(x_0)

# Make controller
nmpc = NMPC(system)

# Set horizon
nmpc.horizon = 15

# Set cost function
nmpc.quad_stage_cost.add_states(names=['x_0', 'x_1'], weights=[100, 100], ref=[1, 0])
nmpc.quad_stage_cost.add_inputs(names=['u'], weights=[10], ref=[2])
nmpc.quad_terminal_cost.add_states(names=['x_0', 'x_1'], weights=np.array([[8358.1, 1161.7], [1161.7, 2022.9]]),
                                   ref=[1, 0])
nmpc.set_box_constraints(u_ub=[15], u_lb=[-20])

# Set up controller
nmpc.setup(options={'print_level': 1})

# Generate data set
Tf = 10  # Final time
n_steps = int(Tf / Ts)
data_set = system.generate_data('closed_loop', nmpc, steps=n_steps, use_input_as_label=True)

# Initialize ANN and create ANN structure
ann = ANN(data_set.features, data_set.labels, learning_rate=1e-3)
ann.add_layers(Layer.dense(10, activation='ReLU'))
ann.add_layers(Layer.dense(10, activation='ReLU'))
ann.add_layers(Layer.dense(10, activation='ReLU'))

# Add dataset
ann.add_data_set(data_set)

# Set up ANN
ann.setup(device='cpu')

# Train ANN
ann.train(1, 2000, validation_split=.2, patience=100)

# Try out the learned controller
system.reset_solution(keep_initial_conditions=False)
system.set_initial_conditions(x0=[10, 0])
scl = SimpleControlLoop(system, ann)
scl.run(n_steps)

# Prepare data for comparison
y_data = []
features, labels = data_set.raw_data
ct = 0
for k, feature in enumerate(data_set.features):
    y_data.append({
        'data': np.append(features[k, :], features[k, -1]),
        'kind': 'line',
        'subplot': ct,
        'label': feature + '_mpc'
    })
    ct += 1
for k, label in enumerate(data_set.labels):
    y_data.append({
        'data': np.append(labels[k, :], labels[k, -1]),
        'kind': 'step',
        'subplot': ct,
        'label': label + '_mpc'
    })
    ct += 1

# Plots
scl.plot(y_data=y_data)
