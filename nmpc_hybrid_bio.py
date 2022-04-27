#
#   This file is part of HILO-MPC
#
#   HILO-MPC is toolbox for easy, flexible and fast development of machine-learning supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with HILO-MPC.  If not, see <http://www.gnu.org/licenses/>.

"""
Example of learning supported NMPC applied on a bioreactor. A artificial neural network is used to learn the reaction
rates. For a more detailed description of the example, please refer to the documentation.
"""

import pandas as pd
from hilo_mpc import ANN, Layer, NMPC, SimpleControlLoop
from hilo_mpc.library.models import ecoli_D1210_conti
from hilo_mpc.util.plotting import set_plot_backend
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np

set_plot_backend('bokeh')
# Load known model
model = ecoli_D1210_conti(model='simple')

# Load model 'real' plant
plant = ecoli_D1210_conti(model='complex')
plant.setup(dt=1)
x0_plant = [0.1, 40, 0, 0, 1, 0]
plant.set_initial_conditions(x0_plant)

# Load dataset
df = pd.read_csv('data/learning_ecoli/complete_dataset_5_batches.csv', index_col=0).dropna()

features = ['S', 'I']
labels = ['mu', 'Rs', 'Rfp']

# Initialize NN
ann = ANN(features, labels, seed=2)
ann.add_layers(Layer.dense(10, activation='sigmoid'))
ann.setup(show_tensorboard=True, tensorboard_log_dir='./runs/ecoli/conti')

# Add dataset
ann.add_data_set(df)
# TODO: I think we want to use validate_split here
ann.train(1, 2000, validation_split=.2, patience=100)

# Create the gray-box model
model.substitute_from(ann)

# Initialize MPC
model.setup(dt=1)
nmpc_hybrid = NMPC(model)

nmpc_hybrid.quad_stage_cost.add_states(names='P', ref=2., weights=10.)
nmpc_hybrid.quad_terminal_cost.add_states(names='P', ref=2., weights=10.)

nmpc_hybrid.horizon = 10
nmpc_hybrid.set_box_constraints(x_lb=[0., 0., 0., 0.], u_lb=[0, 0])
nmpc_hybrid.set_initial_guess(x_guess=[.1, 40., 0., 0.], u_guess=[0, 0])
nmpc_hybrid.setup()

n_steps = 100
p0 = [100, 4]
solution_hybrid = plant.solution

scl = SimpleControlLoop(plant, nmpc_hybrid)
scl.run(steps=n_steps, p=p0)
scl.plot()

# If you want to save the solutions to mat files, uncomment the following. Remember to use an existing path in the
# 'file_name' argument
# nmpc_hybrid.solution.to_mat('t', 'x', 'u', file_name='../Results/nmpc_hybrid_bio/nmpc_hybrid.mat')
# plant.solution.to_mat('t', 'x', 'u', file_name='../Results/nmpc_hybrid_bio/plant_hybrid.mat')

plant = ecoli_D1210_conti(model='complex')
plant.setup(dt=1)
x0_plant = [0.1, 40, 0, 0, 1, 0]
plant.set_initial_conditions(x0_plant)

# Initialize MPC
nmpc_real = NMPC(plant)
nmpc_real.quad_stage_cost.add_states(names='P', ref=2., weights=10.)
nmpc_real.quad_terminal_cost.add_states(names='P', ref=2., weights=10.)
nmpc_real.horizon = 10
nmpc_real.set_box_constraints(x_lb=[0., 0., 0., 0., 0, 0], u_lb=[0, 0])
nmpc_real.set_initial_guess(x_guess=[.1, 40., 0., 0., 1, 0], u_guess=[0, 0])
nmpc_real.setup()

solution_real = plant.solution
scl = SimpleControlLoop(plant, nmpc_real)
scl.run(steps=n_steps, p=p0)
scl.plot()



p_tot = []
for state in plant.dynamical_state_names:
    p = figure(background_fill_color="#fafafa", width=300, height=300)
    p.line(x=np.array(solution_hybrid['t']).squeeze(), y=np.array(solution_hybrid[state]).squeeze(),
           legend_label=state + '_real', line_width=2)
    p.line(x=np.array(solution_real['t']).squeeze(), y=np.array(solution_real[state]).squeeze(),
           legend_label=state, line_width=2, color='green')
    for i in range(len(nmpc_hybrid.quad_stage_cost._references_list)):
        if state in nmpc_hybrid.quad_stage_cost._references_list[i]['names']:
            position = nmpc_hybrid.quad_stage_cost._references_list[i]['names'].index(state)
            value = nmpc_hybrid.quad_stage_cost._references_list[i]['ref'][position]
            p.line([np.array(solution_hybrid['t'][1]).squeeze(), np.array(solution_hybrid['t'][-1]).squeeze()],
                   [value, value], legend_label=state + '_ref',
                   line_dash='dashed', line_color="red", line_width=2)

    p.yaxis.axis_label = state
    p.xaxis.axis_label = 'time'
    p.yaxis.axis_label_text_font_size = "18pt"
    p.yaxis.major_label_text_font_size = "18pt"
    p.xaxis.major_label_text_font_size = "18pt"
    p.xaxis.axis_label_text_font_size = "18pt"

    p_tot.append(p)

show(gridplot(p_tot, ncols=2))
