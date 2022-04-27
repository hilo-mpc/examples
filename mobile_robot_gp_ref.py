from hilo_mpc import Model, NMPC, SimpleControlLoop, GP
import hilo_mpc.modules.machine_learning.gp.kernel as kernels
import numpy as np
import casadi as ca


model = Model(plot_backend='bokeh')

equations = """
dx1/dt = u1(k)*cos(x3(t))
dx2/dt = u1(k)*sin(x3(t))
dx3/dt = u2(k)
"""
model.set_equations(equations=equations)

dt = 0.1

model.setup(dt=dt)
x0 = [1, 1, 0]
Tf = 20
N = int(Tf / dt)
time = np.linspace(0, Tf, N)
model.set_initial_conditions(x0)

"""Generate data"""
u1 = 2 + 0.1 * np.sin(time)
u2 = 0.5 + np.sin(time)

for i in range(N):
    model.simulate(u=[u1[i], u2[i]])

model.solution.plot()
# If you want to save the results to a mat file decomment this. Remember to pass a valid path to file_name
# model.solution.to_mat('t', 'x', file_name='../Results/mobile_robots/data_generation.mat')
# trajectory = ca.vertcat(model.solution['x1'], model.solution['x2'], model.solution['x3'])

""" Learn data """
kernel = kernels.SquaredExponentialKernel( length_scales=0.1,
                                    bounds={'length_scales': (0.01, 1e2),
                                            'variance': (0.01, 1e2)})
gp_1 = GP(['t'], ['x1'], bounds=(.01, 100.), kernel=kernel, id='gp1')

kernel = kernels.SquaredExponentialKernel(length_scales=0.1,
                                    bounds={'length_scales': (0.01, 1e2),
                                            'variance': (0.01, 1e2)})
gp_2 = GP(['t'], ['x2'], bounds=(.01, 100.), kernel=kernel, id='gp2')

kernel = kernels.SquaredExponentialKernel(length_scales=0.1,
                                    bounds={'length_scales': (0.01, 1e2),
                                            'variance': (0.01, 1e2)})

gp_3 = GP(['t'], ['x3'], bounds=(.01, 100.), kernel=kernel, id='gp3')

gp_1.set_training_data(np.array(model.solution['t']), np.array(model.solution['x1']))
gp_2.set_training_data(np.array(model.solution['t']), np.array(model.solution['x2']))
gp_3.set_training_data(np.array(model.solution['t']), np.array(model.solution['x3']))

gp_1.setup()
gp_2.setup()
gp_3.setup()

gp_1.fit_model()
gp_2.fit_model()
gp_3.fit_model()

gp_1.plot()
gp_2.plot()
gp_3.plot()

""" Setup MPC """
model.reset_solution()

nmpc = NMPC(model)
t = nmpc.get_time_variable('t')

gp_1_mean, _ = gp_1.predict(t)
gp_2_mean, _ = gp_2.predict(t)
gp_3_mean, _ = gp_3.predict(t)

nmpc.horizon = 50
nmpc.quad_stage_cost.add_states(names=['x1', 'x2', 'x3'],
                                ref={'x1':gp_1_mean, 'x2':gp_2_mean, 'x3':gp_3_mean},
                                trajectory_tracking=True,
                                weights=[10, 10, 10])
nmpc.quad_terminal_cost.add_states(names=['x1', 'x2', 'x3'],
                                   ref={'x1':gp_1_mean, 'x2':gp_2_mean, 'x3':gp_3_mean},
                                   weights=[10, 10, 10],
                                   trajectory_tracking=True)
nmpc.setup(options={'objective_function': 'discrete'})

""" Run loop"""
sloop = SimpleControlLoop(model, nmpc)
sloop.run(N - nmpc.horizon - 1)
sloop.plot()

# If you want to save the results to a mat file decomment this. Remember to pass avalid path to file_name
# model.solution.to_mat('t', 'x', file_name='../Results/mobile_robots/mpc.mat')
