"""
OPC UA pendulum control using OPCUASimpleControlLoop (SimpleControlLoop-style interface).

- Reads angle from the server, computes control with an NMPC controller, and writes torque.
- Uses an adapter so the controller can work with phi=theta-pi and an omega estimate derived
  from successive reads (no plant simulation here).

Run after starting the server:
  python examples/opcua_pendulum_server.py

Then in another terminal run this controller:
  python examples/opcua_pendulum_simple_control_loop.py --ns-idx 2
"""

from __future__ import annotations

import argparse
import math
from typing import Optional, Sequence

import casadi as ca
import numpy as np

from hilo_mpc import Model, NMPC
from hilo_mpc.plugins.io import IOMapping, OPCUASimpleControlLoop


# Physical parameters (match server)
g = 9.81
m = 0.5
l = 0.5
c = 0.05
I = m * l * l


def build_pendulum_nmpc(Ts_ctrl: float) -> NMPC:
    """NMPC around upright using phi = theta - pi to avoid wrap at ±pi."""
    model = Model()
    x = model.set_dynamical_states(["phi", "omega"])  # reparameterized angle
    u = model.set_inputs(["u"])  # torque

    phi = x[0]
    omega = x[1]
    torque = u[0]

    ode = [
        omega,
        (g / l) * ca.sin(phi) - (c / I) * omega + (1.0 / I) * torque,
    ]
    model.set_equations(ode=ode)
    model.discretize("erk", order=4, inplace=True)
    model.setup(dt=Ts_ctrl)

    mpc = NMPC(model)
    mpc.horizon = 20
    mpc.quad_stage_cost.add_states(names=["phi", "omega"], ref=[0.0, 0.0], weights=[25.0, 2.0])
    mpc.quad_terminal_cost.add_states(names=["phi", "omega"], ref=[0.0, 0.0], weights=[25.0, 10.0])
    u_max = 5.0
    mpc.set_box_constraints(u_lb=[-u_max], u_ub=[u_max])
    mpc.set_initial_guess(x_guess=[0.0, 0.0], u_guess=[0.0])
    mpc.setup(options={"objective_function": "discrete"})
    return mpc


class PendulumNMPCAdapter:
    """Adapter presenting a SimpleControlLoop-like optimize(x) for OPCUASimpleControlLoop.

    The incoming x is a 1-element vector [theta]. We internally construct
    [phi, omega] and call the wrapped NMPC.optimize().
    """

    type = "NMPC"  # so OPCUASimpleControlLoop treats this as MPC for optimize()

    def __init__(self, nmpc: NMPC, Ts_ctrl: float) -> None:
        self.nmpc = nmpc
        self.Ts = float(Ts_ctrl)
        self._phi_prev: Optional[float] = None

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        while a <= -math.pi:
            a += 2 * math.pi
        while a > math.pi:
            a -= 2 * math.pi
        return a

    def is_setup(self) -> bool:
        return True  # underlying NMPC already set up

    def optimize(self, x: Sequence[float], cp=None):  # SimpleControlLoop-compatible
        theta = float(x[0]) if len(x) > 0 else 0.0
        phi = self._wrap_to_pi(theta - math.pi)
        if self._phi_prev is None:
            omega = 0.0
        else:
            dphi = self._wrap_to_pi(phi - self._phi_prev)
            omega = dphi / self.Ts
        self._phi_prev = phi
        return self.nmpc.optimize([phi, omega], cp=cp)


def main() -> None:
    ap = argparse.ArgumentParser(description="OPC UA pendulum via OPCUASimpleControlLoop")
    ap.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840/freeopcua/pendulum/", help="OPC UA endpoint")
    ap.add_argument("--ns-idx", type=int, default=2, help="Namespace index for Pendulum nodes")
    ap.add_argument("--Ts-ctrl", type=float, default=0.02, dest="Ts_ctrl", help="Controller period [s]")
    ap.add_argument("--iters", type=int, default=None, help="Run for N iterations then exit")
    ap.add_argument("--reconnect", type=float, nargs=2, default=(0.5, 5.0), metavar=("LO","HI"), help="Reconnect backoff range [s]")
    args = ap.parse_args()

    # Build controller (wrapped)
    nmpc = build_pendulum_nmpc(args.Ts_ctrl)
    ctrl = PendulumNMPCAdapter(nmpc, args.Ts_ctrl)

    # Explicit mapping and loop
    mapping = IOMapping(
        reads={
            "theta": {"node": f"ns={args.ns_idx};s=Pendulum/Angle_rad"},
        },
        writes={
            "u": {"node": f"ns={args.ns_idx};s=Pendulum/Torque_Nm"},
        },
    )

    loop = OPCUASimpleControlLoop(
        endpoint=args.endpoint,
        mapping=mapping,
        controller=ctrl,
        estimator=None,
        state_aliases=["theta"],
        control_aliases=["u"],
        parameter_aliases=[],
        period=args.Ts_ctrl,
        reconnect_backoff=tuple(args.reconnect),
        safe_shutdown={"u": 0.0},
    )

    print("OPC UA pendulum (OPCUASimpleControlLoop) started. Ctrl+C to stop.")
    loop.run_sync(max_iters=args.iters)


if __name__ == "__main__":
    main()
