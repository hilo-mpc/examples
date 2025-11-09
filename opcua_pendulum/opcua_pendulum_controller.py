"""
Stage 2 demo (part 2): MPC controller process connecting to the pendulum OPC UA server.

- Connects to the endpoint and reads angle; writes torque.
- Uses the AsyncOPCUAClient with simple auto-reconnect and backoff.
- Prints timing metrics: read, optimize, write, cycle time, and jitter.

Run this after starting `opcua_pendulum_server.py` in a separate terminal.
"""

import argparse
import asyncio
import math


import casadi as ca


from hilo_mpc import Model, NMPC
from hilo_mpc.plugins.io import OPCUAConnector, IOMapping


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

    # phi_dot = omega
    # omega_dot = + (g/l) sin(phi) - (c/I) omega + (1/I) u
    ode = [
        omega,
        (g / l) * ca.sin(phi) - (c / I) * omega + (1.0 / I) * torque,
    ]
    model.set_equations(ode=ode)
    model.discretize("erk", order=4, inplace=True)  # higher accuracy at same Ts
    model.setup(dt=Ts_ctrl)

    mpc = NMPC(model)
    mpc.horizon = 20
    mpc.quad_stage_cost.add_states(names=["phi", "omega"], ref=[0.0, 0.0], weights=[25.0, 2.0])
    mpc.quad_terminal_cost.add_states(names=["phi", "omega"], ref=[0.0, 0.0], weights=[25.0, 10.0])
    # mpc.quad_stage_cost.add_inputs(names=["u"], weights=[0.05])
    u_max = 5.0
    mpc.set_box_constraints(u_lb=[-u_max], u_ub=[u_max])
    mpc.set_initial_guess(x_guess=[0.0, 0.0], u_guess=[0.0])
    mpc.setup(options={"objective_function": "discrete"})
    return mpc


async def controller_loop(
    endpoint: str,
    ns_idx: int,
    Ts_ctrl: float,
    iters: int|None,
    reconnect: tuple[float, float],
) -> None:
    """Run the MPC controller against the OPC UA server.

    Parameters
    ----------
    endpoint: str
        OPC UA endpoint, e.g. opc.tcp://127.0.0.1:4840/freeopcua/pendulum/
    ns_idx: int
        Namespace index used by the server for the Pendulum nodes.
    Ts_ctrl: float
        Controller period in seconds.
    iters: int | None
        If provided, run for a finite number of iterations; otherwise run indefinitely.
    reconnect: (float, float)
        Reconnect backoff bounds (low, high) in seconds.
    """
    # Build controller
    mpc = build_pendulum_nmpc(Ts_ctrl)

    # Use OPCUAConnector.from_model to auto-build mapping from reparameterized model, then override to match server
    # - Read 'phi' alias from server node Pendulum/Angle_rad (raw theta value; we'll convert to phi in compute)
    # - Write only 'u' to server node Pendulum/Torque_Nm
    node_overrides = {
        "phi": f"ns={ns_idx};s=Pendulum/Angle_rad",
        "u": f"ns={ns_idx};s=Pendulum/Torque_Nm",
    }

    # Build explicit IOMapping and plain OPCUAConnector (no from_model)
    mapping = IOMapping(
        reads={
            "phi": {"node": node_overrides["phi"]},
        },
        writes={
            "u": {"node": node_overrides["u"]},
        },
    )

    loop = OPCUAConnector(
        endpoint=endpoint,
        mapping=mapping,
        period=Ts_ctrl,
        reconnect_backoff=reconnect,
        safe_shutdown={"u": 0.0},
    )

    # Finite-difference state augmentation for omega based on wrapped phi
    state = {"phi_prev": None}

    def _wrap_to_pi(a: float) -> float:
        while a <= -math.pi:
            a += 2 * math.pi
        while a > math.pi:
            a -= 2 * math.pi
        return a

    def control_loop(reads: dict[str, float]) -> dict[str, float]:
        theta_raw = float(reads.get("phi", 0.0))  # actually theta from server
        phi = _wrap_to_pi(theta_raw - math.pi)

        phi_prev = state["phi_prev"]
        if phi_prev is None:
            omega_est = 0.0
        else:
            dphi = _wrap_to_pi(phi - phi_prev)
            omega_est = dphi / Ts_ctrl
        state["phi_prev"] = phi

        u_opt = mpc.optimize([phi, omega_est])
        u_scalar = float(u_opt[0])
        return {"u": u_scalar}

    print("Controller started. Press Ctrl+C to stop.")
    await loop.run(control_loop, max_iters=iters)


def main() -> None:
    """CLI entry-point for the controller example."""
    parser = argparse.ArgumentParser(description="Pendulum MPC controller over OPC UA")
    parser.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840/freeopcua/pendulum/", help="OPC UA endpoint URL")
    parser.add_argument("--ns-idx", type=int, default=2, help="Namespace index used by server for Pendulum nodes")
    parser.add_argument("--Ts-ctrl", type=float, default=0.02, dest="Ts_ctrl", help="Controller sample period [s]")
    parser.add_argument("--iters", type=int, default=None, help="Number of iterations before exit")
    parser.add_argument("--reconnect", type=float, nargs=2, default=(0.5, 5.0), metavar=("LO","HI"), help="Reconnect backoff range [s]")
    args = parser.parse_args()

    asyncio.run(
        controller_loop(
            endpoint=args.endpoint,
            ns_idx=args.ns_idx,
            Ts_ctrl=args.Ts_ctrl,
            iters=args.iters,
            reconnect=tuple(args.reconnect),
        )
    )


if __name__ == "__main__":
    main()
