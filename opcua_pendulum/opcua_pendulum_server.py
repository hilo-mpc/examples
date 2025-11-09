"""
Stage 2 demo (part 1): Standalone OPC UA server + pendulum plant.

- Exposes two nodes with explicit NodeIds in a custom namespace:
  - ns=<idx>;s=Pendulum/Angle_rad
  - ns=<idx>;s=Pendulum/Torque_Nm
- Integrates a simple pendulum model and publishes the angle at Ts_plant.
- The torque command is read from the Torque_Nm node (writable by a client).

Run this server in one terminal, and the controller in another terminal.
You can limit the runtime with --run-seconds for testing. Press Ctrl+C to stop.
"""

import argparse
import asyncio
import math
import signal
from typing import Any, Tuple

import casadi as ca
from asyncua import Server, ua

# Physical parameters (SI)
g = 9.81
m = 0.5
l = 0.5
c = 0.05
I = m * l * l


async def start_pendulum_server(endpoint: str) -> Tuple[Server, Any, Any, int]:
    server = Server()
    await server.init()
    server.set_endpoint(endpoint)

    # Use a stable namespace URI; asyncua will assign a numeric index at runtime.
    uri = "http://examples.freeopcua.github.io"
    idx = await server.register_namespace(uri)

    objects = await server.nodes.objects.add_object(idx, "Pendulum")
    angle_node = await objects.add_variable(ua.NodeId("Pendulum/Angle_rad", idx), "Angle_rad", 0.0)
    torque_node = await objects.add_variable(ua.NodeId("Pendulum/Torque_Nm", idx), "Torque_Nm", 0.0)
    await torque_node.set_writable()

    await server.start()
    return server, angle_node, torque_node, idx


async def plant_loop(angle_node: Any, torque_node: Any, Ts_plant: float, run_seconds: float | None) -> None:
    """Simulate the pendulum using a continuous-time integrator (cvodes) via CasADi.

    State x = [theta, omega], input u = torque. We integrate over fixed intervals Ts_plant.
    """
    # Build integrator once (continuous, higher fidelity than explicit Euler)
    x = ca.SX.sym("x", 2)
    u_sym = ca.SX.sym("u", 1)
    theta = x[0]
    omega = x[1]
    ode = ca.vertcat(
        omega,
        -(g / l) * ca.sin(theta) - (c / I) * omega + (1.0 / I) * u_sym,
    )
    f = ca.Function("f", [x, u_sym], [ode])
    F = ca.integrator(
        "F",
        "cvodes",
        {"x": x, "p": u_sym, "ode": f(x, u_sym)},
        {"tf": Ts_plant},
    )

    xk = ca.DM([0.25, 0.0])
    t = 0.0
    await angle_node.write_value(float(xk[0]))

    try:
        while run_seconds is None or t < run_seconds:
            try:
                u = float(await torque_node.read_value())
            except Exception:
                u = 0.0

            res = F(x0=xk, p=u)
            xk = res["xf"]

            # Keep theta wrapped to [-pi, pi] for neatness (optional)
            th = float(xk[0])
            if th > math.pi:
                th -= 2 * math.pi
                xk[0] = th
            elif th < -math.pi:
                th += 2 * math.pi
                xk[0] = th

            await angle_node.write_value(float(xk[0]))
            await asyncio.sleep(Ts_plant)
            t += Ts_plant
    except asyncio.CancelledError:
        # Task was cancelled during shutdown; no special cleanup needed.
        pass


async def amain(endpoint: str, Ts_plant: float, run_seconds: float | None) -> None:
    server, angle_node, torque_node, ns_idx = await start_pendulum_server(endpoint)
    print(f"OPC UA server started at {endpoint} | namespace index = {ns_idx}")
    print("Nodes:")
    print(f"  ns={ns_idx};s=Pendulum/Angle_rad (readable)")
    print(f"  ns={ns_idx};s=Pendulum/Torque_Nm (writable)")

    task = asyncio.create_task(plant_loop(angle_node, torque_node, Ts_plant, run_seconds))

    # Handle Ctrl+C gracefully
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _on_sigint():
        stop_event.set()
    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
    except NotImplementedError:
        # Windows event loop may not support add_signal_handler for SIGINT
        pass

    try:
        if run_seconds is None:
            await stop_event.wait()
        else:
            await asyncio.wait_for(stop_event.wait(), timeout=run_seconds + 1.0)
    except asyncio.TimeoutError:
        pass
    finally:
        task.cancel()
        try:
            await task
        except Exception:
            pass
        await server.stop()
        print("Server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pendulum OPC UA server + plant")
    parser.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840/freeopcua/pendulum/", help="OPC UA endpoint URL")
    parser.add_argument("--Ts-plant", type=float, default=0.02, dest="Ts_plant", help="Plant integration step [s]")
    parser.add_argument("--run-seconds", type=float, default=None, help="Optional duration to run [s]")
    args = parser.parse_args()

    asyncio.run(amain(endpoint=args.endpoint, Ts_plant=args.Ts_plant, run_seconds=args.run_seconds))


if __name__ == "__main__":
    main()
