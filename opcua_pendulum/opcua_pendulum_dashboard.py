"""
Simple OPC UA dashboard: shows a live pendulum sketch with angle and torque.

- Connects to the same OPC UA endpoint as the server (read-only).
- Polls Angle_rad and Torque_Nm at a fixed interval using the AsyncOPCUAClient.
- Serves a tiny HTTP dashboard with a canvas rendering and live numeric values.

No extra dependencies required (uses Python stdlib http.server + threading + our async OPC UA client).

Usage (in a third terminal):
    python examples/opcua_pendulum_dashboard.py --endpoint opc.tcp://127.0.0.1:4840/freeopcua/pendulum/ --ns-idx 2
Then open your browser at http://127.0.0.1:8000/
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any

from hilo_mpc.plugins.io import AsyncOPCUAClient, IOMapping


DASHBOARD_HTML = b"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Pendulum Dashboard</title>
<style>
  body { font-family: system-ui, Arial, sans-serif; margin: 16px; color: #222; }
  .row { display: flex; gap: 24px; align-items: flex-start; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  #canvas { background: #fafafa; border: 1px solid #eee; border-radius: 8px; }
  .kv { margin: 4px 0; }
  .kv label { display: inline-block; width: 120px; color: #555; }
  .header { margin-bottom: 12px; }
</style>
</head>
<body>
  <div class=\"header\">
    <h2>Pendulum Dashboard</h2>
    <div>Reading from OPC UA: <code id=\"endpoint\"></code></div>
  </div>
  <div class=\"row\">
    <div class=\"card\">
  <canvas id=\"canvas\" width=520 height=420></canvas>
    </div>
    <div class=\"card\">
      <div class=\"kv\"><label>Angle (rad):</label><span id=\"theta\">--</span></div>
      <div class=\"kv\"><label>Angle (deg):</label><span id=\"theta_deg\">--</span></div>
      <div class=\"kv\"><label>Torque (Nm):</label><span id=\"torque\">--</span></div>
      <div class=\"kv\"><label>Updated:</label><span id=\"updated\">--</span></div>
    </div>
  </div>
<script>
  const endpointEl = document.getElementById('endpoint');
  const thetaEl = document.getElementById('theta');
  const thetaDegEl = document.getElementById('theta_deg');
  const torqueEl = document.getElementById('torque');
  const updatedEl = document.getElementById('updated');

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width; const H = canvas.height;
  // Move pivot lower to create more room above for inverted tip; increase rod length slightly
  const cx = W/2; const cy = 180; const L = 170; // pivot lower, rod length

  function draw(theta) {
    ctx.clearRect(0,0,W,H);

    // axes / ground
    ctx.strokeStyle = '#ddd'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();

    // pivot
    ctx.fillStyle = '#444';
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2*Math.PI); ctx.fill();

    // rod (theta=0 points straight down)
    const x2 = cx + L * Math.sin(theta);
    const y2 = cy + L * Math.cos(theta);
    ctx.strokeStyle = '#1f77b4'; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x2, y2); ctx.stroke();

    // bob
    ctx.fillStyle = '#ff7f0e';
    ctx.beginPath(); ctx.arc(x2, y2, 10, 0, 2*Math.PI); ctx.fill();
  }

  async function refresh() {
    try {
      const r = await fetch('/metrics', {cache: 'no-store'});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const j = await r.json();
      endpointEl.textContent = j.endpoint;
      thetaEl.textContent = j.theta.toFixed(4);
      thetaDegEl.textContent = (j.theta * 180/Math.PI).toFixed(2);
      torqueEl.textContent = j.torque.toFixed(3);
      updatedEl.textContent = new Date(j.ts*1000).toLocaleTimeString();
      draw(j.theta);
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(refresh, 200);
    }
  }
  draw(0);
  refresh();
</script>
</body>
</html>
"""


class State:
    def __init__(self, endpoint: str, ns_idx: int, interval: float) -> None:
        self.endpoint = endpoint
        self.ns_idx = ns_idx
        self.interval = interval
        self.theta = 0.0
        self.torque = 0.0
        self.ts = time.time()
        self._lock = threading.Lock()

    def set(self, theta: float, torque: float) -> None:
        with self._lock:
            self.theta = float(theta)
            self.torque = float(torque)
            self.ts = time.time()

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return {"theta": self.theta, "torque": self.torque, "ts": self.ts}


class DashboardHandler(BaseHTTPRequestHandler):
    state: State = None  # type: ignore

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML)
            return
        if self.path.startswith("/metrics"):
            snap = self.state.get()
            body = json.dumps({
                "endpoint": self.state.endpoint,
                "ns_idx": self.state.ns_idx,
                **snap,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - suppress default noisy logging
        return


def opcua_reader_loop(state: State, stop_evt: threading.Event) -> None:
    """Run an asyncio client loop in this background thread, polling OPC UA and updating state."""
    import asyncio

    async def runner():
        mapping = IOMapping(
            reads={
                "y_angle": {"node": f"ns={state.ns_idx};s=Pendulum/Angle_rad"},
                "y_torque": {"node": f"ns={state.ns_idx};s=Pendulum/Torque_Nm"},
            }
        )
        client = AsyncOPCUAClient(endpoint=state.endpoint, mapping=mapping)
        await client.connect()
        try:
            while not stop_evt.is_set():
                try:
                    data = await client.read(["y_angle", "y_torque"])
                    theta = float(data["y_angle"]["value"]) if "y_angle" in data else 0.0
                    torque = float(data["y_torque"]["value"]) if "y_torque" in data else 0.0
                    state.set(theta, torque)
                except Exception:
                    # Auto-reconnect handled by client methods; keep going
                    pass
                await asyncio.sleep(state.interval)
        finally:
            await client.disconnect()

    asyncio.run(runner())


def main() -> None:
    ap = argparse.ArgumentParser(description="OPC UA pendulum live dashboard")
    ap.add_argument("--endpoint", default="opc.tcp://127.0.0.1:4840/freeopcua/pendulum/", help="OPC UA endpoint")
    ap.add_argument("--ns-idx", type=int, default=2, help="Namespace index for Pendulum nodes")
    ap.add_argument("--interval", type=float, default=0.1, help="Polling interval [s]")
    ap.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    ap.add_argument("--port", type=int, default=8000, help="Dashboard port")
    args = ap.parse_args()

    state = State(endpoint=args.endpoint, ns_idx=args.ns_idx, interval=args.interval)

    # Start background reader
    stop_evt = threading.Event()
    t = threading.Thread(target=opcua_reader_loop, args=(state, stop_evt), daemon=True)
    t.start()

    # Start HTTP server
    DashboardHandler.state = state
    httpd = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard at http://{args.host}:{args.port}/ (endpoint {args.endpoint}, ns-idx {args.ns_idx})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        httpd.server_close()
        print("Dashboard stopped.")


if __name__ == "__main__":
    main()
