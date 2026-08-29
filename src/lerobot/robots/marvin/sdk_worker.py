"""
Marvin SDK worker process.

Runs in a separate subprocess WITHOUT torch/numpy loaded, so libMarvinSDK.so
does not conflict with PyTorch's bundled C++ libraries.

Communication: multiprocessing.Pipe (duplex).
Protocol: send a (command, kwargs) tuple, receive result or raise exception.
"""

import sys
import time


def run_worker(conn):
    """Main loop of the SDK worker process."""
    # Minimal imports — no torch, no numpy
    sys.path.insert(0, sys.argv[1])  # sdk_dir passed as argv[1]
    from fx_robot import Marvin_Robot, DCSS

    sdk = None
    dcss = None

    while True:
        try:
            cmd, kwargs = conn.recv()
        except EOFError:
            break

        try:
            if cmd == "connect":
                sdk = Marvin_Robot()
                dcss = DCSS()
                result = sdk.connect(kwargs["ip"])
                conn.send(("ok", result))

            elif cmd == "clear_error":
                result = sdk.clear_error(kwargs["arm"])
                conn.send(("ok", result))

            elif cmd == "subscribe":
                data = sdk.subscribe(dcss)
                conn.send(("ok", data))

            elif cmd == "clear_set":
                sdk.clear_set()
                conn.send(("ok", None))

            elif cmd == "send_cmd":
                sdk.send_cmd()
                conn.send(("ok", None))

            elif cmd == "set_state":
                sdk.set_state(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_vel_acc":
                sdk.set_vel_acc(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_impedance_type":
                sdk.set_impedance_type(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_joint_kd_params":
                sdk.set_joint_kd_params(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_drag_space":
                sdk.set_drag_space(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_joint_cmd_pose":
                sdk.set_joint_cmd_pose(**kwargs)
                conn.send(("ok", None))

            elif cmd == "set_cyr_kd_params":
                result = sdk.set_cyr_kd_params(**kwargs)
                conn.send(("ok", result))

            elif cmd == "set_cyr_state_on":
                result = sdk.set_cyr_state_on(**kwargs)
                conn.send(("ok", result))

            elif cmd == "set_cyr_state_off":
                result = sdk.set_cyr_state_off(**kwargs)
                conn.send(("ok", result))

            elif cmd == "set_485_data":
                result = sdk.set_485_data(arm=kwargs["arm"], data=kwargs["data"], size_int=kwargs["size_int"], com=kwargs["com"])
                conn.send(("ok", result))

            elif cmd == "get_485_data":
                result = sdk.get_485_data(arm=kwargs["arm"], com=kwargs["com"])
                conn.send(("ok", result))

            elif cmd == "release_robot":
                if sdk is not None:
                    sdk.release_robot()
                conn.send(("ok", None))
                break

            else:
                conn.send(("error", f"Unknown command: {cmd}"))

        except Exception as e:
            conn.send(("error", str(e)))


if __name__ == "__main__":
    import multiprocessing
    # argv: worker.py <sdk_dir> <fd_in> <fd_out>
    # Use fd-based pipe handles passed from parent
    fd_in = int(sys.argv[2])
    fd_out = int(sys.argv[3])
    conn = multiprocessing.connection.Connection(fd_in, readable=True, writable=False)
    conn_out = multiprocessing.connection.Connection(fd_out, readable=False, writable=True)

    # Use a combined duplex connection if same fd (shouldn't happen), else wrap both
    class DuplexConn:
        def recv(self): return conn.recv()
        def send(self, v): conn_out.send(v)

    run_worker(DuplexConn())
