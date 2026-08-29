"""
Marvin SDK proxy — runs libMarvinSDK.so in a subprocess to avoid
symbol conflicts with PyTorch's bundled C++ libraries.
"""

import os
import sys
import multiprocessing
import multiprocessing.connection
import threading
import time

# Use 'spawn' instead of the default 'fork' to avoid inheriting PyTorch's
# C++ libraries into the SDK subprocess.  libMarvinSDK.so can conflict with
# PyTorch's bundled libs (symbol collisions → segfault on OnGetBuf/subscribe).
_mp = multiprocessing.get_context("spawn")

_proxies: dict[str, "_SDKProxy"] = {}  # ip -> proxy
_refcount: dict[str, int] = {}
_latest_data: dict[str, dict] = {}
_locks: dict[str, threading.Lock] = {}

_SDK_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Marvin_sdk"))


class _SDKProxy:
    """Proxy that forwards SDK calls to a clean subprocess via a Pipe."""

    def __init__(self):
        parent_conn, child_conn = _mp.Pipe(duplex=True)
        self._conn = parent_conn
        self._proc = _mp.Process(
            target=_worker_main,
            args=(child_conn, _SDK_DIR),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    def _call(self, cmd, **kwargs):
        self._conn.send((cmd, kwargs))
        status, result = self._conn.recv()
        if status == "error":
            raise RuntimeError(f"SDK worker error: {result}")
        return result

    def connect(self, ip):              return self._call("connect", ip=ip)
    def clear_error(self, arm):         return self._call("clear_error", arm=arm)
    def subscribe(self):                return self._call("subscribe")
    def clear_set(self):                return self._call("clear_set")
    def send_cmd(self):                 return self._call("send_cmd")
    def release_robot(self):
        """
        Release the robot connection and terminate the SDK worker process.

        Note: The SDK's C++ library may crash during cleanup (double free error),
        so we terminate the process gracefully and suppress any errors.
        """
        try:
            # Try to send release command, but don't wait for response
            # The worker may crash during sdk.release_robot()
            self._conn.send(("release_robot", {}))
            # Give it a brief moment to process
            import time
            time.sleep(0.05)
        except Exception:
            pass

        # Close the pipe to prevent broken pipe errors
        try:
            self._conn.close()
        except Exception:
            pass

        # Terminate the process (it may have already crashed)
        try:
            self._proc.terminate()
            self._proc.join(timeout=1)
        except Exception:
            pass

        # Force kill if still alive
        if self._proc.is_alive():
            try:
                self._proc.kill()
                self._proc.join(timeout=0.5)
            except Exception:
                pass

    def set_state(self, arm, state):
        return self._call("set_state", arm=arm, state=state)

    def set_vel_acc(self, arm, velRatio, AccRatio):
        return self._call("set_vel_acc", arm=arm, velRatio=velRatio, AccRatio=AccRatio)

    def set_impedance_type(self, arm, type):
        return self._call("set_impedance_type", arm=arm, type=type)

    def set_joint_kd_params(self, arm, K, D):
        return self._call("set_joint_kd_params", arm=arm, K=list(K), D=list(D))

    def set_drag_space(self, arm, dgType):
        return self._call("set_drag_space", arm=arm, dgType=dgType)

    def set_joint_cmd_pose(self, arm, joints):
        return self._call("set_joint_cmd_pose", arm=arm, joints=list(joints))

    def set_cyr_kd_params(self, K, D):
        return self._call("set_cyr_kd_params", K=list(K), D=list(D))

    def set_cyr_state_on(self, state):
        return self._call("set_cyr_state_on", state=state)

    def set_cyr_state_off(self, state):
        return self._call("set_cyr_state_off", state=state)

    def set_tool(self, arm, kineParams, dynamicParams):
        return self._call("set_tool", arm=arm, kineParams=list(kineParams), dynamicParams=list(dynamicParams))

    def set_485_data(self, arm, data, size_int, com):
        # Don't convert data - pass hex string or bytes as-is
        return self._call("set_485_data", arm=arm, data=data, size_int=size_int, com=com)

    def get_485_data(self, arm, com):
        return self._call("get_485_data", arm=arm, com=com)

    def log_switch(self, flag):
        return self._call("log_switch", flag=flag)

    def local_log_switch(self, flag):
        return self._call("local_log_switch", flag=flag)



def _worker_main(conn, sdk_dir):
    """Runs in child process — no torch here."""
    import signal

    # Ignore KeyboardInterrupt in the worker — let parent handle cleanup
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    sys.path.insert(0, sdk_dir)
    from fx_robot import Marvin_Robot, DCSS

    sdk = Marvin_Robot()
    dcss = DCSS()

    while True:
        try:
            cmd, kwargs = conn.recv()
        except EOFError:
            break

        try:
            if cmd == "connect":
                result = sdk.connect(kwargs["ip"])
                conn.send(("ok", result))
            elif cmd == "clear_error":
                sdk.clear_error(kwargs["arm"])
                conn.send(("ok", None))
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
                sdk.set_state(arm=kwargs["arm"], state=kwargs["state"])
                conn.send(("ok", None))
            elif cmd == "set_vel_acc":
                sdk.set_vel_acc(arm=kwargs["arm"], velRatio=kwargs["velRatio"], AccRatio=kwargs["AccRatio"])
                conn.send(("ok", None))
            elif cmd == "set_impedance_type":
                sdk.set_impedance_type(arm=kwargs["arm"], type=kwargs["type"])
                conn.send(("ok", None))
            elif cmd == "set_joint_kd_params":
                sdk.set_joint_kd_params(arm=kwargs["arm"], K=kwargs["K"], D=kwargs["D"])
                conn.send(("ok", None))
            elif cmd == "set_drag_space":
                sdk.set_drag_space(arm=kwargs["arm"], dgType=kwargs["dgType"])
                conn.send(("ok", None))
            elif cmd == "set_joint_cmd_pose":
                sdk.set_joint_cmd_pose(arm=kwargs["arm"], joints=kwargs["joints"])
                conn.send(("ok", None))
            elif cmd == "set_cyr_kd_params":
                result = sdk.set_cyr_kd_params(K=kwargs["K"], D=kwargs["D"])
                conn.send(("ok", result))
            elif cmd == "set_cyr_state_on":
                result = sdk.set_cyr_state_on(state=kwargs["state"])
                conn.send(("ok", result))
            elif cmd == "set_cyr_state_off":
                result = sdk.set_cyr_state_off(state=kwargs["state"])
                conn.send(("ok", result))
            elif cmd == "set_tool":
                result = sdk.set_tool(arm=kwargs["arm"], kineParams=kwargs["kineParams"], dynamicParams=kwargs["dynamicParams"])
                conn.send(("ok", result))
            elif cmd == "set_485_data":
                result = sdk.set_485_data(arm=kwargs["arm"], data=kwargs["data"], size_int=kwargs["size_int"], com=kwargs["com"])
                conn.send(("ok", result))
            elif cmd == "get_485_data":
                result = sdk.get_485_data(arm=kwargs["arm"], com=kwargs["com"])
                conn.send(("ok", result))
            elif cmd == "log_switch":
                result = sdk.log_switch(flag=kwargs["flag"])
                conn.send(("ok", result))
            elif cmd == "local_log_switch":
                result = sdk.local_log_switch(flag=kwargs["flag"])
                conn.send(("ok", result))
            elif cmd == "release_robot":
                sdk.release_robot()
                conn.send(("ok", None))
                break
            else:
                conn.send(("error", f"Unknown command: {cmd}"))
        except Exception as e:
            conn.send(("error", str(e)))


# ── pool API ─────────────────────────────────────────────────────────────────

def get_sdk(ip: str):
    """Return (proxy, lock, already_connected)."""
    already_connected = ip in _proxies
    if not already_connected:
        _proxies[ip] = _SDKProxy()
        _locks[ip] = threading.Lock()
        _refcount[ip] = 0
        _latest_data[ip] = {}
    _refcount[ip] += 1
    return _proxies[ip], _locks[ip], already_connected


def set_latest_data(ip: str, data: dict):
    _latest_data[ip] = data


def get_latest_data(ip: str) -> dict:
    return _latest_data.get(ip, {})


def is_shared(ip: str) -> bool:
    """True when more than one component (e.g. teleop + robot) shares this SDK connection."""
    return _refcount.get(ip, 0) > 1


def release_sdk(ip: str, disable_cyr: bool = False, power_down: bool = False):
    """
    Release SDK reference and cleanup when refcount drops to 0.

    Args:
        ip: Robot IP address
        disable_cyr: Whether to disable CYR teleoperation mode before releasing
        power_down: Whether to power down arms before releasing
    """
    import logging
    logger = logging.getLogger(__name__)

    if ip not in _refcount:
        logger.warning(f"release_sdk called for {ip} but no refcount found")
        return

    _refcount[ip] -= 1
    logger.info(f"release_sdk: ip={ip}, new refcount={_refcount[ip]}, disable_cyr={disable_cyr}, power_down={power_down}")

    if _refcount[ip] <= 0:
        logger.info(f"Last user disconnecting, performing cleanup...")

        # Perform cleanup BEFORE calling release_robot to avoid pipe errors
        if (disable_cyr or power_down) and ip in _proxies:
            proxy = _proxies[ip]
            lock = _locks[ip]

            with lock:  # CRITICAL: Use lock to serialize SDK calls
                try:
                    # Disable teleoperation mode
                    if disable_cyr:
                        logger.info("Disabling CYR teleoperation mode...")
                        proxy.clear_set()
                        proxy.set_cyr_state_on(state=0)  # Use set_cyr_state_on with state=0
                        proxy.send_cmd()
                        import time
                        time.sleep(0.2)
                        logger.info("CYR teleoperation mode disabled")

                    # Power down arms
                    if power_down:
                        logger.info("Powering down arms (state=0)...")
                        proxy.clear_set()
                        proxy.set_state(arm='A', state=0)
                        proxy.set_state(arm='B', state=0)
                        proxy.send_cmd()
                        import time
                        time.sleep(0.3)
                        logger.info("Arms powered down successfully")
                except Exception as e:
                    logger.error(f"Cleanup failed with exception: {e}", exc_info=True)

        # Now release the robot (may cause SDK to crash, but cleanup is already done)
        try:
            logger.info("Calling release_robot()...")
            _proxies[ip].release_robot()
            logger.info("release_robot() completed")
        except Exception as e:
            logger.warning(f"release_robot() failed (expected): {e}")

        del _proxies[ip]
        del _locks[ip]
        del _refcount[ip]
        del _latest_data[ip]
        logger.info("SDK resources cleaned up")
    else:
        logger.info(f"Not last user (refcount={_refcount[ip]}), skipping cleanup")


def get_gripper_imports():
    sdk_dir = _SDK_DIR
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    from KM_CAN import KMGripperControl, Motor, KM_Motor_Type, Control_Type
    return KMGripperControl, Motor, KM_Motor_Type, Control_Type
