"""Memory budgets for tiling: free GPU memory, available system RAM, or ``SYNAPSE_SR_MEMORY`` (GB)."""

import os
import sys

CPU_MAX = 6e9           # larger CPU windows stop paying off; the RAM budget never exceeds this
CPU_MIN = 0.5e9


def override():
    """``SYNAPSE_SR_MEMORY`` in GB, or None."""
    v = os.environ.get("SYNAPSE_SR_MEMORY")
    if not v:
        return None
    try:
        gb = float(v)
    except ValueError:
        raise ValueError(f"SYNAPSE_SR_MEMORY must be a number of GB, got {v!r}") from None
    if gb <= 0:
        raise ValueError(f"SYNAPSE_SR_MEMORY must be positive, got {v!r}")
    return gb * 1e9


def available_ram():
    """Available system RAM in bytes, or None when it cannot be read."""
    try:
        import psutil
        return float(psutil.virtual_memory().available)
    except ImportError:
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) * 1024
        elif sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("total_page", ctypes.c_ulonglong),
                            ("avail_page", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                            ("avail_virtual", ctypes.c_ulonglong), ("avail_ext", ctypes.c_ulonglong)]
            s = Status()
            s.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                return float(s.avail)
        elif sys.platform == "darwin":
            import subprocess
            total = float(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                         timeout=5).stdout)
            return 0.5 * total                             # macOS compresses memory; half the total is a safe guess
    except Exception:                                      # unreadable: callers fall back to a fixed budget
        return None
    return None


def cpu_budget():
    """Working budget for CPU tiles: half the available RAM, within [0.5, 6] GB."""
    o = override()
    if o is not None:
        return o
    ram = available_ram()
    if ram is None:
        return CPU_MAX
    return min(CPU_MAX, max(CPU_MIN, 0.5 * ram))
