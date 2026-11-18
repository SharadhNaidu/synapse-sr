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


DEVICE_BYTES_PER_PX = 3 * 4 * 25 * 4       # x, confidence, x_base: 4 bands of float32 on the 5x grid, per source pixel
HOST_BYTES_PER_PX = 4 * 4 * 25 * 4 + 25 * 2 + 10 * 4    # image, confidence, x_base, prior + support / valid + input
CONTEXT_BYTES_PER_PX = 6 * 25 * 2


def estimate(H, W, context=True):
    """(device bytes, host bytes) the output mosaic of an H x W source scene needs, excluding tile working memory."""
    host = HOST_BYTES_PER_PX + (CONTEXT_BYTES_PER_PX if context else 0)
    return float(DEVICE_BYTES_PER_PX * H * W), float(host * H * W)


def preflight(H, W, device, context=True):
    """Raise MemoryError before any work when the output of an H x W scene clearly cannot fit.
    ``SYNAPSE_SR_PREFLIGHT=0`` skips the check."""
    if os.environ.get("SYNAPSE_SR_PREFLIGHT", "1") in ("0", "false", "no"):
        return
    dev, host = estimate(H, W, context)
    need, have, where = host + (dev if device.type != "cuda" else 0), available_ram(), "RAM"
    if device.type == "cuda":
        import torch
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free = float(torch.cuda.mem_get_info(idx)[0])
        if dev > 0.9 * free:
            _fail(H, W, dev, free, "free GPU memory")
    if have is not None and need > 0.9 * have:
        _fail(H, W, need, have, f"available {where}")


def _fail(H, W, need, have, where):
    side = int((0.9 * have / (need / (H * W))) ** 0.5)
    raise MemoryError(f"a {H} x {W} px scene needs about {need / 1e9:.1f} GB for its 2 m output but only "
                      f"{have / 1e9:.1f} GB of {where} is there. Process it in parts of at most ~{side} x {side} px "
                      f"(super_resolve(..., window=...) or bounds=, CLI --window / --bounds), pass context=False, or set "
                      f"SYNAPSE_SR_PREFLIGHT=0 to try anyway.")
