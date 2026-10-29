from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HardwareProfile:
    cpu: str
    logical_cpus: int
    ram_bytes: int | None
    gpu: str | None
    vram_bytes: int | None
    multiprocessors: int | None
    compute_capability: str | None
    recommended_population: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _windows_ram() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.total_physical)


def detect_hardware() -> HardwareProfile:
    cpu = platform.processor() or platform.machine()
    logical_cpus = os.cpu_count() or 1
    ram_bytes = _windows_ram()
    gpu = None
    vram = None
    multiprocessors = None
    capability = None
    recommended = 4096
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(0)
        raw_name = properties["name"]
        gpu = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        vram = int(properties["totalGlobalMem"])
        multiprocessors = int(properties["multiProcessorCount"])
        capability = f"{int(properties['major'])}.{int(properties['minor'])}"
        # Each P/Q state is tiny; throughput, host generation, and checkpoint
        # latency matter more than raw capacity. Scale conservatively by SMs.
        recommended = 1 << max(12, min(16, (multiprocessors * 2048).bit_length() - 1))
    except (ImportError, RuntimeError, OSError):
        pass
    return HardwareProfile(
        cpu,
        logical_cpus,
        ram_bytes,
        gpu,
        vram,
        multiprocessors,
        capability,
        recommended,
    )
