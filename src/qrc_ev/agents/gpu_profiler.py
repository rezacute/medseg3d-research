"""GPU profiling context manager for CUDA-Q / NVIDIA GPU workloads.

Tracks:
  - Peak VRAM usage
  - Wall-clock time
  - Throughput (operations/second)

Usage:
    with GPUProfiler("reservoir_step") as p:
        result = reservoir.process_batch(batch)
    p.report()  # prints formatted report
"""

from __future__ import annotations
import time
import sys
from typing import Optional


class GPUProfiler:
    """Context manager for profiling GPU operations.

    Args:
        name: Label for this profiling session.
        device: GPU device index. Default 0.
        verbose: If True, prints report on exit.
    """

    _instance_count = 0

    def __init__(self, name: str = "GPUOp", device: int = 0, verbose: bool = True):
        self.name = name
        self.device = device
        self.verbose = verbose
        self._id = GPUProfiler._instance_count
        GPUProfiler._instance_count += 1

        # Try pynvml first, then cupy, then None
        self._nvml: Optional[object] = None
        self._handle = None
        self._cupy = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        except Exception:
            try:
                import cupy as cp
                self._cupy = cp
            except Exception:
                pass

        # Timing
        self._start_wall: Optional[float] = None
        self._end_wall: Optional[float] = None

        # VRAM
        self._start_mem: Optional[int] = None   # bytes
        self._peak_mem: int = 0                  # bytes

        # Operations counter
        self._operation_count: int = 0
        self._start_time_for_ops: Optional[float] = None

    def __enter__(self):
        self._start_wall = time.perf_counter()
        if self._nvml is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._start_mem = info.used
        elif self._cupy is not None:
            self._cupy.cuda.Device(self.device).use()
            mempool = self._cupy.get_default_memory_pool()
            self._start_mem = mempool.used_bytes()
        self._start_time_for_ops = self._start_wall
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_wall = time.perf_counter()

        # Update peak VRAM
        if self._nvml is not None:
            info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            current_mem = info.used
            self._peak_mem = max(self._peak_mem, current_mem)
        elif self._cupy is not None:
            self._cupy.cuda.Device(self.device).use()
            mempool = self._cupy.get_default_memory_pool()
            current_mem = mempool.used_bytes()
            self._peak_mem = max(self._peak_mem, current_mem)

        if self.verbose:
            self.report()
        return False

    def count_op(self, n: int = 1):
        """Increment operation counter by n."""
        self._operation_count += n

    def report(self):
        """Print formatted profiling report."""
        if self._start_wall is None:
            print(f"[{self.name}] Not started", file=sys.stderr)
            return

        elapsed = self._end_wall - self._start_wall if self._end_wall else 0.0
        throughput = self._operation_count / elapsed if elapsed > 0 else 0.0

        # VRAM delta
        if self._nvml is not None:
            peak_used = self._peak_mem / (1024**3)
            start_gb = self._start_mem / (1024**3)
            print(
                f"\n{'='*60}\n"
                f"  GPU Profiler: {self.name}\n"
                f"{'='*60}\n"
                f"  Wall time:       {elapsed*1000:.2f} ms\n"
                f"  Operations:      {self._operation_count}\n"
                f"  Throughput:      {throughput:.2f} ops/s\n"
                f"  VRAM start:      {start_gb:.2f} GB\n"
                f"  VRAM peak delta: {peak_used:.2f} GB\n"
                f"{'='*60}",
                file=sys.stdout
            )
        elif self._cupy is not None:
            peak_gb = self._peak_mem / (1024**3)
            start_gb = (self._start_mem or 0) / (1024**3)
            print(
                f"\n{'='*60}\n"
                f"  GPU Profiler: {self.name} (cupy)\n"
                f"{'='*60}\n"
                f"  Wall time:       {elapsed*1000:.2f} ms\n"
                f"  Operations:      {self._operation_count}\n"
                f"  Throughput:      {throughput:.2f} ops/s\n"
                f"  VRAM start:      {start_gb:.2f} GB\n"
                f"  VRAM peak delta: {peak_gb:.2f} GB\n"
                f"{'='*60}",
                file=sys.stdout
            )
        else:
            print(
                f"\n{'='*60}\n"
                f"  GPU Profiler: {self.name} (CPU fallback)\n"
                f"{'='*60}\n"
                f"  Wall time:       {elapsed*1000:.2f} ms\n"
                f"  Operations:      {self._operation_count}\n"
                f"  Throughput:      {throughput:.2f} ops/s\n"
                f"  VRAM:            (not available)\n"
                f"{'='*60}",
                file=sys.stdout
            )

    @property
    def elapsed_ms(self) -> float:
        """Elapsed wall time in milliseconds."""
        if self._start_wall is None:
            return 0.0
        end = self._end_wall if self._end_wall else time.perf_counter()
        return (end - self._start_wall) * 1000.0

    @property
    def peak_vram_gb(self) -> float:
        """Peak VRAM delta in GB."""
        return self._peak_mem / (1024**3)

    @property
    def throughput(self) -> float:
        """Operations per second."""
        elapsed = self._end_wall - self._start_wall if self._end_wall else 0.0
        return self._operation_count / elapsed if elapsed > 0 else 0.0


# Convenience decorator
def profile_gpu(name: str = None):
    """Decorator to profile a function with GPUProfiler.

    Usage:
        @profile_gpu("my_kernel")
        def my_function(...):
            ...
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            label = name or fn.__name__
            with GPUProfiler(label) as p:
                result = fn(*args, **kwargs)
                p.count_op()
                return result
        return wrapper
    return decorator
