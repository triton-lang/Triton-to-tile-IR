from __future__ import annotations
import os

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    # If tile is explicitly enabled, force CuTileDriver
    if os.environ.get("ENABLE_TILE", "0") == "1":
        from ..backends.cutile.driver import CuTileDriver
        return CuTileDriver()

    # Otherwise, auto-select from active drivers
    active_driver_classes = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(active_driver_classes) == 1:
        return active_driver_classes[0]()
    if len(active_driver_classes) == 0:
        raise RuntimeError("No active Triton backend drivers found.")

    # Multiple active drivers: apply a deterministic preference
    # 1) Prefer CUDA (nvidia) if available
    try:
        from ..backends.nvidia.driver import CudaDriver
        for dc in active_driver_classes:
            if dc is CudaDriver:
                return CudaDriver()
    except Exception:
        pass

    # 2) Then prefer triton-cuda-tile if available
    try:
        from ..backends.cutile.driver import CuTileDriver
        for dc in active_driver_classes:
            if dc is CuTileDriver:
                return CuTileDriver()
    except Exception:
        pass

    # 3) Fallback: pick the first and warn via exception message to guide users
    raise RuntimeError(f"{len(active_driver_classes)} active drivers ({active_driver_classes}). "
                       "Set ENABLE_TILE=1 to force cutile or call "
                       "triton.runtime.driver.set_active(...) before use.")


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default


driver = DriverConfig()
