from __future__ import annotations
import os

import os

from ..backends import backends, DriverBase


def _create_driver() -> DriverBase:
    # 1) Explicit selection by backend name.
    selected = os.environ.get("TRITON_DEFAULT_BACKEND", None)
    if selected:
        if selected not in backends:
            raise RuntimeError(f"Unknown backend device '{selected}'. Available backends: {list(backends.keys())}")
        driver_cls = backends[selected].driver
        if not driver_cls.is_active():
            raise RuntimeError(f"Backend device '{selected}' is not active.")
        return driver_cls()

    # 2) Explicit opt-in to CUDA Tile IR (tileir).
    if os.environ.get("ENABLE_TILE", "0") == "1":
        from ..backends.tileir.driver import TileIRDriver
        return TileIRDriver()

    # 3) Otherwise auto-select from active drivers.
    active_driver_classes = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(active_driver_classes) == 1:
        return active_driver_classes[0]()
    if len(active_driver_classes) == 0:
        raise RuntimeError("No active Triton backend drivers found.")

    # Multiple active drivers: apply a deterministic preference.
    # Prefer CUDA (nvidia) if available, then CUDA Tile IR (tileir) if available.
    try:
        from ..backends.nvidia.driver import CudaDriver
        if any((dc is CudaDriver for dc in active_driver_classes)):
            return CudaDriver()
    except Exception:
        pass

    try:
        from ..backends.tileir.driver import TileIRDriver
        if any((dc is TileIRDriver for dc in active_driver_classes)):
            return TileIRDriver()
    except Exception:
        pass

    raise RuntimeError(
        f"{len(active_driver_classes)} active drivers ({active_driver_classes}). "
        "Set TRITON_DEFAULT_BACKEND to select a backend, set ENABLE_TILE=1 to force CUDA Tile IR (tileir), "
        "or call triton.runtime.driver.set_active(...) before use."
    )


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
