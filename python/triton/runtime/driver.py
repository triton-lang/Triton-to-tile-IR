from __future__ import annotations

import os
import functools

from ..backends import backends, DriverBase


def _is_tileir_enabled(default=False):
    selected = os.environ.get("TRITON_DEFAULT_BACKEND")
    if selected:
        return selected == "tileir"
    return os.environ.get("ENABLE_TILE", "0") == "1" or default


@functools.lru_cache()
def _get_backend_driver(name):
    if name == "tileir":
        from ..backends.tileir.driver import get_tileir_driver
        return get_tileir_driver()
    return backends[name].driver()


def _get_driver_for_target(target):
    matches = [name for name, backend in backends.items() if backend.compiler.supports_target(target)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one driver for target {target}, got {matches}")
    # Respect an already selected compatible driver (including its stream
    # integration), without initializing a different environment-selected one.
    active = driver._active
    if active is not None and isinstance(active, backends[matches[0]].driver):
        return active
    return _get_backend_driver(matches[0])


def _create_driver() -> DriverBase:
    selected = os.environ.get("TRITON_DEFAULT_BACKEND")
    if selected:
        if selected not in backends:
            raise RuntimeError(f"Unknown backend device '{selected}'. Available backends: {list(backends.keys())}")
        if not backends[selected].driver.is_active():
            raise RuntimeError(f"Backend device '{selected}' is not active.")
        return _get_backend_driver(selected)

    if _is_tileir_enabled():
        return _get_backend_driver("tileir")

    active = [name for name, backend in backends.items() if backend.driver.is_active()]
    if len(active) != 1:
        raise RuntimeError(f"{len(active)} active drivers ({active}). There should only be one.")
    return _get_backend_driver(active[0])


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
