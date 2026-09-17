"""Native TileIR acquire/release message passing between independent CTAs."""
import pytest
import torch
import triton as tr
import triton.language as tl
from triton.runtime.driver import driver


@tr.jit
def _message(data, flags, observed, ROUNDS: tl.constexpr):
    role = tl.program_id(0)
    for round in range(ROUNDS):
        if role == 0:
            spins = 0
            while tl.atomic_load(flags + 1, sem="acquire") != round and spins < 100000:
                spins += 1
            tl.store(data, round * 17 + 3)
            tl.atomic_store(flags, round + 1, sem="release")
        else:
            spins = 0
            while tl.atomic_load(flags, sem="acquire") != round + 1 and spins < 100000:
                spins += 1
            value = tl.load(data)
            tl.store(observed + round, value)
            tl.atomic_store(flags + 1, round + 1, sem="release")


def test_tileir_atomic_message_passing():
    if type(driver.active).__name__ != "TileIRDriver":
        pytest.skip("requires the TileIR backend")
    rounds = 32
    data = torch.zeros(1, dtype=torch.int32, device="cuda")
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    observed = torch.full((rounds,), -1, dtype=torch.int32, device="cuda")
    _message[(2,)](data, flags, observed, rounds)
    torch.testing.assert_close(observed, torch.arange(rounds, device="cuda", dtype=torch.int32) * 17 + 3)
    torch.testing.assert_close(flags, torch.full_like(flags, rounds))
