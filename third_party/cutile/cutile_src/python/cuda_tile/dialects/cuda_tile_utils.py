from .cuda_tile_ops import *


def mutex_synchronize(mutex_ptr, critical_section):
    """
    Helper function that tries to acquire the mutex at the specified global
    memory location and then executes the critical section. The memory
    location is a !cuda_tile.tile<cuda_tile.ptr<i32>>. A value of "1"
    indicates that the mutex is available. A value of "0" indicates that
    another block tile currently holds the mutex.
    """

    c0 = constant(0)
    c1 = constant(1)

    # Acquire the mutex.
    def busy_loop():
        prev = atomic_cas_tko(
            MemoryOrderingSemantics.RELAXED,
            MemoryScope.DEVICE,
            mutex_ptr,
            cmp=c1,
            val=c0,
        )
        # Exit busy loop if the mutex was acquired.
        prev_i1 = trunci(Boolean, prev)
        if_generate(prev_i1, lambda: loop_break([]))
        loop_continue([])

    loop_generate([], busy_loop)

    # Execute the critical section.
    critical_section()

    # Release the mutex.
    atomic_rmw_tko(
        MemoryOrderingSemantics.RELAXED,
        MemoryScope.DEVICE,
        mutex_ptr,
        AtomicRMWMode.XCHG,
        c1,
    )
