"""torch API-glue adapters for the FAITHFUL_COMPATIBILITY_STACK.

Upstream `utils/models.py` constructs
`torch.optim.lr_scheduler.ReduceLROnPlateau(..., verbose=True)`. The
`verbose` keyword only controlled learning-rate-change printing; it was
deprecated in torch 2.2 and removed in torch 2.7, so the unmodified released
call site raises TypeError on the compatibility stack. The adapter below
temporarily replaces the scheduler class with a subclass that drops the
removed no-op keyword — numerical scheduler behavior (mode, factor,
patience, thresholds) is byte-for-byte the released configuration.

The patch is applied only when the installed torch actually rejects
`verbose` (feature detection), so era-matched environments run the released
code with no adapter at all.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager


def scheduler_accepts_verbose(scheduler_cls) -> bool:
    """Feature detection: does this ReduceLROnPlateau accept `verbose`?"""
    try:
        params = inspect.signature(scheduler_cls.__init__).parameters
    except (TypeError, ValueError):
        return True  # cannot introspect: assume released signature
    if "verbose" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def make_verbose_tolerant(scheduler_cls):
    """Subclass that drops the removed, print-only `verbose` keyword."""

    class VerboseTolerantReduceLROnPlateau(scheduler_cls):
        def __init__(self, *args, verbose=None, **kwargs):  # noqa: ARG002
            kwargs.pop("verbose", None)
            super().__init__(*args, **kwargs)

    VerboseTolerantReduceLROnPlateau.__name__ = scheduler_cls.__name__
    VerboseTolerantReduceLROnPlateau.__qualname__ = scheduler_cls.__qualname__
    return VerboseTolerantReduceLROnPlateau


@contextmanager
def released_scheduler_compat():
    """Context under which the released configure_optimizers() call sites
    work unchanged on torch builds that removed `verbose`.

    Yields a dict describing whether the adapter was active, for recording
    in run artifacts.
    """
    import torch  # noqa: PLC0415

    sched_mod = torch.optim.lr_scheduler
    original = sched_mod.ReduceLROnPlateau
    if scheduler_accepts_verbose(original):
        yield {"active": False, "reason": "torch still accepts verbose"}
        return
    sched_mod.ReduceLROnPlateau = make_verbose_tolerant(original)
    try:
        yield {
            "active": True,
            "reason": (
                "torch removed the print-only ReduceLROnPlateau verbose "
                "kwarg; adapter drops it without touching scheduler numerics"
            ),
        }
    finally:
        sched_mod.ReduceLROnPlateau = original
