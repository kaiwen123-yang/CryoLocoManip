"""Tests for the ReduceLROnPlateau verbose-kwarg compatibility adapter."""

import inspect

from tools.borealtc.compat import torch_api


class _ModernScheduler:
    """Mimics torch>=2.7: no `verbose`, no **kwargs."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience


class _EraScheduler(_ModernScheduler):
    """Mimics torch<=2.2: accepts `verbose`."""

    def __init__(self, optimizer, mode="min", factor=0.1, patience=10, verbose=False):
        super().__init__(optimizer, mode=mode, factor=factor, patience=patience)
        self.verbose = verbose


def test_feature_detection():
    assert torch_api.scheduler_accepts_verbose(_EraScheduler)
    assert not torch_api.scheduler_accepts_verbose(_ModernScheduler)


def test_shim_drops_verbose_and_preserves_numerics():
    shim = torch_api.make_verbose_tolerant(_ModernScheduler)
    sched = shim("opt", mode="max", factor=0.25, patience=4, verbose=True)
    assert isinstance(sched, _ModernScheduler)
    assert (sched.mode, sched.factor, sched.patience) == ("max", 0.25, 4)
    assert not hasattr(sched, "verbose")
    assert shim.__name__ == _ModernScheduler.__name__

    plain = shim("opt", mode="max", factor=0.25, patience=4)
    assert (plain.mode, plain.factor, plain.patience) == ("max", 0.25, 4)


def test_shim_signature_matches_released_call_site():
    shim = torch_api.make_verbose_tolerant(_ModernScheduler)
    params = inspect.signature(shim.__init__).parameters
    assert "verbose" in params
