"""Tests for smoke-status, label-domain, finiteness, and terminal-state logic."""

from tools.borealtc.runtime import _rt


def _good_measurements() -> dict:
    return {
        "train_loss": 1.5,
        "val_loss": 1.4,
        "grad_global_norm": 3.2,
        "param_update_norm": 0.01,
        "optimizer_steps": 1,
        "backward_calls": 1,
        "labels_in_domain": True,
        "logits_finite": True,
        "inputs_finite": True,
        "grads_all_finite": True,
        "params_finite_after_step": True,
    }


def test_finite_rejects_nan_inf_none_bool():
    assert _rt.finite(1.0)
    assert _rt.finite(0)
    assert not _rt.finite(float("nan"))
    assert not _rt.finite(float("inf"))
    assert not _rt.finite(None)
    assert not _rt.finite(True)
    assert not _rt.finite("1.0e3x")


def test_labels_in_domain():
    assert _rt.labels_in_domain([0, 1, 4])
    assert _rt.labels_in_domain([2.0])  # integral float is acceptable
    assert not _rt.labels_in_domain([5])
    assert not _rt.labels_in_domain([-1])
    assert not _rt.labels_in_domain([1.5])
    assert not _rt.labels_in_domain([])


def test_smoke_status_pass():
    assert _rt.smoke_status(_good_measurements()) == "PASS"


def test_smoke_status_failure_modes():
    m = _good_measurements()
    m["train_loss"] = float("nan")
    assert _rt.smoke_status(m) == "FAIL_NONFINITE_TRAIN_LOSS"

    m = _good_measurements()
    m["optimizer_steps"] = 2
    assert _rt.smoke_status(m) == "FAIL_OPTIMIZER_STEP_COUNT"

    m = _good_measurements()
    m["backward_calls"] = 0
    assert _rt.smoke_status(m) == "FAIL_BACKWARD_CALL_COUNT"

    m = _good_measurements()
    m["param_update_norm"] = 0.0
    assert _rt.smoke_status(m) == "FAIL_NO_PARAMETER_UPDATE"

    m = _good_measurements()
    m["labels_in_domain"] = False
    assert _rt.smoke_status(m) == "FAIL_LABEL_DOMAIN"

    m = _good_measurements()
    m["grads_all_finite"] = False
    assert _rt.smoke_status(m) == "FAIL_NONFINITE_GRADIENTS"

    m = _good_measurements()
    del m["val_loss"]
    assert _rt.smoke_status(m) == "FAIL_NONFINITE_VAL_LOSS"


def test_terminal_state_logic():
    assert (
        _rt.terminal_state("PASS", "PASS", mamba_blocker_complete=False)
        == _rt.TERMINAL_PASS
    )
    assert (
        _rt.terminal_state("PASS", "FAIL_EXCEPTION", mamba_blocker_complete=True)
        == _rt.TERMINAL_PASS_BLOCKED_MAMBA
    )
    assert _rt.terminal_state("PASS", "FAIL_EXCEPTION", False).startswith(
        "FAIL_BOREALTC_RUNTIME_FEASIBILITY"
    )
    assert _rt.terminal_state("FAIL_X", "PASS", True).startswith(
        "FAIL_BOREALTC_RUNTIME_FEASIBILITY_CNN"
    )


def test_blocker_completeness():
    from tools.borealtc.runtime import mamba_blocker as mb

    complete = {
        "failed_environment_label": "FAITHFUL_COMPATIBILITY_STACK",
        "exact_command": "pip install mamba-ssm==1.2.0.post1",
        "compiler_identity": "gcc 11",
        "cuda_identity": "cuda 12.8",
        "torch_identity": "2.7.1+cu128",
        "gpu_identity": "RTX 5080, sm_120",
        "retained_logs": [{"path": "x.log", "sha256": "ab", "size_bytes": 1}],
        "first_root_cause_error": "error: ...",
        "dependency_chain": "mamba-ssm -> causal_conv1d -> nvcc",
        "workaround_semantics_analysis": "...",
        "cpu_only_analysis": "...",
        "recommended_terminal_substatus": "BLOCKED_MAMBA",
    }
    assert mb.is_blocker_complete(complete)
    for key in mb.REQUIRED_FIELDS:
        broken = dict(complete)
        broken[key] = "" if key != "retained_logs" else []
        assert not mb.is_blocker_complete(broken), key
    no_hash = dict(complete)
    no_hash["retained_logs"] = [{"path": "x.log", "sha256": "", "size_bytes": 1}]
    assert not mb.is_blocker_complete(no_hash)
