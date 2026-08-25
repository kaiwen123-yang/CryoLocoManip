"""Tests for the checkpoint-audit pure logic (no torch required)."""

from tools.borealtc.runtime import checkpoint_audit as ca


def test_param_count_and_prefix_summary():
    shapes = {
        "imu_in_layer.weight": (32, 6),
        "imu_in_layer.bias": (32,),
        "mamba_block_imu.mixer.in_proj.weight": (256, 32),
        "out_layer.weight": (5, 40),
        "out_layer.bias": (5,),
    }
    assert ca.param_count(shapes) == 32 * 6 + 32 + 256 * 32 + 5 * 40 + 5
    prefixes = ca.prefix_summary(shapes)
    assert prefixes["imu_in_layer"] == 32 * 6 + 32
    assert prefixes["mamba_block_imu"] == 256 * 32
    assert prefixes["out_layer"] == 5 * 40 + 5


def test_infer_constructor_args_marks_missing():
    ctor = ca.infer_constructor_args({"d_model_imu": 32, "num_classes": 5})
    assert ctor["d_model_imu"] == 32
    assert ctor["num_classes"] == 5
    assert ctor["ssm_cfg_imu"] == "__missing__"


def test_to_jsonable_reprs_unknown_types():
    class Odd:
        def __repr__(self):
            return "<odd>"

    out = ca.to_jsonable({"a": [1, Odd()], "b": {"c": None}})
    assert out == {"a": [1, "<odd>"], "b": {"c": None}}


def test_fold_evidence_detects_markers():
    hit = ca.fold_evidence(["checkpoints/model_fold_3-epoch=12.ckpt", "x"])
    assert hit["verdict"] == "FOLD_MARKER_PRESENT"
    assert hit["fold_marker_strings"] == ["checkpoints/model_fold_3-epoch=12.ckpt"]
    miss = ca.fold_evidence(["nothing", "here"])
    assert miss["verdict"] == "NO_FOLD_MARKER_IN_CHECKPOINT"


def test_compare_expected_matches_source_literals():
    semantics = {
        "training_configs": {
            "mamba_train.py": {
                "ssm_cfg_imu": {"d_state": 16, "d_conv": 4, "expand": 4},
                "ssm_cfg_pro": {"d_state": 16, "d_conv": 3, "expand": 6},
                "mamba_train_opt": {
                    "d_model_imu": 32,
                    "d_model_pro": 8,
                    "norm_epsilon": 6.3e-6,
                    "out_method": "last_state",
                    "init_learn_rate": 1.5e-3,
                },
            }
        }
    }
    ctor = {
        "d_model_imu": 32,
        "d_model_pro": 8,
        "norm_epsilon": 6.3e-6,
        "ssm_cfg_imu": {"d_state": 16, "d_conv": 4, "expand": 4},
        "ssm_cfg_pro": {"d_state": 16, "d_conv": 3, "expand": 6},
        "out_method": "last_state",
        "num_classes": 5,
        "lr": 1.5e-3,
    }
    cmp = ca.compare_expected(ctor, semantics)
    assert cmp["available"] is True
    assert all(row["match"] for row in cmp["fields"].values())
    ctor_bad = dict(ctor, d_model_imu=64)
    cmp_bad = ca.compare_expected(ctor_bad, semantics)
    assert cmp_bad["fields"]["d_model_imu"]["match"] is False
    assert ca.compare_expected(ctor, None) == {"available": False}


def test_string_scan_walks_nested_structures():
    out: list[str] = []
    ca._string_scan({"a": ["x", {"b": "y"}], "c": 3}, out)
    assert set(out) >= {"a", "x", "b", "y", "c"}
