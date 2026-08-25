"""Tests for the Stage 0B.2A source-runtime semantics parsers."""

from pathlib import Path

from tools.borealtc.runtime import source_semantics as ss

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party" / "BorealTC"


def test_parse_requirement_line_exact():
    parsed = ss.parse_requirement_line("numpy==1.26.4")
    assert parsed == {
        "raw": "numpy==1.26.4",
        "name": "numpy",
        "spec_type": "exact",
        "version": "1.26.4",
    }


def test_parse_requirement_line_unpinned_and_range():
    assert ss.parse_requirement_line("torch")["spec_type"] == "unpinned"
    ranged = ss.parse_requirement_line("matplotlib>=3.7.0")
    assert ranged["spec_type"] == "range"
    assert ranged["name"] == "matplotlib"


def test_parse_requirement_line_git_and_comments():
    git = ss.parse_requirement_line("git+https://github.com/willGuimont/pipeline")
    assert git["spec_type"] == "git_unpinned"
    assert git["name"] == "pipeline"
    pinned = ss.parse_requirement_line("git+https://github.com/a/b@0123abc")
    assert pinned["spec_type"] == "git_pinned"
    assert ss.parse_requirement_line("# comment") is None
    assert ss.parse_requirement_line("   ") is None


def test_classify_lock_flags_material_underdetermined():
    entries = ss.parse_requirements("numpy==1.0\ntorch\nmatplotlib>=3.7\n")
    cls = ss.classify_lock(entries)
    assert cls["exact_count"] == 1
    assert "torch" in cls["material_underdetermined"]
    assert "matplotlib" not in cls["material_underdetermined"]


def test_extract_dockerfile_facts_arch_list():
    text = (
        'FROM docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04\n'
        'ENV TZ=X \\\n'
        '    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"\n'
        "RUN pip3 install packaging torch torchvision wheel\n"
        "COPY requirements-lock.txt ./requirements.txt\n"
    )
    facts = ss.extract_dockerfile_facts(text)
    assert facts["base_image"] == "docker.io/nvidia/cuda:12.2.0-devel-ubuntu22.04"
    assert facts["arch_list_max_sm"] == 8.6
    assert facts["arch_list_has_ptx"] is True
    assert facts["requirements_file_copied_as"] == (
        "requirements-lock.txt -> requirements.txt"
    )
    assert any("torch" in line for line in facts["pip_torch_install_lines"])


def test_extract_literal_configs_handles_non_literals():
    src = (
        "X = {'a': 1, 'b': len(items), 'c': None}\n"
        "RANDOM_STATE = 21\n"
        "OTHER = 3\n"
    )
    out = ss.extract_literal_configs(src, ("X", "RANDOM_STATE"))
    assert out["X"]["a"] == 1
    assert out["X"]["b"] == {"__expr__": "len(items)"}
    assert out["X"]["c"] is None
    assert out["RANDOM_STATE"] == 21
    assert "OTHER" not in out


def test_upstream_lock_leaves_torch_unpinned():
    entries = ss.parse_requirements(
        (UPSTREAM / "requirements-lock.txt").read_text(encoding="utf-8")
    )
    by_name = {e["name"]: e for e in entries}
    assert by_name["torch"]["spec_type"] == "unpinned"
    assert by_name["torchvision"] == {
        "raw": "torchvision==0.17.0",
        "name": "torchvision",
        "spec_type": "exact",
        "version": "0.17.0",
    }
    assert by_name["mamba-ssm"]["version"] == "1.2.0.post1"
    assert by_name["causal_conv1d"]["version"] == "1.2.0.post1"
    assert by_name["pipeline"]["spec_type"] == "git_unpinned"


def test_upstream_dockerfile_arch_list_ends_at_ampere():
    facts = ss.extract_dockerfile_facts(
        (UPSTREAM / "Dockerfile").read_text(encoding="utf-8")
    )
    assert facts["arch_list_max_sm"] == 8.6
    assert facts["arch_list_has_ptx"] is True


def test_upstream_training_configs_extracted():
    cfg = ss.extract_literal_configs(
        (UPSTREAM / "mamba_train.py").read_text(encoding="utf-8"),
        ("ssm_cfg_imu", "ssm_cfg_pro", "mamba_train_opt", "RANDOM_STATE"),
    )
    assert cfg["RANDOM_STATE"] == 21
    assert cfg["ssm_cfg_imu"] == {"d_state": 16, "d_conv": 4, "expand": 4}
    assert cfg["ssm_cfg_pro"] == {"d_state": 16, "d_conv": 3, "expand": 6}
    assert cfg["mamba_train_opt"]["d_model_imu"] == 32
    assert cfg["mamba_train_opt"]["norm_epsilon"] == 6.3e-6
    assert cfg["mamba_train_opt"]["num_classes"] == {"__expr__": "len(terrains)"}


def test_smoke_configs_match_upstream_sources():
    """The transcribed smoke configurations must equal the released literals."""
    from tools.borealtc.runtime import smoke

    mamba_cfg = ss.extract_literal_configs(
        (UPSTREAM / "mamba_train.py").read_text(encoding="utf-8"),
        ("ssm_cfg_imu", "ssm_cfg_pro", "mamba_train_opt"),
    )
    assert smoke.MAMBA_SSM_CFG_IMU == mamba_cfg["ssm_cfg_imu"]
    assert smoke.MAMBA_SSM_CFG_PRO == mamba_cfg["ssm_cfg_pro"]
    for key, value in smoke.MAMBA_TRAIN_OPT.items():
        if key == "num_classes":
            continue  # upstream: len(terrains) == 5 official classes
        assert mamba_cfg["mamba_train_opt"][key] == value, key

    cnn_cfg = ss.extract_literal_configs(
        (UPSTREAM / "cnn_train.py").read_text(encoding="utf-8"),
        ("cnn_par", "cnn_train_opt"),
    )
    for key, value in smoke.CNN_PAR.items():
        if key == "num_classes":
            continue
        if key == "num_filters":
            assert cnn_cfg["cnn_par"][key] == {"__expr__": "16 * 2"} or (
                cnn_cfg["cnn_par"][key] == 32
            )
            assert value == 32
            continue
        assert cnn_cfg["cnn_par"][key] == value, key
    for key, value in smoke.CNN_TRAIN_OPT.items():
        assert cnn_cfg["cnn_train_opt"][key] == value, key
