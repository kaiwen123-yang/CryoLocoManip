"""Repository-contract tests: path-template parsing and layout assumptions.

Standard-library only (unittest) and also collectable by pytest. The tests
read committed files exclusively and never depend on ignored machine-local
configuration such as configs/local/paths.local.yaml, installed GPUs,
simulators, or datasets.
"""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PROJECT_ROOT = "/mnt/g/CryoLocoManip"
EXPECTED_ORIGIN = "git@github.com:kaiwen123-yang/CryoLocoManip.git"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load_module("verify_workspace", REPO_ROOT / "scripts" / "verify_workspace.py")
PACKAGE = load_module(
    "cryolocomanip_init", REPO_ROOT / "src" / "cryolocomanip" / "__init__.py"
)


class PathTemplateParsingTests(unittest.TestCase):
    def test_example_template_parses_and_stays_on_gdrive(self):
        text = (REPO_ROOT / "configs" / "paths.local.example.yaml").read_text(
            encoding="utf-8"
        )
        config = VERIFY.parse_simple_yaml(text)
        self.assertEqual(config["project_root"], EXPECTED_PROJECT_ROOT)
        paths = config["paths"]
        self.assertIsInstance(paths, dict)
        expected_keys = {
            "data_root",
            "third_party_root",
            "simulator_root",
            "checkpoint_root",
            "cache_root",
            "run_root",
            "output_root",
            "local_config_root",
        }
        self.assertTrue(expected_keys.issubset(paths.keys()))
        for key, value in paths.items():
            self.assertTrue(
                VERIFY.is_within(value, EXPECTED_PROJECT_ROOT),
                f"paths.{key}={value} escapes {EXPECTED_PROJECT_ROOT}",
            )

    def test_parser_skips_comments_blank_lines_and_inline_comments(self):
        text = "\n".join(
            [
                "# leading comment",
                "",
                "project_root: /mnt/g/CryoLocoManip  # inline comment",
                "paths:",
                "  data_root: /mnt/g/CryoLocoManip/data",
                "",
            ]
        )
        config = VERIFY.parse_simple_yaml(text)
        self.assertEqual(config["project_root"], EXPECTED_PROJECT_ROOT)
        self.assertEqual(
            config["paths"]["data_root"], f"{EXPECTED_PROJECT_ROOT}/data"
        )

    def test_parser_supports_one_level_lists(self):
        text = (
            "allowed_external_roots:\n"
            "  - /mnt/g/SharedDatasets\n"
            "  - /mnt/g/OtherRoot\n"
        )
        config = VERIFY.parse_simple_yaml(text)
        self.assertEqual(
            config["allowed_external_roots"],
            ["/mnt/g/SharedDatasets", "/mnt/g/OtherRoot"],
        )

    def test_parser_rejects_orphan_indentation(self):
        with self.assertRaises(ValueError):
            VERIFY.parse_simple_yaml("  data_root: /mnt/g/x\n")


class ContainmentTests(unittest.TestCase):
    def test_root_and_children_are_within(self):
        self.assertTrue(
            VERIFY.is_within(EXPECTED_PROJECT_ROOT, EXPECTED_PROJECT_ROOT)
        )
        self.assertTrue(
            VERIFY.is_within(f"{EXPECTED_PROJECT_ROOT}/runs/a/b", EXPECTED_PROJECT_ROOT)
        )

    def test_linux_filesystem_paths_are_rejected(self):
        for path in (
            "/home/user/CryoLocoManip/data",
            "/root/CryoLocoManip",
            "/opt/CryoLocoManip",
            "/tmp/runs",
        ):
            self.assertFalse(VERIFY.is_within(path, EXPECTED_PROJECT_ROOT), path)

    def test_sibling_prefix_is_rejected(self):
        self.assertFalse(
            VERIFY.is_within("/mnt/g/CryoLocoManipX/data", EXPECTED_PROJECT_ROOT)
        )

    def test_dot_segments_are_normalized(self):
        self.assertTrue(
            VERIFY.is_within(f"{EXPECTED_PROJECT_ROOT}/./data", EXPECTED_PROJECT_ROOT)
        )
        self.assertFalse(
            VERIFY.is_within(f"{EXPECTED_PROJECT_ROOT}/../escape", EXPECTED_PROJECT_ROOT)
        )


class RepositoryLayoutTests(unittest.TestCase):
    def test_contract_constants_match_docs(self):
        self.assertEqual(VERIFY.CANONICAL_ROOT, EXPECTED_PROJECT_ROOT)
        self.assertEqual(VERIFY.EXPECTED_ORIGIN, EXPECTED_ORIGIN)

    def test_committed_layout_exists(self):
        for rel in (
            "README.md",
            "docs/PROJECT_CONTRACT.md",
            "docs/STAGE0_PLAN.md",
            "docs/REPRODUCIBILITY_CONTRACT.md",
            "docs/REPOSITORY_LAYOUT.md",
            "configs/paths.local.example.yaml",
            "prompts/CODEX_00_WSL_BOOTSTRAP.md",
            "scripts/preflight_wsl.sh",
            "scripts/verify_workspace.py",
            "scripts/collect_environment.sh",
            "src/cryolocomanip/__init__.py",
            "schemas/run_manifest.schema.json",
            "literature/README.md",
            "pyproject.toml",
        ):
            self.assertTrue((REPO_ROOT / rel).is_file(), f"missing committed file: {rel}")

    def test_gitignore_excludes_local_and_runtime_paths(self):
        lines = {
            line.strip()
            for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
        for pattern in (
            "configs/local/",
            "*.local.yaml",
            "/data/",
            "/third_party/",
            "/simulators/",
            "/checkpoints/",
            "/runs/",
            "/outputs/",
            "/artifacts/runtime/",
            "/logs/",
        ):
            self.assertIn(pattern, lines, f".gitignore must exclude {pattern}")

    def test_package_version_is_semver_like(self):
        self.assertRegex(PACKAGE.__version__, r"^\d+\.\d+\.\d+$")


class RunManifestSchemaTests(unittest.TestCase):
    def test_schema_is_valid_json_and_covers_contract_minimum(self):
        schema = json.loads(
            (REPO_ROOT / "schemas" / "run_manifest.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema.get("type"), "object")
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        for field in (
            "stage",
            "method",
            "run_id",
            "git",
            "command",
            "working_directory",
            "machine",
            "versions",
            "inputs",
            "seeds",
            "started_at_utc",
            "ended_at_utc",
            "exit_code",
            "exclusions",
            "parent_run_id",
        ):
            self.assertIn(field, required)
            self.assertIn(field, properties)
        self.assertIn("commit", properties["git"].get("required", []))
        self.assertIn("dirty", properties["git"].get("required", []))
        versions_required = set(properties["versions"].get("required", []))
        self.assertTrue(
            {"os", "kernel", "python", "cuda", "gpu_driver", "ros", "simulator"}.issubset(
                versions_required
            )
        )


if __name__ == "__main__":
    unittest.main()
