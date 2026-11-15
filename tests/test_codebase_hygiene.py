# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for Issue #3: codebase hygiene, single authoritative classes, and retired testing mode."""

import ast
import importlib
import pathlib

import pytest

from src import cli


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def test_every_src_module_imports_cleanly():
    """Every remaining Python module under src/ must import without errors or missing dependencies."""
    py_files = sorted(SRC_ROOT.rglob("*.py"))
    assert len(py_files) >= 35
    for py_file in py_files:
        rel = py_file.relative_to(REPO_ROOT)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mod_name = ".".join(parts)
        importlib.import_module(mod_name)


def test_no_backup_or_duplicate_core_classes():
    """Ensure no *_backup.py / *_3008.py files remain and core classes have a single definition."""
    py_files = sorted(SRC_ROOT.rglob("*.py"))
    bad_names = [
        p.name for p in py_files
        if "backup" in p.name or "3008" in p.name or p.name.startswith("teste_")
    ]
    assert bad_names == [], f"Orphaned backup/scratch files remain: {bad_names}"

    class_defs: dict[str, list[str]] = {
        "Decoy": [],
        "PercolatorProcessing": [],
        "TSVConverter": [],
        "AltCodons": [],
        "SpectralForest": [],
    }
    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in class_defs:
                class_defs[node.name].append(str(py_file.relative_to(REPO_ROOT)))

    for cls_name, locations in class_defs.items():
        assert len(locations) == 1, f"Expected 1 definition of {cls_name}, found {locations}"


def test_broken_testing_subcommand_retired(tmp_path):
    """Ensure 'testing' mode is cleanly rejected by argparse instead of crashing with KeyError."""
    parser, subparsers = cli.get_parsers()
    assert "testing" not in subparsers
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["testing", "--outdir", str(tmp_path)])
    assert excinfo.value.code == 2
