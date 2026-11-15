import argparse
import ast
import os
import pathlib
import stat
import sys

import pytest

from src import cli
from src.cli import _types
from src.utils import (
    REPO_ROOT,
    SubprocessExecutionError,
    ToolNotFoundError,
    concat_files,
    ensure_dir,
    resolve_executable,
    resolve_free2bind_script,
    resolve_msgf_jar,
    run_cmd,
)


def test_zero_os_system_or_shell_true_in_src():
    """Audit M4 invariant: zero os.system() or subprocess(..., shell=True) in src/."""
    src_root = REPO_ROOT / "src"
    offenders: list[str] = []

    for py_file in sorted(src_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        rel = py_file.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # Detect os.system(...)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "system"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
            ):
                offenders.append(f"{rel}:{node.lineno} calls os.system()")
            # Detect shell=True on any call
            for kw in node.keywords:
                if (
                    kw.arg == "shell"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    offenders.append(f"{rel}:{node.lineno} passes shell=True")

    assert offenders == [], f"Unsafe shell invocations found: {offenders}"


@pytest.mark.parametrize("bad_val", ["", "   ", "maybe", "2", "yep"])
def test_yes_or_no_action_rejects_empty_and_invalid_tokens(bad_val):
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", dest="transcriptome", action=_types.YesOrNoBooleanAction)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["-T", bad_val])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "token,expected",
    [
        ("YES", True),
        ("yes", True),
        ("TRUE", True),
        ("true", True),
        ("NO", False),
        ("no", False),
        ("FALSE", False),
        ("false", False),
    ],
)
def test_yes_or_no_action_accepts_boolean_tokens(token, expected):
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", dest="transcriptome", action=_types.YesOrNoBooleanAction)
    parsed = parser.parse_args(["-T", token])
    assert parsed.transcriptome is expected


@pytest.mark.parametrize("good_prob", ["0.0001", "0.01", "0.5", "1", "1.0"])
def test_probability_type_valid(good_prob):
    val = _types.Probability(good_prob)
    assert 0.0 < val <= 1.0


@pytest.mark.parametrize("bad_prob", ["0", "0.0", "-0.01", "1.01", "nan", "inf", "abc"])
def test_probability_type_invalid(bad_prob):
    with pytest.raises((TypeError, ValueError)):
        _types.Probability(bad_prob)


def test_ms_mod_resolves_to_absolute_path_and_help_strings_say_comma_separated(tmp_path):
    mod_file = tmp_path / "mods.txt"
    mod_file.write_text("C2H3N1O1,C,fix,any,Carbamidomethyl\n", encoding="utf-8")
    ms_dir = tmp_path / "mzml"
    ms_dir.mkdir()

    parser, subparsers = cli.get_parsers()
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        parsed = parser.parse_args(
            ["ms", "-o", "out", "-M", "mzml", "--mod", "mods.txt"]
        )
        assert isinstance(parsed.mod, pathlib.Path)
        assert parsed.mod.is_absolute()
        assert parsed.mod == mod_file.resolve()
    finally:
        os.chdir(cwd)

    for mode_name in ("database", "postms"):
        help_text = subparsers[mode_name].format_help()
        assert "Space separated list" not in help_text
        assert "Comma-separated list" in help_text


def test_resolve_msgf_jar_and_bioconda_discovery(tmp_path, monkeypatch):
    monkeypatch.delenv("UPROTEINS_MSGF_JAR", raising=False)

    # Simulate a Bioconda prefix: <prefix>/bin/msgf_plus and <prefix>/share/msgf_plus-2024/MSGFPlus.jar
    conda_prefix = tmp_path / "conda env with spaces"
    bin_dir = conda_prefix / "bin"
    share_dir = conda_prefix / "share" / "msgf_plus-2024.03.26-0"
    bin_dir.mkdir(parents=True)
    share_dir.mkdir(parents=True)

    wrapper = bin_dir / "msgf_plus"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    jar_file = share_dir / "MSGFPlus.jar"
    jar_file.write_bytes(b"PK\x03\x04fakejar")

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    assert resolve_msgf_jar() == jar_file.resolve()

    # Env var override takes precedence
    custom_jar = tmp_path / "custom_MSGFPlus.jar"
    custom_jar.write_bytes(b"PK\x03\x04custom")
    monkeypatch.setenv("UPROTEINS_MSGF_JAR", str(custom_jar))
    assert resolve_msgf_jar() == custom_jar.resolve()

    # Non-existent explicit jar raises ToolNotFoundError
    with pytest.raises(ToolNotFoundError):
        resolve_msgf_jar(tmp_path / "missing.jar")


def test_resolve_executable_and_free2bind_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("UPROTEINS_PERCOLATOR", raising=False)
    with pytest.raises(ToolNotFoundError):
        resolve_executable(
            "percolator_nonexistent_xyz",
            "percolator_nonexistent_xyz",
            env_var="UPROTEINS_PERCOLATOR",
            cli_flag="--percolator_path",
        )

    f2b_dir = tmp_path / "free2bind"
    f2b_dir.mkdir()
    script = f2b_dir / "free_align.pl"
    script.write_text("#!/usr/bin/env perl\nprint \"-6.5\\n\";\n", encoding="utf-8")
    assert resolve_free2bind_script(f2b_dir) == script.resolve()


def test_run_cmd_and_concat_files_handle_spaces_and_nonzero_exit(tmp_path):
    # Reject shell strings outright
    with pytest.raises(TypeError):
        run_cmd("echo unsafe")  # type: ignore[arg-type]

    # Non-zero exit raises SubprocessExecutionError
    with pytest.raises(SubprocessExecutionError) as exc:
        run_cmd([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(7)"])
    assert exc.value.returncode == 7

    # Paths with spaces & metacharacters work cleanly without shell
    weird_dir = ensure_dir(tmp_path / "dir with spaces & (parens); $var")
    f1 = weird_dir / "part 1.txt"
    f2 = weird_dir / "part 2.txt"
    out = weird_dir / "combined.txt"
    f1.write_text("alpha\n", encoding="utf-8")
    f2.write_text("beta\n", encoding="utf-8")
    concat_files([f1, f2], out)
    assert out.read_text(encoding="utf-8") == "alpha\nbeta\n"
