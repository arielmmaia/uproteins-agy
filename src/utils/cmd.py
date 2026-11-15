# Copyright © 2021-2025 Eduardo Vieira de Souza
# Copyright © 2021-2025 Adriana Canedo
# Copyright © 2021-2025 Cristiano Valim Bizarro
#
# This file is part of uProteInS.
#
# uProteInS is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

"""Centralized external binary discovery, safe argv subprocess execution, and
pure-Python filesystem helpers (eliminating all shell-interpolated commands).
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import typing as t


REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]


class ToolNotFoundError(FileNotFoundError):
    """Raised when an external bioinformatics binary or JAR cannot be found."""


class SubprocessExecutionError(RuntimeError):
    """Raised when an external tool exits with a non-zero return code."""

    def __init__(
        self,
        argv: t.Sequence[str],
        returncode: int,
        stderr: str | None = None,
    ) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr or ""
        detail = f": {self.stderr.strip()}" if self.stderr.strip() else ""
        super().__init__(
            f"Command {self.argv[0]!r} failed with exit code {returncode}{detail}"
        )


def ensure_dir(path: str | os.PathLike[str]) -> pathlib.Path:
    """Create directory (including parents) idempotently without shell calls."""
    target = pathlib.Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def concat_files(
    inputs: t.Iterable[str | os.PathLike[str]],
    output: str | os.PathLike[str],
) -> pathlib.Path:
    """Deterministically concatenate multiple files into ``output`` in pure Python."""
    out_path = pathlib.Path(output)
    ensure_dir(out_path.parent)
    resolved_out = out_path.resolve()
    with open(out_path, "wb") as dst:
        for src in inputs:
            src_path = pathlib.Path(src)
            if src_path.resolve() == resolved_out:
                continue
            with open(src_path, "rb") as handle:
                shutil.copyfileobj(handle, dst)
    return out_path


def _check_candidate(candidate: str | os.PathLike[str]) -> pathlib.Path | None:
    text = str(candidate).strip()
    if not text:
        return None
    found = shutil.which(text)
    if found is not None:
        return pathlib.Path(found).resolve()
    p = pathlib.Path(text)
    if p.is_file() and os.access(p, os.X_OK):
        return p.resolve()
    return None


def resolve_executable(
    configured: str | os.PathLike[str] | None,
    default_name: str,
    *,
    env_var: str | None = None,
    extra_candidates: t.Sequence[str | os.PathLike[str]] = (),
    cli_flag: str | None = None,
) -> pathlib.Path:
    """Resolve an external executable from CLI flag, env var, PATH, or fallbacks.

    Raises :class:`ToolNotFoundError` with actionable remediation hints when
    the executable is missing or non-executable.
    """
    # 1. Explicit non-default CLI argument
    if configured is not None and str(configured) != default_name:
        resolved = _check_candidate(configured)
        if resolved is not None:
            return resolved
        raise ToolNotFoundError(
            f"Configured executable {str(configured)!r} for {default_name!r} "
            "does not exist or is not executable."
        )

    # 2. Environment variable override
    if env_var:
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            resolved = _check_candidate(env_val)
            if resolved is not None:
                return resolved
            raise ToolNotFoundError(
                f"Environment variable {env_var}={env_val!r} for {default_name!r} "
                "does not point to a valid executable."
            )

    # 3. Standard PATH lookup (covers default CLI value as well)
    resolved = _check_candidate(default_name)
    if resolved is not None:
        return resolved

    # 4. Bundled / legacy fallback locations
    for extra in extra_candidates:
        resolved = _check_candidate(extra)
        if resolved is not None:
            return resolved

    hints = [f"install {default_name!r} on PATH (e.g. via conda environment.yml)"]
    if env_var:
        hints.append(f"set ${env_var}")
    if cli_flag:
        hints.append(f"pass {cli_flag}")
    raise ToolNotFoundError(
        f"Required external executable {default_name!r} was not found; "
        + ", or ".join(hints)
        + "."
    )


def resolve_msgf_jar(
    configured_jar: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Locate ``MSGFPlus.jar`` across CLI flag, env var, Bioconda share/, and legacy dirs."""
    if configured_jar is not None and str(configured_jar).strip():
        jar = pathlib.Path(configured_jar).expanduser()
        if jar.is_file():
            return jar.resolve()
        raise ToolNotFoundError(
            f"Configured --msgf_jar {str(configured_jar)!r} does not exist."
        )

    env_jar = os.environ.get("UPROTEINS_MSGF_JAR", "").strip()
    if env_jar:
        jar = pathlib.Path(env_jar).expanduser()
        if jar.is_file():
            return jar.resolve()
        raise ToolNotFoundError(
            f"Environment variable UPROTEINS_MSGF_JAR={env_jar!r} does not exist."
        )

    # Check Bioconda msgf_plus wrapper installation layout:
    # <prefix>/bin/msgf_plus -> <prefix>/share/msgf_plus-*/MSGFPlus.jar
    msgf_wrapper = shutil.which("msgf_plus")
    if msgf_wrapper is not None:
        wrapper_path = pathlib.Path(msgf_wrapper)
        for prefix in {wrapper_path.parent.parent, wrapper_path.resolve().parent.parent}:
            matches = sorted(prefix.glob("share/msgf_plus*/MSGFPlus.jar"))
            if matches:
                return matches[-1].resolve()
        resolved_target = wrapper_path.resolve()
        if resolved_target.suffix.lower() == ".jar" and resolved_target.is_file():
            return resolved_target

    which_jar = shutil.which("MSGFPlus.jar")
    if which_jar is not None and pathlib.Path(which_jar).is_file():
        return pathlib.Path(which_jar).resolve()

    legacy_candidates = (
        REPO_ROOT / "dependencies" / "MSGFPlus" / "MSGFPlus.jar",
        REPO_ROOT / "dependencies" / "MSGF" / "MSGFPlus.jar",
    )
    for candidate in legacy_candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise ToolNotFoundError(
        "Could not locate MSGFPlus.jar. Install 'msgf_plus' via Bioconda, "
        "set $UPROTEINS_MSGF_JAR, pass --msgf_jar /path/to/MSGFPlus.jar, "
        "or run scripts/install_bio_tools.sh."
    )


def build_msgf_search_cmd(
    msgf_args: t.Sequence[str],
    *,
    msgf_jar: str | os.PathLike[str] | None = None,
    java_mem: str = "48G",
) -> list[str]:
    """Construct a safe argv list to invoke MS-GF+ peptide database search."""
    java_bin = resolve_executable(None, "java", env_var="UPROTEINS_JAVA")
    jar_path = resolve_msgf_jar(msgf_jar)
    return [str(java_bin), f"-Xmx{java_mem}", "-jar", str(jar_path), *map(str, msgf_args)]


def build_mzid_to_tsv_cmd(
    mzid_path: str | os.PathLike[str],
    tsv_path: str | os.PathLike[str],
    *,
    msgf_jar: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Construct a safe argv list for MS-GF+'s MzIDToTsv converter."""
    java_bin = resolve_executable(None, "java", env_var="UPROTEINS_JAVA")
    jar_path = resolve_msgf_jar(msgf_jar)
    return [
        str(java_bin),
        "-cp",
        str(jar_path),
        "edu.ucsd.msjava.ui.MzIDToTsv",
        "-i",
        str(mzid_path),
        "-o",
        str(tsv_path),
        "-showQValue",
        "1",
    ]


def resolve_free2bind_script(
    free2bind_dir: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Locate ``free_align.pl`` from free2bind across CLI, env var, PATH, or dependencies/."""
    if free2bind_dir is not None and str(free2bind_dir).strip():
        p = pathlib.Path(free2bind_dir).expanduser()
        script = p / "free_align.pl" if p.is_dir() else p
        if script.is_file():
            return script.resolve()
        raise ToolNotFoundError(
            f"Configured --free2bind_dir {str(free2bind_dir)!r} does not contain free_align.pl."
        )

    env_dir = os.environ.get("UPROTEINS_FREE2BIND_DIR", "").strip()
    if env_dir:
        p = pathlib.Path(env_dir).expanduser()
        script = p / "free_align.pl" if p.is_dir() else p
        if script.is_file():
            return script.resolve()
        raise ToolNotFoundError(
            f"Environment variable UPROTEINS_FREE2BIND_DIR={env_dir!r} "
            "does not contain free_align.pl."
        )

    which_script = shutil.which("free_align.pl")
    if which_script is not None and pathlib.Path(which_script).is_file():
        return pathlib.Path(which_script).resolve()

    legacy = REPO_ROOT / "dependencies" / "free2bind" / "free_align.pl"
    if legacy.is_file():
        return legacy.resolve()

    raise ToolNotFoundError(
        "Could not locate free2bind's free_align.pl. Pass --free2bind_dir, "
        "set $UPROTEINS_FREE2BIND_DIR, or run scripts/install_bio_tools.sh."
    )


def run_cmd(
    argv: t.Sequence[str | os.PathLike[str]],
    *,
    capture_output: bool = False,
    text: bool = False,
    input_data: bytes | str | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[t.Any]:
    """Execute ``argv`` strictly without a shell and raise on non-zero exit status."""
    if isinstance(argv, (str, bytes)):
        raise TypeError("run_cmd requires a sequence of arguments, never a shell string")
    cmd_list = [str(item) for item in argv]
    if not cmd_list:
        raise ValueError("run_cmd received an empty command sequence")

    result = subprocess.run(
        cmd_list,
        shell=False,
        check=False,
        capture_output=capture_output,
        text=text,
        input=input_data,
        cwd=str(cwd) if cwd is not None else None,
    )
    if result.returncode != 0:
        stderr_str: str | None = None
        if result.stderr is not None:
            stderr_str = (
                result.stderr
                if isinstance(result.stderr, str)
                else result.stderr.decode("utf-8", errors="replace")
            )
        raise SubprocessExecutionError(cmd_list, result.returncode, stderr_str)
    return result
