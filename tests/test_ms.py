# Copyright © 2025 Eduardo Vieira de Souza
# Copyright © 2025 Adriana Canedo
# Copyright © 2025 Cristiano Valim Bizarro
# Copyright © 2025 Bruno Maestri A Becker
#
# This file is part of uProteInS.
#
# uProteInS is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# uProteInS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# uProteInS. If not, see <https://www.gnu.org/licenses/>.


from argparse import Namespace
import os
import pathlib

import pytest

from src import cli
import src.peptide_search as ps


def test_ms_parser_java_mem_default_and_override(tmp_path: pathlib.Path):
    ms_dir = tmp_path / "raw_ms"
    ms_dir.mkdir()
    parser, _ = cli.get_parsers()

    args_default = parser.parse_args([
        "ms", "-o", str(tmp_path / "out"), "-M", str(ms_dir)
    ])
    assert args_default.java_mem == "8G"

    args_custom = parser.parse_args([
        "ms", "-o", str(tmp_path / "out"), "-M", str(ms_dir), "--java_mem", "16G"
    ])
    assert args_custom.java_mem == "16G"


def test_peptide_search_explicit_o_flag_readonly_mass_spec_and_allowlist(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #21 / upstream Issue #5):
    1. Both target and decoy MS-GF+ runs pass explicit `-o {outdir}/{db_type}/{stem}_{target|decoy}.mzid`.
    2. `--Mass_spec` is never written to (works even when chmod 0555 read-only!).
    3. `.mzML`, `.mzml`, `.mzXML`, and `.mgf` are discovered case-insensitively, ignoring `.txt`/`.raw`.
    4. `MSGF_PARAM_MAP` allowlist forwards valid MS-GF+ flags (`-t`, `-inst`, `-e`, `-ntt`) and
       never leaks unrelated Namespace attributes (`unknown_future_flag`) into MS-GF+ argv.
    5. `--java_mem` controls `-Xmx` (`-Xmx8G` default, `-Xmx12G` when overridden).
    """
    raw_dir = tmp_path / "readonly_raw_spectra"
    raw_dir.mkdir()
    for fname in ("sampleA.mzML", "sampleB.mzml", "sampleC.mzXML", "sampleD.mgf", "notes.txt", "vendor.raw"):
        (raw_dir / fname).write_text("spectrum")

    out_dir = tmp_path / "workdir"
    out_dir.mkdir()
    monkeypatch.chdir(out_dir)

    target_fa = out_dir / "genome_database.fasta"
    target_fa.write_text(">gORF_chr1_1_1-24_forward\nMPEPTIDER\n")
    decoy_fa = out_dir / "Genome_decoy.fasta"
    decoy_fa.write_text(">decoy_gORF_chr1_1_1-24_forward\nMEDITPEPR\n")

    fake_java = tmp_path / "java"
    fake_java.write_text("#!/bin/sh\nexit 0\n")
    fake_java.chmod(0o755)
    fake_jar = tmp_path / "MSGFPlus.jar"
    fake_jar.write_text("jar")
    monkeypatch.setenv("UPROTEINS_JAVA", str(fake_java))

    # Make raw_dir strictly read-only to prove zero writes/copies occur inside --Mass_spec!
    raw_before = sorted(p.name for p in raw_dir.iterdir())
    raw_dir.chmod(0o555)

    recorded_cmds: list[list[str]] = []

    def fake_run_cmd(cmd, **kwargs):
        cmd_list = [str(x) for x in cmd]
        recorded_cmds.append(cmd_list)
        # Simulate MS-GF+ writing to the explicit `-o` path
        out_idx = cmd_list.index("-o") + 1
        pathlib.Path(cmd_list[out_idx]).write_text("<MzIdentML/>")

    monkeypatch.setattr(ps, "run_cmd", fake_run_cmd)

    try:
        args = Namespace(
            mode="ms",
            outdir=out_dir,
            mass_spec=raw_dir,
            transcriptome=False,
            msgf_jar=fake_jar,
            java_mem="12G",
            t="10ppm",
            inst="3",
            e="1",
            ntt="2",
            unknown_future_flag="MUST_NOT_LEAK",
        )

        # Target search
        target_search = ps.PeptideSearch("Genome", raw_dir, target_fa, args, decoy=False)
        target_search.peptide_identification()

        # Decoy search
        decoy_search = ps.PeptideSearch("Genome", raw_dir, decoy_fa, args, decoy=True)
        decoy_search.peptide_identification()
    finally:
        raw_dir.chmod(0o755)

    # Raw spectrum directory untouched!
    assert sorted(p.name for p in raw_dir.iterdir()) == raw_before

    # 4 spectrum files * 2 searches (target + decoy) = 8 commands
    assert len(recorded_cmds) == 8

    for cmd in recorded_cmds:
        assert "-Xmx12G" in cmd
        assert "-o" in cmd
        assert "-Unknown_future_flag" not in cmd
        assert "-unknown_future_flag" not in cmd
        assert "MUST_NOT_LEAK" not in cmd
        assert cmd[cmd.index("-t") + 1] == "10ppm"
        assert cmd[cmd.index("-inst") + 1] == "3"
        assert cmd[cmd.index("-e") + 1] == "1"
        assert cmd[cmd.index("-ntt") + 1] == "2"
        # Staged FASTA lives inside Genome/db_index/
        db_arg = pathlib.Path(cmd[cmd.index("-d") + 1])
        assert db_arg.parent == (out_dir / "Genome" / "db_index").resolve()

    # Verify both *_target.mzid and *_decoy.mzid exist directly inside out_dir/Genome/
    mzids = sorted(p.name for p in (out_dir / "Genome").glob("*.mzid"))
    assert mzids == [
        "sampleA_decoy.mzid",
        "sampleA_target.mzid",
        "sampleB_decoy.mzid",
        "sampleB_target.mzid",
        "sampleC_decoy.mzid",
        "sampleC_target.mzid",
        "sampleD_decoy.mzid",
        "sampleD_target.mzid",
    ]
