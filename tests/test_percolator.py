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
import pathlib

import pytest

import src.postprocess.percolator as perc_mod
from src.postprocess.percolator import (
    PercolatorProcessing,
    classify_mzid_file,
    compete_target_decoy_pin,
    normalize_pin_spec_id,
    resolve_enzyme_pair,
)
import src.postprocess.utils as post_utils


def test_classify_mzid_strict_suffix_and_xml_fallback(tmp_path: pathlib.Path):
    """Regression test (Issue #23 / H4):
    - Sample names containing 'decoy' (e.g. `decoy_ctrl_rep1_target.mzid`) MUST be
      classified as `'target'` via exact `_target.mzid` suffix, never `'decoy'`.
    - Legacy unnamed `.mzid` files fall back to inspecting `<DBSequence accession="...">`.
    """
    tricky_target = tmp_path / "decoy_ctrl_rep1_target.mzid"
    tricky_target.write_text("<MzIdentML/>")
    assert classify_mzid_file(tricky_target) == "target"

    real_decoy = tmp_path / "decoy_ctrl_rep1_decoy.mzid"
    real_decoy.write_text("<MzIdentML/>")
    assert classify_mzid_file(real_decoy) == "decoy"

    legacy_target = tmp_path / "non_decoy_study_run1.mzid"
    legacy_target.write_text(
        '<MzIdentML><SequenceCollection>\n'
        '  <DBSequence id="DBSeq1" accession="Rv0001_ANNO"/>\n'
        '</SequenceCollection></MzIdentML>\n'
    )
    assert classify_mzid_file(legacy_target) == "target"

    legacy_decoy = tmp_path / "run1_rev.mzid"
    legacy_decoy.write_text(
        '<MzIdentML><SequenceCollection>\n'
        '  <DBSequence id="DBSeq1" accession="decoy_Rv0001_ANNO"/>\n'
        '</SequenceCollection></MzIdentML>\n'
    )
    assert classify_mzid_file(legacy_decoy) == "decoy"

    legacy_concat = tmp_path / "run1_both.mzid"
    legacy_concat.write_text(
        '<MzIdentML><SequenceCollection>\n'
        '  <DBSequence id="DBSeq1" accession="Rv0001_ANNO"/>\n'
        '  <DBSequence id="DBSeq2" accession="decoy_Rv0001_ANNO"/>\n'
        '</SequenceCollection></MzIdentML>\n'
    )
    assert classify_mzid_file(legacy_concat) == "concatenated"


@pytest.mark.parametrize(
    "e_code,expected_pin,expected_perc",
    [
        ("0", "unspecific", "no_enzyme"),
        ("1", "trypsin", "trypsin"),
        ("2", "chymotrypsin", "chymotrypsin"),
        ("3", "lys-c", "lys-c"),
        ("4", "lys-n", "lys-n"),
        ("5", "glu-c", "glu-c"),
        ("6", "arg-c", "arg-c"),
        ("7", "asp-n", "asp-n"),
        ("8", "alpha_lp", "elastase"),
        ("9", "no_enzyme", "no_enzyme"),
    ],
)
def test_resolve_enzyme_pair_all_msgf_codes(
    e_code: str, expected_pin: str, expected_perc: str
):
    args = Namespace(e=e_code, enzyme=None)
    assert resolve_enzyme_pair(args) == (expected_pin, expected_perc)


def test_resolve_enzyme_pair_autodetects_mzid_xml_and_honors_override(
    tmp_path: pathlib.Path,
):
    lysc_mzid = tmp_path / "sampleA_target.mzid"
    lysc_mzid.write_text(
        '<MzIdentML><Enzymes><Enzyme id="Enz1"><EnzymeName>'
        '<cvParam accession="MS:1001309" name="Lys-C"/>'
        '</EnzymeName></Enzyme></Enzymes></MzIdentML>\n'
    )
    assert resolve_enzyme_pair(Namespace(e=None, enzyme=None), lysc_mzid) == (
        "lys-c",
        "lys-c",
    )
    # Explicit --enzyme override is honored on percolator
    assert resolve_enzyme_pair(Namespace(e="1", enzyme="trypsinp"), lysc_mzid) == (
        "trypsin",
        "trypsinp",
    )


def test_normalize_pin_spec_id_unifies_target_and_decoy_stems():
    norm_t, stem_t = normalize_pin_spec_id("sample_A_target_SII_1_1_250_2_1")
    norm_d, stem_d = normalize_pin_spec_id("sample_A_decoy_SII_1_1_250_2_1")
    assert norm_t == "sample_A_SII_1_1_250_2_1"
    assert norm_d == "sample_A_SII_1_1_250_2_1"
    assert stem_t == "sample_A"
    assert stem_d == "sample_A"


def test_compete_target_decoy_pin_retains_top1_per_scan_and_unbiased_ties(
    tmp_path: pathlib.Path,
):
    """Verify per-scan top-1 Target-Decoy Competition:
    - Scan 100: target lnSpecEValue (28.5) > decoy lnSpecEValue (12.1) -> target wins.
    - Scan 200: decoy lnSpecEValue (24.0) > target lnSpecEValue (15.2) -> decoy wins.
    - Scans 300..399: exact ties between target and decoy -> deterministic ~50/50 split
      without systematic bias toward target or decoy.
    """
    pin_file = tmp_path / "Genome_pin.txt"
    lines = [
        "SpecId\tLabel\tScanNr\tExpMass\tlnSpecEValue\tPeptide\tProteins\n",
        "DefaultDirection\t-\t-\t-\t1\t-\t-\n",
        # Scan 100: target beats decoy
        "run1_target_SII_1_1_100_2_1\t1\t100\t1000.5\t28.5\tR.PEPTIDER.A\tgORF_chr1_1_1-30_forward\n",
        "run1_decoy_SII_1_1_100_2_1\t-1\t100\t1000.5\t12.1\tR.REDITPEPR.A\tdecoy_gORF_chr1_1_1-30_forward\n",
        # Scan 200: decoy beats target
        "run1_target_SII_2_1_200_2_1\t1\t200\t1100.5\t15.2\tK.WEAKPEPK.A\tgORF_chr1_2_40-70_forward\n",
        "run1_decoy_SII_2_1_200_2_1\t-1\t200\t1100.5\t24.0\tK.KPEPKAEWK.A\tdecoy_gORF_chr1_2_40-70_forward\n",
    ]
    for sc in range(300, 400):
        lines.append(
            f"run1_target_SII_{sc}_1_{sc}_2_1\t1\t{sc}\t900.0\t18.0\tK.TIEPEPK.A\tgORF_chr1_3_80-110_forward\n"
        )
        lines.append(
            f"run1_decoy_SII_{sc}_1_{sc}_2_1\t-1\t{sc}\t900.0\t18.0\tK.KPEPEITK.A\tdecoy_gORF_chr1_3_80-110_forward\n"
        )
    pin_file.write_text("".join(lines))

    compete_target_decoy_pin(pin_file, seed=42)

    backup = tmp_path / "Genome_pin_uncompeted.txt"
    assert backup.is_file()
    assert len(backup.read_text().splitlines()) == 2 + 204

    competed_lines = pin_file.read_text().splitlines()
    assert competed_lines[0].startswith("SpecId\tLabel\tScanNr")
    assert competed_lines[1].startswith("DefaultDirection\t")
    data_rows = [row.split("\t") for row in competed_lines[2:]]
    # 102 unique scans -> strictly 102 rows!
    assert len(data_rows) == 102

    by_scan = {int(r[2]): r for r in data_rows}
    # SpecIds normalized (`_target` / `_decoy` stripped)
    assert by_scan[100][0] == "run1_SII_1_1_100_2_1"
    assert by_scan[100][1] == "1"
    assert by_scan[200][0] == "run1_SII_2_1_200_2_1"
    assert by_scan[200][1] == "-1"

    tie_targets = sum(1 for sc in range(300, 400) if by_scan[sc][1] == "1")
    assert 30 <= tie_targets <= 70


def test_percolator_processing_end_to_end_commands_and_tsv_skip_decoy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    genome_dir = tmp_path / "Genome"
    genome_dir.mkdir()

    # Include a `.mzid.tsv` file from an earlier run + a sample with `decoy` in its name!
    (genome_dir / "decoy_study_A_target.mzid").write_text(
        '<MzIdentML><Enzymes><Enzyme><EnzymeName>'
        '<cvParam name="Chymotrypsin"/>'
        '</EnzymeName></Enzyme></Enzymes></MzIdentML>\n'
    )
    (genome_dir / "decoy_study_A_decoy.mzid").write_text("<MzIdentML/>")
    (genome_dir / "stale_previous.mzid.tsv").write_text("SpecFile\tScanNum\n")

    fake_pin = tmp_path / "msgf2pin"
    fake_pin.write_text("#!/bin/sh\nexit 0\n")
    fake_pin.chmod(0o755)
    fake_perc = tmp_path / "percolator"
    fake_perc.write_text("#!/bin/sh\nexit 0\n")
    fake_perc.chmod(0o755)
    fake_java = tmp_path / "java"
    fake_java.write_text("#!/bin/sh\nexit 0\n")
    fake_java.chmod(0o755)
    fake_jar = tmp_path / "MSGFPlus.jar"
    fake_jar.write_text("jar")
    monkeypatch.setenv("UPROTEINS_JAVA", str(fake_java))

    recorded: list[list[str]] = []

    def fake_run_cmd(cmd, **kwargs):
        cmd_list = [str(x) for x in cmd]
        recorded.append(cmd_list)
        if "-o" in cmd_list and cmd_list[0].endswith("msgf2pin"):
            out_pin = pathlib.Path(cmd_list[cmd_list.index("-o") + 1])
            out_pin.write_text(
                "SpecId\tLabel\tScanNr\tlnSpecEValue\tPeptide\tProteins\n"
                "decoy_study_A_target_SII_1_1_10_2_1\t1\t10\t22.0\tF.PEPF.A\tgORF_c1_1_1-21_forward\n"
                "decoy_study_A_decoy_SII_1_1_10_2_1\t-1\t10\t9.0\tF.FPEPF.A\tdecoy_gORF_c1_1_1-21_forward\n"
            )

    monkeypatch.setattr(perc_mod, "run_cmd", fake_run_cmd)
    monkeypatch.setattr(post_utils, "run_cmd", fake_run_cmd)

    args = Namespace(
        msgf2pin_path=fake_pin,
        percolator_path=fake_perc,
        msgf_jar=fake_jar,
        java_mem="8G",
        e=None,
        enzyme=None,
    )

    proc = PercolatorProcessing(folder="Genome", filetype="genome", args=args)
    proc.create_metafiles().convert_to_pin().percolate()

    target_meta_lines = pathlib.Path(proc.target).read_text().splitlines()
    decoy_meta_lines = pathlib.Path(proc.decoy).read_text().splitlines()
    assert len(target_meta_lines) == 1
    assert target_meta_lines[0].endswith("decoy_study_A_target.mzid")
    assert len(decoy_meta_lines) == 1
    assert decoy_meta_lines[0].endswith("decoy_study_A_decoy.mzid")

    # Auto-detected Chymotrypsin passed to BOTH msgf2pin and percolator!
    pin_cmd = recorded[0]
    perc_cmd = recorded[1]
    assert pin_cmd[pin_cmd.index("-e") + 1] == "chymotrypsin"
    assert perc_cmd[perc_cmd.index("--protein-enzyme") + 1] == "chymotrypsin"
    assert "--search-input" in perc_cmd
    assert perc_cmd[perc_cmd.index("--search-input") + 1] == "concatenated"
    assert "-Y" in perc_cmd

    # TSVConverter skips `_decoy.mzid` and `.mzid.tsv`, converting only `decoy_study_A_target.mzid`!
    tsv_conv = post_utils.TSVConverter("Genome", args=args)
    tsv_conv.convert_files()
    assert len(recorded) == 3
    tsv_cmd = recorded[2]
    assert tsv_cmd[tsv_cmd.index("-i") + 1].endswith("decoy_study_A_target.mzid")
