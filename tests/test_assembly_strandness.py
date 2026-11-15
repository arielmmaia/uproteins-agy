from argparse import Namespace
import os
import stat
from types import SimpleNamespace

import pytest

from src import cli
from src.assembly import (
    ReadMapper,
    TranscriptAssembly,
    hisat2_strand_args,
    stringtie_strand_args,
)


@pytest.mark.parametrize(
    "strand,expected_hisat,expected_stringtie",
    [
        (None, [], []),
        ("R", ["--rna-strandness", "R"], ["--rf"]),
        ("RF", ["--rna-strandness", "RF"], ["--rf"]),
        ("F", ["--rna-strandness", "F"], ["--fr"]),
        ("FR", ["--rna-strandness", "FR"], ["--fr"]),
    ],
)
def test_strand_arg_builders(strand, expected_hisat, expected_stringtie):
    assert hisat2_strand_args(strand) == expected_hisat
    assert stringtie_strand_args(strand) == expected_stringtie


@pytest.mark.parametrize(
    "strand,expected_flag",
    [
        (None, None),
        ("R", "--rf"),
        ("RF", "--rf"),
        ("F", "--fr"),
        ("FR", "--fr"),
    ],
)
def test_transcript_assembly_never_passes_empty_argv_element(
    tmp_path, monkeypatch, strand, expected_flag
):
    monkeypatch.chdir(tmp_path)
    bam_dir = tmp_path / "HISAT" / "bam"
    bam_dir.mkdir(parents=True)
    (bam_dir / "aligned_0.bam").write_bytes(b"BAM")
    gtf = tmp_path / "ref.gtf"
    gtf.write_text("#gtf\n", encoding="utf-8")

    fake_bin = tmp_path / "stringtie_stub"
    fake_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("UPROTEINS_STRINGTIE", str(fake_bin))

    recorded_cmds: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        recorded_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("src.assembly.subprocess.run", fake_run)

    args = Namespace(threads=2, gtf=gtf, strandness=strand)
    assembler = TranscriptAssembly(args)
    assembler.assemble()

    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    # Audit H1 regression check: no empty string may ever appear in argv!
    assert "" not in cmd
    if expected_flag is None:
        assert "--rf" not in cmd and "--fr" not in cmd
    else:
        assert expected_flag in cmd


def test_read_mapper_passes_single_end_r_and_unstranded_to_hisat2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    read_file = tmp_path / "reads.fastq"
    read_file.write_text("@r1\nACTG\n+\nIIII\n", encoding="utf-8")
    genome = tmp_path / "genome.fasta"
    genome.write_text(">chr1\nACTG\n", encoding="utf-8")
    gtf = tmp_path / "ref.gtf"
    gtf.write_text("#gtf\n", encoding="utf-8")

    fake_hisat = tmp_path / "hisat2_stub"
    fake_hisat.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_hisat.chmod(fake_hisat.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("UPROTEINS_HISAT2", str(fake_hisat))

    recorded_hisat: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        recorded_hisat.append(list(cmd))
        # Create dummy SAM so unlink doesn't fail if _sort_alignment ran
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("src.assembly.subprocess.run", fake_run)
    monkeypatch.setattr(ReadMapper, "_sort_alignment", lambda self, name: self)

    for strand in (None, "R", "F"):
        recorded_hisat.clear()
        args = Namespace(
            gtf=gtf,
            genome=genome,
            threads=1,
            memory=None,
            single=[read_file],
            reads1=None,
            reads2=None,
            strandness=strand,
        )
        ReadMapper(args).align_reads()
        assert len(recorded_hisat) == 1
        cmd = recorded_hisat[0]
        assert "" not in cmd
        if strand is None:
            assert "--rna-strandness" not in cmd
        else:
            idx = cmd.index("--rna-strandness")
            assert cmd[idx + 1] == strand


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--single", "FILE", "-L", "RF"],
        ["--single", "FILE", "-L", "FR"],
        ["--reads1", "FILE", "--reads2", "FILE", "-L", "R"],
        ["--reads1", "FILE", "--reads2", "FILE", "-L", "F"],
    ],
)
def test_validate_assembly_rejects_mismatched_read_layout_and_strandness(
    assembly_args, tmp_file, extra_args
):
    resolved = [str(tmp_file) if token == "FILE" else token for token in extra_args]
    parser, subparsers = cli.get_parsers()
    parsed = parser.parse_args([*assembly_args, *resolved])
    with pytest.raises(SystemExit) as exc:
        cli.validate_assembly(parsed, subparsers["assembly"])
    assert exc.value.code == 2
