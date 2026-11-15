from argparse import Namespace
import pathlib
import stat
import sys
from types import SimpleNamespace

import pytest

from src import cli
from src.assembly import CompareTranscripts, ReadError, ReadMapper, TranscriptAssembly


def _make_exec(path: pathlib.Path, script: str) -> pathlib.Path:
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_hisat2_defaults_to_no_spliced_alignment_and_passes_threads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    genome = tmp_path / "genome.fasta"
    genome.write_text(">chr1\nACTG\n", encoding="utf-8")
    gtf = tmp_path / "ref.gtf"
    gtf.write_text("#gtf\n", encoding="utf-8")
    r1 = tmp_path / "r1.fq"
    r1.write_text("@r\nACTG\n+\nIIII\n", encoding="utf-8")

    stub = _make_exec(tmp_path / "stub.sh", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("UPROTEINS_HISAT2_BUILD", str(stub))
    monkeypatch.setenv("UPROTEINS_HISAT2", str(stub))
    monkeypatch.setenv("UPROTEINS_SAMTOOLS", str(stub))

    index_cmds: list[list[str]] = []
    monkeypatch.setattr(
        "src.assembly.subprocess.run",
        lambda cmd, *a, **kw: (index_cmds.append(list(cmd)), SimpleNamespace(returncode=0))[1],
    )

    hisat_cmds: list[list[str]] = []
    sort_cmds: list[list[str]] = []
    monkeypatch.setattr(
        ReadMapper,
        "_stream_align_and_sort",
        staticmethod(lambda ch, cs: (hisat_cmds.append(list(ch)), sort_cmds.append(list(cs)))),
    )

    # Default bacterial mode: --no-spliced-alignment present, -p 4 passed everywhere
    args_default = Namespace(
        gtf=gtf,
        genome=genome,
        threads=4,
        memory="2G",
        allow_splicing=False,
        single=[r1],
        reads1=None,
        reads2=None,
        strandness="R",
    )
    mapper = ReadMapper(args_default)
    mapper.create_indexes()
    mapper.align_reads()

    assert ["-p", "4"] == index_cmds[0][1:3]
    assert "--no-spliced-alignment" in hisat_cmds[0]
    p_idx = hisat_cmds[0].index("-p")
    assert hisat_cmds[0][p_idx + 1] == "4"
    assert ["-@","4"] == [sort_cmds[0][sort_cmds[0].index("-@")], sort_cmds[0][sort_cmds[0].index("-@") + 1]]

    # Opt-in --allow_splicing omits --no-spliced-alignment
    hisat_cmds.clear()
    args_spliced = Namespace(
        gtf=gtf,
        genome=genome,
        threads=2,
        memory=None,
        allow_splicing=True,
        single=[r1],
        reads1=None,
        reads2=None,
        strandness=None,
    )
    ReadMapper(args_spliced).align_reads()
    assert "--no-spliced-alignment" not in hisat_cmds[0]


def test_real_os_pipe_streaming_and_nonzero_exit_detection(tmp_path):
    out_bam = tmp_path / "sorted.bam"
    producer = [sys.executable, "-c", "import sys; sys.stdout.write('@HD\\tVN:1.6\\nread1\\n')"]
    consumer = [
        sys.executable,
        "-c",
        f"import sys, pathlib; pathlib.Path({str(out_bam)!r}).write_text(sys.stdin.read())",
    ]
    ReadMapper._stream_align_and_sort(producer, consumer)
    assert out_bam.read_text(encoding="utf-8") == "@HD\tVN:1.6\nread1\n"

    # Producer failure exits with code 3
    bad_producer = [sys.executable, "-c", "import sys; sys.exit(2)"]
    with pytest.raises(SystemExit) as exc:
        ReadMapper._stream_align_and_sort(bad_producer, consumer)
    assert exc.value.code == 3


def test_stringtie_filters_only_bam_and_writes_per_sample_abundance(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bam_dir = tmp_path / "HISAT" / "bam"
    bam_dir.mkdir(parents=True)
    (bam_dir / "aligned_0.bam").write_bytes(b"BAM0")
    (bam_dir / "aligned_0.bam.bai").write_bytes(b"BAI0")
    (bam_dir / "aligned_1.bam").write_bytes(b"BAM1")
    (bam_dir / ".DS_Store").write_bytes(b"junk")

    gtf = tmp_path / "ref.gtf"
    gtf.write_text("#gtf\n", encoding="utf-8")
    stub = _make_exec(tmp_path / "stringtie_stub", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("UPROTEINS_STRINGTIE", str(stub))

    cmds: list[list[str]] = []
    monkeypatch.setattr(
        "src.assembly.subprocess.run",
        lambda cmd, *a, **kw: (cmds.append(list(cmd)), SimpleNamespace(returncode=0))[1],
    )

    TranscriptAssembly(Namespace(threads=2, gtf=gtf, strandness="R")).assemble()
    assert len(cmds) == 2
    assert "HISAT/bam/aligned_0.bam" in cmds[0]
    assert "StringTie/gene_abundances_aligned_0.tab" in cmds[0]
    assert "HISAT/bam/aligned_1.bam" in cmds[1]
    assert "StringTie/gene_abundances_aligned_1.tab" in cmds[1]


def test_unequal_paired_end_reads_and_samefile_genome_copy(assembly_args, tmp_path, tmp_file, monkeypatch):
    # CLI validator rejects unequal -1 and -2 lists
    parser, subparsers = cli.get_parsers()
    parsed = parser.parse_args(
        [*assembly_args, "--reads1", f"{tmp_file},{tmp_file}", "--reads2", str(tmp_file)]
    )
    with pytest.raises(SystemExit) as exc:
        cli.validate_assembly(parsed, subparsers["assembly"])
    assert exc.value.code == 2

    # ReadMapper constructor also guards unequal paired-end lists directly
    with pytest.raises(ReadError):
        ReadMapper(
            Namespace(
                gtf=tmp_file,
                genome=tmp_file,
                threads=1,
                memory=None,
                single=None,
                reads1=[tmp_file, tmp_file],
                reads2=[tmp_file],
                strandness=None,
            )
        )

    # CompareTranscripts.copy_genome does not raise SameFileError when genome is already ./genome.fasta
    monkeypatch.chdir(tmp_path)
    in_place_genome = tmp_path / "genome.fasta"
    in_place_genome.write_text(">chr1\nACTG\n", encoding="utf-8")
    stub = _make_exec(tmp_path / "tool_stub", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("UPROTEINS_GFFCOMPARE", str(stub))
    monkeypatch.setenv("UPROTEINS_GFFREAD", str(stub))
    monkeypatch.setenv("UPROTEINS_SAMTOOLS", str(stub))

    ct = CompareTranscripts(
        Namespace(
            genome=in_place_genome,
            gtf=tmp_file,
            gffcompare_path="gffcompare",
            gffread_path="gffread",
        ),
        assembled_gtf="assembled.gtf",
    )
    assert in_place_genome.read_text(encoding="utf-8") == ">chr1\nACTG\n"
    assert ct is not None
