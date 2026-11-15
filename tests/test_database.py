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


import pathlib
import pytest

from src import uproteins, cli, assembly  # noqa: F401
from src.sequtils.conversion.translate import Translator
from src.sequtils.frame_translation import GenomeTranslator, TranscriptomeTranslator


@pytest.mark.database
def test_database_parser_defaults_and_flags(database_args, tmp_file):
    database_args += ['--genome', str(tmp_file), '--proteome', str(tmp_file)]
    parser, subparsers = cli.get_parsers()
    args = parser.parse_args(database_args)
    cli.validate_database(args, subparsers['database'])

    assert args.maxsize == 300
    assert args.minsize == 18
    assert args.circular is True
    assert args.genetic_code == 11

    linear_args = parser.parse_args(database_args + ['--linear', '--genetic_code', '4'])
    cli.validate_database(linear_args, subparsers['database'])
    assert linear_args.circular is False
    assert linear_args.genetic_code == 4


@pytest.mark.database
def test_database_validator_rejects_minsize_gt_maxsize(database_args, tmp_file):
    parser, subparsers = cli.get_parsers()
    args = parser.parse_args(
        database_args + ['--minsize', '120', '--maxsize', '60']
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.validate_database(args, subparsers['database'])
    assert excinfo.value.code == 2


@pytest.mark.database
def test_immediate_stop_codon_never_skips_to_downstream_stop(tmp_path: pathlib.Path):
    """Regression test: ATG immediately followed by a custom or canonical stop at codon #2
    must terminate immediately instead of stepping over codon #2 via (...)+?."""
    fa = tmp_path / "immediate_stop.fasta"
    # Custom stop 'CTC' at codon #2 followed by 6 sense codons and another 'CTC':
    # Old regex `(%s)(...)+?(%s)` stepped over the first CTC and fused `ATG CTC AAA AAA AAA AAA AAA AAA`!
    fa.write_text(">contig1\nATGCTCAAAAAAAAAAAAAAAAAACTC\n")

    translator = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=3,
        maxsize=300,
        circular=False,
    )
    orfs = list(translator.parse_frames(starts=["ATG"], stops=["CTC"], seqtype="aa"))
    forward_seqs = [o.seq for o in orfs if o.strand == "forward"]
    assert forward_seqs == ["M"]


@pytest.mark.database
def test_mycoplasma_custom_stops_retains_internal_tga_as_trp(tmp_path: pathlib.Path):
    """Regression test: when TGA is excluded from --stops (Mycoplasma/Spiroplasma),
    internal TGA must translate as W (Tryptophan) rather than '*' and must NOT be discarded."""
    fa = tmp_path / "mycoplasma.fasta"
    # ATG (M) AAA (K) TGA (W) CCC (P) GGG (G) CGC (R) TAA (stop) -> 18 nt = 6 aa "MKWPGR"
    fa.write_text(">myco_chr\nATGAAATGACCCGGGCGCTAA\n")

    tr = Translator("ATGAAATGACCCGGGCGC", stops=["TAA", "TAG"])
    assert tr.translate() == "MKWPGR"

    gt = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=False,
    )
    orfs = list(gt.parse_frames(starts=["ATG"], stops=["TAA", "TAG"], seqtype="aa"))
    assert any(o.seq == "MKWPGR" and o.strand == "forward" for o in orfs)

    # Also verify explicit --genetic_code 4
    gt_code4 = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=False,
        genetic_code=4,
    )
    orfs_code4 = list(
        gt_code4.parse_frames(starts=["ATG"], stops=["TAA", "TAG"], seqtype="aa")
    )
    assert any(o.seq == "MKWPGR" for o in orfs_code4)


@pytest.mark.database
def test_circular_oric_wrapping_forward_and_reverse_without_duplicates(tmp_path: pathlib.Path):
    """Circular bacterial replicons must capture ORFs straddling oriC on both strands
    without duplicating non-wrapping internal ORFs, whereas --linear omits oriC-spanning ORFs."""
    fa = tmp_path / "circular_chr.fasta"
    # Construct a 60-nt circular replicon:
    # - Internal forward ORF at [13..30] (0-based 12..30): ATG GCT GCT GCT GCT GCT TAA ("MAAAAA", 18 nt)
    # - oriC-wrapping forward ORF starting at 0-based 48 (1-based 49):
    #   tail [48..60] = "ATGCCCCCCTTT" (12 nt = M P P F)
    #   head [0..9]   = "AAACGCTAA"   (6 nt = K R + stop TAA) -> "MPPFKR" (18 nt, 1-based 49..6)
    head = "AAACGCTAA"       # 9 nt (0..9)
    spacer1 = "GGG"          # 3 nt (9..12)
    internal = "ATGGCTGCTGCTGCTGCTTAA"  # 21 nt (12..33) -> MAAAAA (18 nt, 1-based 13..30)
    spacer2 = "CCCCCCCCCCCCCCC"         # 15 nt (33..48)
    tail = "ATGCCCCCCTTT"    # 12 nt (48..60)
    seq = head + spacer1 + internal + spacer2 + tail
    assert len(seq) == 60
    fa.write_text(f">chr_circular\n{seq}\n")

    circ = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=True,
    )
    circ_orfs = list(circ.parse_frames(starts=["ATG"], stops=["TAA", "TAG", "TGA"], seqtype="aa"))
    fwd_circ = [(o.seq, o.start, o.end) for o in circ_orfs if o.strand == "forward"]

    # Internal ORF appears exactly once, and oriC-spanning ORF is recovered with wrapped coords 49..6
    assert fwd_circ.count(("MAAAAA", 13, 30)) == 1
    assert ("MPPFKR", 49, 6) in fwd_circ

    # Linear mode recovers internal ORF but NOT oriC-spanning ORF
    lin = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=False,
    )
    lin_orfs = list(lin.parse_frames(starts=["ATG"], stops=["TAA", "TAG", "TGA"], seqtype="aa"))
    fwd_lin = [(o.seq, o.start, o.end) for o in lin_orfs if o.strand == "forward"]
    assert ("MAAAAA", 13, 30) in fwd_lin
    assert ("MPPFKR", 49, 6) not in fwd_lin

    # Also verify a reverse-strand oriC-spanning ORF:
    # Place reverse-strand ORF 5'->3' "ATGCCCCCCTTTAAACGCTAA" across oriC by reverse-complementing the circular genome
    from src.sequtils.conversion import StrandConverter
    rev_seq = StrandConverter([seq]).complement().reverse()[0]
    fa_rev = tmp_path / "circular_chr_rev.fasta"
    fa_rev.write_text(f">chr_circular_rev\n{rev_seq}\n")

    circ_rev = GenomeTranslator(
        sequence=str(fa_rev),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=True,
    )
    rev_orfs = [
        o for o in circ_rev.parse_frames(starts=["ATG"], stops=["TAA", "TAG", "TGA"], seqtype="aa")
        if o.strand == "reverse"
    ]
    assert any(o.seq == "MPPFKR" for o in rev_orfs)

    # TranscriptomeTranslator is always linear by default
    tx = TranscriptomeTranslator(sequence=str(fa), form="fasta", minsize=18, maxsize=300)
    tx_orfs = list(tx.parse_frames(starts=["ATG"], stops=["TAA", "TAG", "TGA"], seqtype="aa"))
    assert all(o.seq != "MPPFKR" for o in tx_orfs)

