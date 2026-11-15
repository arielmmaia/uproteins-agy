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


@pytest.mark.database
def test_iupac_ambiguous_and_lowercase_sequences_translate_without_keyerror(tmp_path: pathlib.Path):
    """Regression test (Issue #13): lowercase soft-masked bases and IUPAC ambiguity symbols
    must complement and translate deterministically without KeyError."""
    from src.sequtils.conversion.complement import StrandConverter

    # Check full IUPAC complementation + lowercase handling
    sc = StrandConverter(["acgtRYKMSWBDHVN"])
    rev = sc.complement().reverse()[0]
    assert rev == "NBDHVWSKMRYACGT"

    # Degenerate codons:
    # atg -> M, ggn -> G, ctn -> L, gtn -> V, ath -> I, aar -> K, nnn -> X, gan -> X, taa -> stop
    fa = tmp_path / "iupac_softmasked.fasta"
    fa.write_text(">draft_contig_gap\nnnnnnatgggnctngtnathaarNNNgantaannnnn\n")

    gt = GenomeTranslator(
        sequence=str(fa),
        form="fasta",
        minsize=18,
        maxsize=300,
        circular=False,
    )
    orfs = list(gt.parse_frames(starts=["ATG"], stops=["TAA", "TAG", "TGA"], seqtype="aa"))
    fwd_seqs = [o.seq for o in orfs if o.strand == "forward"]
    assert "MGLVIKXX" in fwd_seqs


@pytest.mark.database
def test_sqlite_database_resets_stale_orfome_and_parameterizes_sql(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #13): re-running DatabaseGenerator in an existing directory
    must drop stale ORFOME rows instead of silently keeping old ORFs, and must safely
    parameterize names containing quotes/SQL metacharacters."""
    from src.sequtils.database import DatabaseGenerator
    from src.sequtils.orflib import ORF

    monkeypatch.chdir(tmp_path)

    # First run writes a stale ORF
    db1 = DatabaseGenerator(name="genome", db_type="sql")
    db1.add_orfs([
        ORF(name="ORF_STALE_1", seq="MSTALEORF", start=1, end=27, strand="forward")
    ])
    db1.to_fasta("genome_ORFs.fasta", identifier="g")
    assert "MSTALEORF" in (tmp_path / "genome_ORFs.fasta").read_text()

    # Second run in the same directory with a quote/SQLi payload in ORF name
    tricky_name = "ORF_O'Reilly'); DROP TABLE ORFOME;--"
    db2 = DatabaseGenerator(name="genome", db_type="sql")
    db2.add_orfs([
        ORF(name=tricky_name, seq="MFRESHORF", start=10, end=36, strand="reverse")
    ])
    db2.to_fasta("genome_ORFs.fasta", identifier="g")

    fasta_content = (tmp_path / "genome_ORFs.fasta").read_text()
    assert "MSTALEORF" not in fasta_content
    assert "MFRESHORF" in fasta_content
    rows = db2.retrieve()
    assert len(rows) == 1
    assert rows[0][1] == tricky_name
    assert rows[0][2] == "MFRESHORF"


@pytest.mark.database
def test_contig_and_isoform_ids_preserved_and_unstranded_transcripts_6frame(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #15):
    1. Multi-contig gORF headers preserve exact contig accessions (including underscores) and parse back cleanly.
    2. StringTie isoforms (uproteins.10.1 vs uproteins.10.2) stay distinct in TranscriptExtractor and StringTieGFF.
    3. Unstranded ('.') transcripts in assembled.gtf are translated in 6 frames, recovering minus-strand smORFs,
       whereas stranded ('+' / '-') transcripts are translated in 3 sense frames.
    """
    from Bio import SeqIO
    from src.sequtils.locus import StringTieGFF, parse_gorf_header, parse_torf_header
    from src.sequtils.transcriptomics.assembled import TranscriptExtractor

    monkeypatch.chdir(tmp_path)

    # 1. Multi-contig genome with underscore contig accession + plasmid
    genome_fa = tmp_path / "multi_contig.fasta"
    genome_fa.write_text(
        ">NC_000962.3\nATGGCTGCTGCTGCTGCTTAA\n"
        ">pPlasmid_A\nATGAAACCCTTTGGGCGCTAA\n"
    )
    proteome_fa = tmp_path / "proteome.fasta"
    proteome_fa.write_text(
        ">Rv0001 Rv0001 chromosomal replication initiator*\nMKLLEIR*\n"
        ">Rv0002_ANNO already tagged\nMPEPTIDE*\n"
    )

    # 2. Two distinct StringTie isoforms:
    # - uproteins.10.1 is '+' stranded (3 sense frames only)
    # - uproteins.10.2 is '.' unstranded and carries a minus-strand ORF (revcomp of ATG CCC GGG AAA TTT CGC TAA)
    from src.sequtils.conversion.complement import reverse_complement_dna
    minus_sense_cds = "ATGCCCGGGAAATTTCGCTAA"  # MP GKFR (18 nt + stop) -> "MPGKFR"
    unstranded_plus_rna = "GGG" + reverse_complement_dna(minus_sense_cds) + "CCC"

    ext_gtf = tmp_path / "ext_assembled.gtf"
    ext_gtf.write_text(
        "# StringTie version 2.2.1\n"
        'NC_000962.3\tStringTie\ttranscript\t10\t40\t1000\t+\t.\tgene_id "uproteins.10"; transcript_id "uproteins.10.1";\n'
        'NC_000962.3\tStringTie\texon\t10\t40\t1000\t+\t.\tgene_id "uproteins.10"; transcript_id "uproteins.10.1";\n'
        'pPlasmid_A\tStringTie\ttranscript\t50\t90\t1000\t.\t.\tgene_id "uproteins.10"; transcript_id "uproteins.10.2";\n'
        'pPlasmid_A\tStringTie\texon\t50\t90\t1000\t.\t.\tgene_id "uproteins.10"; transcript_id "uproteins.10.2";\n'
    )
    ext_tx = tmp_path / "ext_transcripts.fasta"
    ext_tx.write_text(
        f">uproteins.10.1\n{reverse_complement_dna(minus_sense_cds)}\n"
        f">uproteins.10.2\n{unstranded_plus_rna}\n"
    )

    out_dir = tmp_path / "db_out"
    # Note: omit -T YES to verify --external_gtf + --external_transcriptome auto-enables transcriptome DB!
    uproteins([
        "database",
        "-o", str(out_dir),
        "--genome", str(genome_fa),
        "--proteome", str(proteome_fa),
        "--external_gtf", str(ext_gtf),
        "--external_transcriptome", str(ext_tx),
        "--starts", "ATG",
        "--linear",
    ])

    # Check genome_database.fasta headers & _ANNO cleanup
    g_records = list(SeqIO.parse(out_dir / "genome_database.fasta", "fasta"))
    g_ids = [r.id for r in g_records]
    assert any(i.startswith("gORF_NC_000962.3_") for i in g_ids)
    assert any(i.startswith("gORF_pPlasmid_A_") for i in g_ids)

    for gid in g_ids:
        if gid.startswith("gORF_"):
            parsed = parse_gorf_header(gid)
            assert parsed is not None
            assert parsed[0] in ("NC_000962.3", "pPlasmid_A")

    # Verify _ANNO idempotency, description deduplication, and '*' stripping
    anno_map = {r.id: (r.description, str(r.seq)) for r in g_records if r.id.endswith("_ANNO")}
    assert "Rv0001_ANNO" in anno_map
    assert "Rv0002_ANNO" in anno_map
    assert "Rv0002_ANNO_ANNO" not in anno_map
    assert anno_map["Rv0001_ANNO"][1] == "MKLLEIR"
    assert anno_map["Rv0002_ANNO"][1] == "MPEPTIDE"
    assert not anno_map["Rv0001_ANNO"][0].startswith("Rv0001_ANNO Rv0001 ")

    # Check transcriptome_database.fasta was auto-generated and translated unstranded '.' isoform in 6 frames
    assert (out_dir / "transcriptome_database.fasta").is_file()
    t_records = [
        r for r in SeqIO.parse(out_dir / "transcriptome_database.fasta", "fasta")
        if r.id.startswith("tORF_")
    ]
    # Stranded '+' uproteins.10.1 had reverse-complemented ORF only on antisense -> must NOT emit MPGKFR
    tx1_seqs = [str(r.seq) for r in t_records if "uproteins.10.1" in r.id]
    assert "MPGKFR" not in tx1_seqs

    # Unstranded '.' uproteins.10.2 MUST emit MPGKFR on the reverse transcript frame with full isoform ID!
    tx2_hits = [r for r in t_records if "uproteins.10.2" in r.id and str(r.seq) == "MPGKFR"]
    assert len(tx2_hits) == 1
    parsed_tx = parse_torf_header(tx2_hits[0].id)
    assert parsed_tx is not None
    assert parsed_tx[0] == "uproteins.10.2"
    assert parsed_tx[3] == "reverse"

    # Both isoforms preserved in TranscriptExtractor and StringTieGFF (without mutating ext_assembled.gtf!)
    tx_dict = TranscriptExtractor(str(ext_tx)).get_transcripts()
    assert "uproteins.10.1" in tx_dict and "uproteins.10.2" in tx_dict
    st_dict = StringTieGFF(str(ext_gtf)).get_dict()
    assert "uproteins.10.1" in st_dict and "uproteins.10.2" in st_dict
    assert st_dict["uproteins.10.1"].start == 10
    assert st_dict["uproteins.10.2"].start == 50


@pytest.mark.database
def test_nested_non_atg_explosion_and_canonical_gtg_readthrough_protection(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #17):
    1. Redundant internal start codons (short <6 aa before K/R or internal ATG after K/R)
       do not duplicate nested sub-ORFs, while a downstream non-ATG start producing a
       distinct >=6 aa N-terminal formyl-Met peptide IS preserved.
    2. Internal non-ATG C-terminal sub-fragments of annotated _ANNO proteins are excluded.
    3. Upstream readthrough across a canonical GTG-initiated _ANNO protein:
       - Preceded immediately by K/R -> zero-upstream 'V...' canonical peptide is NOT emitted.
       - Preceded by non-K/R upstream residues -> trimmed right after first canonical K/R,
         retaining the genuine upstream-spanning N-terminal extension peptide.
    """
    from Bio import SeqIO

    monkeypatch.chdir(tmp_path)

    # Construct 4 loci on separate linear contigs:
    # Locus 1 (novel_collapsed):
    #   ATG AAA GTG CGC ATT AAA ATG GCT GCT GCT GCT GCT TAA
    #   Maximal ORF = M K V R I K M A A A A A (12 aa).
    #   Internal GTG (before R -> "MR", 2 aa), ATT (before K -> "MK", 2 aa), and internal ATG after K ("MAAAAA", already in max!)
    #   must ALL collapse into the single maximal ORF!
    locus1_dna = "ATGAAAGTGCGCATTAAAATGGCTGCTGCTGCTGCTTAA"

    # Locus 2 (novel_distinct_nterm):
    #   ATG AAA CGC GTG CCC GAA ACC ATC CGC GCT GCT TAA
    #   Maximal ORF = M K R V P E T I R A A (11 aa, tryptic peptide "VPETIR").
    #   Downstream GTG initiates M P E T I R A A (distinct 6-aa N-terminal Met peptide "MPETIR"!) -> both retained!
    locus2_dna = "ATGAAACGCGTGCCCGAAACCATCCGCGCTGCTTAA"

    # Locus 3 (cterm_of_long_anno):
    #   Annotated 120-aa protein RvLong = "M" + ("G" * 105) + "K" + "VAETVLRQ" (total 115 aa > 100 aa).
    #   On DNA, the C-terminal "VAETVLRQ" starts at an internal GTG ("GTGGCTGAAACCGTGCTGCGCCAATAA"):
    #   Without CanonicalProteomeMatcher, GenomeTranslator translates internal GTG as "MAETVLRQ" (8 aa <= 100 aa)!
    rv_long_prot = "M" + ("G" * 105) + "KVAETVLRQ"
    locus3_dna = "ATG" + ("GGT" * 105) + "AAAGTG" + "GCTGAAACCGTGCTGCGCCAATAA"

    # Locus 4a (false_readthrough_after_K):
    #   Canonical small GTG-start protein RvGTG1 = "MAETVLRKLLQ" (encoded on DNA with GTG start!).
    #   Upstream in-frame ATG CCC AAA (MPK) immediately precedes GTG:
    #   Full readthrough would be "MPKVAETVLRKLLQ" -> trypsin cuts after K, creating FAKE "VAETVLR" with 0 upstream aa!
    rv_gtg1_prot = "MAETVLRKLLQ"
    locus4a_dna = "ATGCCCAAA" + "GTGGCTGAAACCGTGCTGCGCAAACTGCTGCAATAA"

    # Locus 4b (true_spanning_extension):
    #   Canonical small GTG-start protein RvGTG2 = "MAERKLLQWY" (GTG start).
    #   Upstream in-frame ATG CCC GCT GCT (MPAA, no K/R before GTG!) reads through GTG:
    #   Full readthrough "MPAAVAERKLLQWY" -> trimmed right after first canonical R -> "MPAAVAER" (8 aa)!
    rv_gtg2_prot = "MAERKLLQWY"
    locus4b_dna = "ATGCCCGCTGCT" + "GTGGCTGAACGCAAACTGCTGCAATGGTACTAA"

    genome_fa = tmp_path / "h7_genome.fasta"
    genome_fa.write_text(
        f">c1\n{locus1_dna}\n"
        f">c2\n{locus2_dna}\n"
        f">c3\n{locus3_dna}\n"
        f">c4a\n{locus4a_dna}\n"
        f">c4b\n{locus4b_dna}\n"
    )
    proteome_fa = tmp_path / "h7_proteome.fasta"
    proteome_fa.write_text(
        f">RvLong\n{rv_long_prot}\n"
        f">RvGTG1\n{rv_gtg1_prot}\n"
        f">RvGTG2\n{rv_gtg2_prot}\n"
    )

    out_dir = tmp_path / "h7_out"
    uproteins([
        "database",
        "-o", str(out_dir),
        "--genome", str(genome_fa),
        "--proteome", str(proteome_fa),
        "--linear",
    ])

    g_orfs = [
        r for r in SeqIO.parse(out_dir / "genome_ORFs.fasta", "fasta")
        if "_forward" in r.id
    ]
    by_contig: dict[str, list[str]] = {}
    for rec in g_orfs:
        # gORF_{contig}_{num}_{start}-{end}_{strand}
        contig = rec.id[len("gORF_") :].rsplit("_", 3)[0]
        by_contig.setdefault(contig, []).append(str(rec.seq))

    # 1. Locus 1: all nested starts collapsed into the single maximal ORF
    assert by_contig.get("c1") == ["MKVRIKMAAAAA"]

    # 2. Locus 2: maximal ORF + distinct N-terminal Met peptide ("MPETIR") retained
    assert by_contig.get("c2") == ["MKRVPETIRAA", "MPETIRAA"]

    # 3. Locus 3: C-terminal internal-GTG fragment "MAETVLRQ" of RvLong completely excluded
    assert "MAETVLRQ" not in by_contig.get("c3", [])

    # 4a. Locus 4a: K-preceded readthrough across canonical GTG start completely excluded (MPK is 3 aa < 6 aa)
    assert by_contig.get("c4a", []) == []

    # 4b. Locus 4b: non-KR upstream extension trimmed right after first canonical R -> "MPAAVAER"
    assert by_contig.get("c4b") == ["MPAAVAER"]




