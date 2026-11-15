from argparse import Namespace
import pathlib
import pytest

from src.sequtils.orflib import ORF, ORFCollection
from src.upstream import (
    SDInspection,
    classify_rbs_category,
    compute_rbs_free_energy,
    extract_16s_anti_sd_tail,
    extract_genomic_upstream_21nt,
    extract_transcript_upstream,
)


def test_extract_genomic_upstream_exact_21nt_both_strands_and_circular_wrap():
    # Build a 60-nt contig:
    # Positions 1..21 (0-based 0..20):   "CCCCCCCCCCCCCCCAGGAGG" (len 21)
    # Positions 22..24 (0-based 21..23): "ATG" (start codon at 1-based 22!)
    # Positions 25..36 (0-based 24..35): "AAACTGGAATGA"
    # Notice if off-by-one [start-22 : start] were used, 'A' of ATG would be included!
    contig = "CCCCCCCCCCCCCCCAGGAGG" + "ATG" + "AAACTGGAATGA" + "TTTTTTTTTTTTTTTTTTTTTTTT"
    assert len(contig) == 60

    up_plus = extract_genomic_upstream_21nt(contig, 22, 36, "forward", circular=False)
    assert len(up_plus) == 21
    assert up_plus == "CCCCCCCCCCCCCCCAGGAGG"
    assert not up_plus.endswith("A")  # Base 1 of ATG MUST NOT leak into upstream window!

    # Minus-strand ORF occupying 1-based [10..24] (start codon at 24..22 on minus strand).
    # Strictly upstream 21 nt on minus strand are genomic 25..45 reverse-complemented!
    # Let's place reverse-complement of "TTTTTAGGAGGTTTTTTTTTT" (len 21) at genomic 25..45 (0-based 24..44):
    # revcomp("TTTTTAGGAGGTTTTTTTTTT") = "AAAAAAAAAACCTCCTAAAAA"
    minus_contig = (
        "GGGGGGGGGG"  # 1..10
        + "TCATTCCAGTTT"  # 11..22
        + "CA"  # 23..24 -> let's make 10..24 exact 15 nt minus ORF: 22..24 = CAT (revcomp ATG!)
    )
    # Construct exact minus contig:
    # 1..9 (9 nt):   "GGGGGGGGG"
    # 10..24 (15 nt): revcomp("ATGAAACTGGAATGA") = "TCATTCCAGTTTCAT" (start codon at 22..24 = "CAT" -> minus ATG at hi=24)
    # 25..45 (21 nt): revcomp("TTTTTAGGAGGTTTTTTTTTT") = "AAAAAAAAAACCTCCTAAAAA"
    # 46..60 (15 nt): "CCCCCCCCCCCCCCC"
    exact_minus_contig = (
        "GGGGGGGGG"
        + "TCATTCCAGTTTCAT"
        + "AAAAAAAAAACCTCCTAAAAA"
        + "CCCCCCCCCCCCCCC"
    )
    assert len(exact_minus_contig) == 60

    # Works identically whether caller passes (24, 10) or (10, 24)
    up_minus_a = extract_genomic_upstream_21nt(
        exact_minus_contig, 24, 10, "reverse", circular=False
    )
    up_minus_b = extract_genomic_upstream_21nt(
        exact_minus_contig, 10, 24, "-", circular=False
    )
    assert len(up_minus_a) == 21
    assert up_minus_a == "TTTTTAGGAGGTTTTTTTTTT"
    assert up_minus_b == "TTTTTAGGAGGTTTTTTTTTT"

    # Circular oriC wrap-around when + ORF starts at position 6 (only 5 nt at 1..5, wraps 16 nt from 3' end)
    circ_contig = "TGATCATGAAACTGGAATGA" + "CCCCCCCCCCCCCCCCCCCC" + "NNNNNTTTTTAGGAGG"
    # len = 20 + 20 + 16 = 56; start codon ATG at 1-based 6..8 -> 1..5 is "TGATC", last 16 nt is "NNNNNTTTTTAGGAGG"
    up_circ = extract_genomic_upstream_21nt(circ_contig, 6, 20, "forward", circular=True)
    assert len(up_circ) == 21
    assert up_circ == "NNNNNTTTTTAGGAGGTGATC"


def test_turner_nearest_neighbor_duplex_dg_matches_canonical_motifs():
    # Bacterial 16S 3' tail: 5'-GAUCACCUCCUUA-3' -> 3'->5' is "AUUCCUCCACUAG"
    anti_sd_3to5 = "AUUCCUCCACUAG"

    # 1. Extended 7-bp Shine-Dalgarno 5'-AGGAGGU-3' -> -10.28 kcal/mol (Strong <= -8.4)
    dg_7bp = compute_rbs_free_energy("UUUUUAGGAGGUUUUUU", anti_sd_3to5)
    assert dg_7bp == pytest.approx(-10.28, abs=0.05)

    # 2. Core hexamer 5'-AGGAGG-3' -> -8.49 kcal/mol (Strong <= -8.4)
    dg_6bp = compute_rbs_free_energy("UUUUUAGGAGGUUUUU"[:-5] + "CCCCC", anti_sd_3to5)
    assert dg_6bp == pytest.approx(-8.49, abs=0.05)
    assert classify_rbs_category(dg_6bp) == "Present. Strong binding."

    # 3. Moderate tetramer 5'-GGAG-3' -> -3.60 kcal/mol (Moderate <= -3.4535)
    dg_4bp = compute_rbs_free_energy("UUUUUGGAGUUUUU", anti_sd_3to5)
    assert dg_4bp == pytest.approx(-3.60, abs=0.05)
    assert classify_rbs_category(dg_4bp) == "Present. Low to moderate binding."

    # 4. Non-SD poly-A upstream -> 0.0 kcal/mol -> "None (SD absent)" in Genome mode!
    dg_none = compute_rbs_free_energy("AAAAAAAAAAAAAAAAAAAAA", anti_sd_3to5)
    assert dg_none == 0.0
    assert classify_rbs_category(dg_none, is_transcriptome=False) == "None (SD absent)"


def test_short_5utr_preserved_and_leaderless_vs_sd_absent_distinguished(
    tmp_path: pathlib.Path,
):
    genome_fa = tmp_path / "genome.fasta"
    genome_fa.write_text(">chr1\n" + ("A" * 200) + "\n")

    # Multi-record rRNA FASTA where record 0 is 5S rRNA and record 1 is 16S rRNA!
    rrna_fa = tmp_path / "rna_from_genomic.fna"
    rrna_fa.write_text(
        ">lcl|rrfA 5S ribosomal RNA\n"
        "UGCCUGGCGGCCGUAGCGCGGUGGUCCCACCUGACCCCAUGCCGAACUCAGAAGUGAAAC\n"
        ">lcl|rrsA 16S ribosomal RNA\n"
        "AAACUGAAGAGUUUGAUCAUGGCUCAGAUUGAACGCUGGCGGCAGGCCUAACACAUGCAAGUCGAACGGUAACAGGAUCACCUCCUUA\n"
    )
    tail = extract_16s_anti_sd_tail(rrna_fa)
    assert tail == "AUUCCUCCACUAG"

    gtf_path = tmp_path / "assembled.gtf"
    gtf_path.write_text(
        'chr1\tStringTie\ttranscript\t1\t80\t1000\t+\t.\tgene_id "g1"; transcript_id "uproteins.1.1";\n'
        'chr1\tStringTie\ttranscript\t81\t160\t1000\t+\t.\tgene_id "g2"; transcript_id "uproteins.2.1";\n'
        'chr1\tStringTie\ttranscript\t161\t200\t1000\t+\t.\tgene_id "g3"; transcript_id "uproteins.3.1";\n'
    )
    tx_fa = tmp_path / "transcripts.fasta"
    # Transcript 1: 12-nt 5' UTR ("AGGAGGTGATCA") before ATG at local nt 13..36 -> Strong SD!
    tx1 = "AGGAGGTGATCA" + "ATGAAACTGGAACGCATCAAATGA"
    # Transcript 2: 2-nt 5' UTR ("AC") before ATG at local nt 3..26 -> True Leaderless (< 5 nt)!
    tx2 = "AC" + "ATGAAACTGGAACGCATCAAATGA"
    # Transcript 3: 25-nt poly-A 5' UTR before ATG at local nt 26..49 -> "None (SD absent)" (NOT Leaderless!)
    tx3 = ("A" * 25) + "ATGAAACTGGAACGCATCAAATGA"
    tx_fa.write_text(
        f">uproteins.1.1\n{tx1}\n>uproteins.2.1\n{tx2}\n>uproteins.3.1\n{tx3}\n"
    )

    c1 = ORFCollection()
    o1 = ORF(name="tORF_uproteins.1.1_1_13-36_forward", start=13, end=36, strand="forward", transcript=tx1)
    o1.transcriptName = "uproteins.1.1"
    c1.add_orf(o1)

    c2 = ORFCollection()
    o2 = ORF(name="tORF_uproteins.2.1_1_3-26_forward", start=3, end=26, strand="forward", transcript=tx2)
    o2.transcriptName = "uproteins.2.1"
    c2.add_orf(o2)

    c3 = ORFCollection()
    o3 = ORF(name="tORF_uproteins.3.1_1_26-49_forward", start=26, end=49, strand="forward", transcript=tx3)
    o3.transcriptName = "uproteins.3.1"
    c3.add_orf(o3)

    alts = {
        "uproteins.1.1_36": c1,
        "uproteins.2.1_26": c2,
        "uproteins.3.1_49": c3,
    }
    args = Namespace(genome=genome_fa, rrna=rrna_fa, rbs_backend="python", circular=True)
    sd = SDInspection(
        args,
        filetype="transcriptome",
        folder="Transcriptome",
        alternatives=alts,
        subset="Transcriptome",
        transcriptome_gff=gtf_path,
        transcripts=tx_fa,
    )
    scored = sd.get_free_energy()

    r1 = list(scored["uproteins.1.1_36"])[0]
    assert r1.upstream == "AGGAGGTGATCA"
    assert r1.freeEnergy <= -8.4
    assert r1.shineDalgarno == "Present. Strong binding."

    r2 = list(scored["uproteins.2.1_26"])[0]
    assert r2.freeEnergy == 0.0
    assert r2.shineDalgarno == "Leaderless"

    r3 = list(scored["uproteins.3.1_49"])[0]
    assert r3.freeEnergy == 0.0
    assert r3.shineDalgarno == "None (SD absent)"


def test_sd_inspection_handles_rrna_none_and_warns_missing_ccucc(tmp_path: pathlib.Path):
    genome_fa = tmp_path / "genome.fasta"
    genome_fa.write_text(">chr1\n" + ("C" * 80) + "\n")

    bad_rrna = tmp_path / "no_antisd.fna"
    bad_rrna.write_text(">custom_rna\nAAAAAAAAAAAAAAAAAAAA\n")
    with pytest.warns(UserWarning, match="CCUCC"):
        extract_16s_anti_sd_tail(bad_rrna)

    col = ORFCollection()
    col.add_orf(ORF(name="gORF_chr1_1_30-60_forward", start=30, end=60, strand="forward", chromosome="chr1"))
    args_no_rrna = Namespace(genome=genome_fa, rrna=None, rbs_backend="python", circular=True)
    sd = SDInspection(
        args_no_rrna,
        filetype="genome",
        folder="Genome",
        alternatives={"60": col},
        subset="Genome",
    )
    scored = sd.get_free_energy()
    orf_out = list(scored["60"])[0]
    assert orf_out.freeEnergy == 0.0
    assert orf_out.shineDalgarno == "Not inspected (--rrna omitted)"
