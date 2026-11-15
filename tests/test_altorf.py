from argparse import Namespace
import pathlib
import pandas as pd
import pytest

from src.postprocess.extinfo import ExtendedInformation
from src.sequtils.orflib import AltCodons
from src.upstream import SDInspection


_AA_TO_DNA = {
    "A": "GCG",
    "C": "TGC",
    "D": "GAC",
    "E": "GAA",
    "F": "TTC",
    "G": "GGC",
    "H": "CAC",
    "I": "ATC",
    "K": "AAG",
    "L": "CTG",
    "M": "ATG",
    "N": "AAC",
    "P": "CCG",
    "Q": "CAG",
    "R": "CGC",
    "S": "AGC",
    "T": "ACC",
    "V": "GTG",
    "W": "TGG",
    "Y": "TAC",
}


def _encode_aa(aa_seq: str) -> str:
    return "".join(_AA_TO_DNA[ch] for ch in aa_seq)


def _revcomp(dna: str) -> str:
    trans = str.maketrans("ACGT", "TGCA")
    return dna.translate(trans)[::-1]


def test_minus_strand_terminates_without_stop_and_plus_reaches_position_1(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    # 1. Plus-strand contig starting AT position 1..3 with ATG, followed by GTG at 10..12, peptide, TGA stop
    #    1..3 = ATG, 4..9 = AAACTG, 10..12 = GTG, 13..36 = encode("PEPTIDER"), 37..39 = TGA
    plus_orf_dna = "ATGAAACTGGTG" + _encode_aa("PEPTIDER") + "TGA"
    assert len(plus_orf_dna) == 39

    # 2. Minus-strand contig with NO upstream stop codon all the way to len(contig)!
    #    Minus sense 5'->3': ATG (at contig 3' end!) + AAACTG + GTG + PEPTIDER + TGA
    minus_sense = "ATGAAACTGGTG" + _encode_aa("PEPTIDER") + "TGA"
    minus_contig = "CCCC" + _revcomp(minus_sense)  # minus ORF spans 5..43; orig GTG is at hi=34..lo=5; ATG is at hi=43 (contig end!)

    genome_fa = tmp_path / "genome.fasta"
    genome_fa.write_text(f">chr_plus\n{plus_orf_dna}CCCCCCCCCC\n>chr_minus\n{minus_contig}\n")

    res02 = tmp_path / "genome_results_02.txt"
    pd.DataFrame(
        {
            "SpecFile": ["r1.mzML", "r1.mzML"],
            "SpecID": ["index=1", "index=2"],
            "ScanNum": [10, 11],
            "FragMethod": ["HCD", "HCD"],
            "Precursor": [500.0, 500.0],
            "Genome Coordinates": ["chr_plus:10-39(+)", "chr_minus:5-34(-)"],
            "ORF Sequence": ["MPEPTIDER", "MPEPTIDER"],
            "IsotopeError": [0, 0],
            "PrecursorError(ppm)": [0.5, 0.5],
            "Charge": [2, 2],
            "Peptide": ["K.PEPTIDER.A", "K.PEPTIDER.A"],
            "Protein": [
                "gORF_chr_plus_1_10-39_forward",
                "gORF_chr_minus_1_5-34_reverse",
            ],
            "DeNovoScore": [80, 80],
            "MSGFScore": [95, 95],
            "SpecEValue": [1e-10, 1e-10],
            "EValue": [1e-7, 1e-7],
        }
    ).to_csv(res02, sep="\t", index=False)

    args = Namespace(
        starts=["TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        stops=["TAA", "TAG", "TGA"],
        maxsize=300,
        minsize=18,
        genetic_code=11,
    )
    alts = AltCodons(file=res02, genome=genome_fa, maxsize=300, subset="Genome")
    # Must terminate immediately without hanging on chr_minus!
    alts.extend_orfs(args)
    alts.sort_by_coordinates()
    alts.sort_by_atg()
    alts.sort_by_peptides()
    priorities = alts.get_priorities()

    p_plus = priorities["chr_plus:39"][0]
    assert (p_plus.start, p_plus.end, p_plus.start_codon) == (1, 39, "ATG")
    assert p_plus.proteinSequence == "MKLVPEPTIDER"

    p_minus = priorities["chr_minus:5(-)"][0]
    assert (p_minus.start, p_minus.end, p_minus.start_codon) == (43, 5, "ATG")
    assert p_minus.proteinSequence == "MKLVPEPTIDER"

    # Zero debug litter written to CWD!
    assert not (tmp_path / "teste_peptide_sort.txt").exists()


def test_rv2975a_nterminal_peptide_never_truncated_by_downstream_atg_rbs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test reproducing M. tuberculosis Rv2975a (`RWDPNIQTVHAVTRPGGNK`):
    Upstream GTG start encodes `M` + `K` + `RWDPNIQTVHAVTRPGGNK` + `AGGAGG-like` internal region +
    downstream internal `ATG` + C-terminal peptide `ALEQAAK` + `TGA`.
    Even when the downstream internal `ATG` has a strong Shine-Dalgarno (`AGGAGG`) upstream of it
    while the upstream `GTG` has NO Shine-Dalgarno, `AltCodons` MUST retain the upstream `GTG`
    so the MS-evidenced N-terminal peptide `RWDPNIQTVHAVTRPGGNK` is NEVER chopped off!
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Genome" / "post_perc").mkdir(parents=True)

    # Construct reading frame:
    # 1..21:   21x 'C' (no RBS upstream of GTG at 22..24)
    # 22..24:  GTG (bacterial initiator -> M)
    # 25..27:  AAG (K -> tryptic cleavage before RWDPNIQTVHAVTRPGGNK!)
    # 28..84:  encode("RWDPNIQTVHAVTRPGGNK") (19 aa = 57 nt)
    # 85..96:  "AGGAGGGGCGAA" (4 codons = R R G E, and provides a strong "AGGAGG" SD right before internal ATG!)
    # 97..99:  "ATG" (internal Met!)
    # 100..102:"AAG" (K)
    # 103..123:encode("ALEQAAK") (7 aa = 21 nt)
    # 124..126:"TGA" (stop at 126!)
    rv_pep = "RWDPNIQTVHAVTRPGGNK"
    cterm_pep = "ALEQAAK"
    genome_seq = (
        ("C" * 21)
        + "GTG"
        + "AAG"
        + _encode_aa(rv_pep)
        + "AGGAGGGGCGAA"
        + "ATG"
        + "AAG"
        + _encode_aa(cterm_pep)
        + "TGA"
        + ("C" * 30)
    )
    genome_fa = tmp_path / "genome.fasta"
    genome_fa.write_text(f">NC_000962.3\n{genome_seq}\n")

    rrna_fa = tmp_path / "rrna.fna"
    rrna_fa.write_text(">16S\nACGTACGTGAUCACCUCCUUA\n")

    res02 = tmp_path / "Genome" / "post_perc" / "genome_results_02.txt"
    full_orf_aa = "MK" + rv_pep + "RRGEMK" + cterm_pep
    pd.DataFrame(
        {
            "SpecFile": ["mtb.mzML", "mtb.mzML"],
            "SpecID": ["index=10", "index=11"],
            "ScanNum": [100, 101],
            "FragMethod": ["HCD", "HCD"],
            "Precursor": [750.3, 400.2],
            "Genome Coordinates": ["22-126", "22-126"],
            "ORF Sequence": [f"{full_orf_aa},{full_orf_aa}", full_orf_aa],
            "IsotopeError": [0, 0],
            "PrecursorError(ppm)": [0.5, 0.4],
            "Charge": [3, 2],
            "Peptide": [f"K.{rv_pep}.R", f"K.{cterm_pep}.-"],
            "Protein": ["gORF_NC_000962.3_1_22-126_forward", "gORF_NC_000962.3_1_22-126_forward"],
            "DeNovoScore": [90, 82],
            "MSGFScore": [120, 95],
            "SpecEValue": [1e-13, 1e-10],
            "EValue": [1e-10, 1e-7],
        }
    ).to_csv(res02, sep="\t", index=False)

    args = Namespace(
        genome=genome_fa,
        rrna=rrna_fa,
        rbs_backend="python",
        circular=False,
        starts=["TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        stops=["TAA", "TAG", "TGA"],
        maxsize=300,
        minsize=18,
        genetic_code=11,
    )
    alts = AltCodons(file=res02, genome=genome_fa, maxsize=300, subset="Genome")
    alts.extend_orfs(args)
    sd = SDInspection(args, filetype="genome", folder="Genome", alternatives=alts.alternatives)
    alts.alternatives = sd.get_free_energy()
    alts.sort_by_coordinates()
    alts.sort_by_atg()
    alts.sort_by_shine()
    alts.sort_by_peptides()
    priorities = alts.get_priorities()

    chosen = priorities["126"][0]
    assert chosen.start == 22
    assert chosen.end == 126
    assert rv_pep in chosen.proteinSequence
    assert cterm_pep in chosen.proteinSequence

    ext = ExtendedInformation(folder="Genome", filetype="genome", alternatives=alts.alternatives)
    ext.filter_alternatives(priorities).extract_spectra()
    df04 = pd.read_csv(tmp_path / "Genome" / "post_perc" / "genome_results_04.txt", sep="\t")
    assert len(df04) == 2
    assert all(rv_pep in s for s in df04["Extended Sequence"])


def test_refines_upstream_ttg_to_internal_atg_with_strong_sd_and_ignores_thermal_noise(
    tmp_path: pathlib.Path,
):
    # Case A: Database ORF started at upstream TTG (no SD), MS peptide ALEQAAK is downstream of
    #         an internal ATG preceded by strong AGGAGG -> internal ATG+AGGAGG wins!
    # Case B: Canonical ATG (dG = 0.0) vs upstream TTG with thermal noise (dG = -1.2 > -3.4535)
    #         -> Canonical ATG wins (thermal noise NEVER overrides ATG!), and zero 2x duplication!
    cterm_pep = "ALEQAAK"
    seq_a = (
        ("C" * 21)
        + "TTG"          # pos 22..24 (upstream TTG, dG = 0.0)
        + "CCCAGGAGGTCATCA"  # 5 sense codons (15 nt: 25..39, CCC AGG AGG TCA TCA) containing AGGAGG before ATG!
        + "ATG"          # pos 40..42 (internal ATG with strong AGGAGG!)
        + "AAG"          # pos 43..45 (K)
        + _encode_aa(cterm_pep)  # pos 46..66 (21 nt)
        + "TGA"          # pos 67..69
        + ("C" * 20)
    )
    # Check frame alignment of seq_a: 22..24 (3) + 15 = 39 -> 40..42 is exact in-frame ATG!

    # Case B on contig2:
    # Upstream TTG at 22..24 has weak thermal-noise dG ~ -1.1 kcal/mol
    # Downstream ATG at 31..33 has dG = 0.0 ("CCCCCCCCCCCCCCCCCCCCC") -> ATG must beat TTG!
    seq_b = (
        "CCCCCCCCCCCCCCCCCCCAG"  # 21 nt upstream of TTG (weak/zero dG > -3.4535)
        + "TTG"                  # 22..24
        + "CCCCCC"               # 25..30 (2 codons)
        + "ATG"                  # 31..33 (canonical ATG!)
        + "AAG"                  # 34..36
        + _encode_aa(cterm_pep)  # 37..57
        + "TGA"                  # 58..60
        + ("C" * 20)
    )
    genome_fa = tmp_path / "genome.fasta"
    genome_fa.write_text(f">c1\n{seq_a}\n>c2\n{seq_b}\n")
    rrna_fa = tmp_path / "rrna.fna"
    rrna_fa.write_text(">16S\nACGTACGTGAUCACCUCCUUA\n")

    res02 = tmp_path / "res02.txt"
    pd.DataFrame(
        {
            "SpecFile": ["s.mzML", "s.mzML"],
            "SpecID": ["index=1", "index=2"],
            "ScanNum": [1, 2],
            "FragMethod": ["HCD", "HCD"],
            "Precursor": [400.0, 400.0],
            "Genome Coordinates": ["c1:22-69(+)", "c2:22-60(+)"],
            "ORF Sequence": ["M" + "PRRSS" + "MK" + cterm_pep, "MPP" + "MK" + cterm_pep],
            "IsotopeError": [0, 0],
            "PrecursorError(ppm)": [0.5, 0.5],
            "Charge": [2, 2],
            "Peptide": [f"K.{cterm_pep}.-", f"K.{cterm_pep}.-"],
            "Protein": ["gORF_c1_1_22-69_forward", "gORF_c2_1_22-60_forward"],
            "DeNovoScore": [80, 80],
            "MSGFScore": [95, 95],
            "SpecEValue": [1e-10, 1e-10],
            "EValue": [1e-7, 1e-7],
        }
    ).to_csv(res02, sep="\t", index=False)

    args = Namespace(
        genome=genome_fa,
        rrna=rrna_fa,
        rbs_backend="python",
        circular=False,
        starts=["TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"],
        stops=["TAA", "TAG", "TGA"],
        maxsize=300,
        minsize=18,
        genetic_code=11,
    )
    alts = AltCodons(file=res02, genome=genome_fa, maxsize=300, subset="Genome")
    alts.extend_orfs(args)
    count_before = len(alts.alternatives["c2:60"])
    sd = SDInspection(args, filetype="genome", folder="Genome", alternatives=alts.alternatives)
    alts.alternatives = sd.get_free_energy()
    # Force a small negative thermal-noise value (-0.8 kcal/mol) on c2's TTG to prove it never overrides ATG
    for cand in alts.alternatives["c2:60"]:
        if cand.start_codon == "TTG":
            cand.freeEnergy = -0.8

    alts.sort_by_shine()
    # Verify sort_by_shine NEVER duplicates candidates 2x!
    assert len(alts.alternatives["c2:60"]) == count_before
    alts.sort_by_peptides()
    priorities = alts.get_priorities()

    winner_c1 = priorities["c1:69"][0]
    assert winner_c1.start == 40
    assert winner_c1.start_codon == "ATG"
    assert winner_c1.freeEnergy <= -8.4

    winner_c2 = priorities["c2:60"][0]
    assert winner_c2.start == 31
    assert winner_c2.start_codon == "ATG"
