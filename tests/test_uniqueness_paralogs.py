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

import pandas as pd
import pytest

from src.postprocess.extinfo import ExtendedInformation
from src.postprocess.specfilt import Coordinator
from src.sequtils.locus import ORF
from src.sequtils.orflib import ORFCollection
from src.sequtils.unique import CanonicalPeptideExclusionIndex, PercolatorUTP


def test_canonical_peptide_exclusion_index_catches_il_isobaric_met_excision_and_contaminants(
    tmp_path: pathlib.Path,
):
    proteome = tmp_path / "proteome.fasta"
    proteome.write_text(
        ">Rv0001\n"
        "MKLLEIRKALK\n"
        ">Rv0002_met_excised\n"
        "MGASVPEPTIDERK\n"
    )
    idx = CanonicalPeptideExclusionIndex(proteome, include_contaminants=True)

    # Exact I<->L mass-isobaric variant of KLLEIRK -> KIIELRK!
    assert idx.contains_peptide("K.KIIELRK.A") is True
    # Initiator-Met excision on M-G (`GASVPEPTIDER`)
    assert idx.contains_peptide("-.GASVPEPTIDER.K") is True
    # Actual forward contaminant peptide from seqlib/contaminants.txt (UBIQUITIN07, with L->I swap)
    assert idx.contains_peptide("R.AQEIHTFEVTGQETVAQLK.A") is True
    # Genuinely novel peptide
    assert idx.contains_peptide("R.MNOVELSMORFPEPK.A") is False


def test_percolator_utp_collapses_same_locus_isoforms_and_annotates_paralogs(
    tmp_path: pathlib.Path,
):
    """Regression test (Issue #29 / H3):
    1. Isoforms `uproteins.10.1` and `uproteins.10.2` encoding the exact same genomic
       interval `(NC_000962.3, +, 200-247)` collapse to `Locus_Multiplicity == 1`!
    2. Duplicated paralogs at `100-160` and `5000-5060` get `Locus_Multiplicity == 2`
       and a deterministic `PARALOG_*` group ID.
    3. Canonical `I/L`-isobaric leak -> MUST BE EXCLUDED!
    """
    proteome = tmp_path / "proteome.fasta"
    proteome.write_text(">Rv0001\nMKLLEIRKALK\n")

    no_anno = tmp_path / "transcriptome_no_anno.txt"
    no_anno.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tGenome Coordinates\tproteinIds\n"
        # Row 1: two StringTie isoforms encoding the EXACT same genomic interval 200-247!
        "run1_SII_1_1_100_2_1\t2.9\t0.001\t0.001\t-.MISOFORMPEPK.R\t200-247,200-247\t"
        "tORF_uproteins.10.1_1_15-62_forward,tORF_uproteins.10.2_1_45-92_forward\n"
        # Row 2: duplicated toxin-antitoxin paralog at two disjoint genomic loci!
        "run1_SII_2_1_101_2_1\t3.2\t0.001\t0.0005\t-.MPARALOGSMORFK.R\t100-160,5000-5060\t"
        "tORF_uproteins.20.1_1_10-70_forward,tORF_uproteins.99.1_1_10-70_forward\n"
        # Row 3: canonical I/L-isobaric leak -> MUST BE EXCLUDED!
        "run1_SII_3_1_102_2_1\t3.5\t0.0005\t0.0002\tK.KIIELRK.A\t8000-8060\t"
        "tORF_uproteins.55.1_1_1-61_forward\n"
    )

    str_dict = {
        "uproteins.10.1": ORF("uproteins.10.1", start=185, end=400, strand="+", chromosome="NC_000962.3"),
        "uproteins.10.2": ORF("uproteins.10.2", start=155, end=450, strand="+", chromosome="NC_000962.3"),
        "uproteins.20.1": ORF("uproteins.20.1", start=90, end=300, strand="+", chromosome="NC_000962.3"),
        "uproteins.99.1": ORF("uproteins.99.1", start=4990, end=5200, strand="+", chromosome="NC_000962.3"),
    }

    out_utps = tmp_path / "transcriptome_utps.txt"
    utp = PercolatorUTP(
        coord_df=no_anno,
        pep=1.0,
        qvalue=0.01,
        proteome_fasta=proteome,
        stringtie_dict=str_dict,
        max_paralog_loci=10,
        unique_loci_only=False,
        include_contaminants=False,
    )
    utp.get_utps().save_table(output=out_utps, keep="paralogs")

    df = pd.read_csv(out_utps, sep="\t")
    assert len(df) == 2

    iso_row = df[df["PSMId"] == "run1_SII_1_1_100_2_1"].iloc[0]
    assert bool(iso_row["Unique Peptide"]) is True
    assert int(iso_row["Locus_Multiplicity"]) == 1
    assert iso_row["Paralog_Group_ID"] == "SINGLE_LOCUS"
    # Redundant second isoform sharing exact genomic interval collapsed to 1 representative!
    assert "," not in iso_row["proteinIds"]

    para_row = df[df["PSMId"] == "run1_SII_2_1_101_2_1"].iloc[0]
    assert bool(para_row["Unique Peptide"]) is False
    assert int(para_row["Locus_Multiplicity"]) == 2
    assert str(para_row["Paralog_Group_ID"]).startswith("PARALOG_")
    assert len(para_row["proteinIds"].split(",")) == 2

    # Also verify --unique_loci_only keeps strictly Locus_Multiplicity == 1
    out_unique_only = tmp_path / "unique_only.txt"
    utp_uniq = PercolatorUTP(
        coord_df=no_anno,
        pep=1.0,
        qvalue=0.01,
        proteome_fasta=proteome,
        stringtie_dict=str_dict,
        unique_loci_only=True,
        include_contaminants=False,
    )
    utp_uniq.get_utps().save_table(output=out_unique_only, keep="unique")
    df_uniq = pd.read_csv(out_unique_only, sep="\t")
    assert len(df_uniq) == 1
    assert df_uniq.iloc[0]["PSMId"] == "run1_SII_1_1_100_2_1"


def test_extended_information_preserves_paralogous_smorf_loci_for_same_spectrum(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #29 / H3):
    Previously, `ExtendedInformation.extract_spectra()` called
    `drop_duplicates(subset=["SpecFile", "ScanNum"], keep="last")`, deleting all
    paralogous novel smORF loci except the last one! Verify both paralogous loci
    (`100-160` and `5000-5060`) matched by `ScanNum == 200` survive in `results_04.txt`.
    """
    monkeypatch.chdir(tmp_path)
    perc_dir = tmp_path / "Genome" / "post_perc"
    perc_dir.mkdir(parents=True)

    utps = perc_dir / "genome_utps.txt"
    utps.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tGenome Coordinates\t"
        "proteinIds\tUnique Peptide\tLocus_Multiplicity\tParalog_Group_ID\n"
        "run1_SII_1_1_200_2_1\t3.1\t0.001\t0.0005\t-.MPARALOGPEPK.R\t100-160,5000-5060\t"
        "gORF_c1_1_100-160_forward,gORF_c1_2_5000-5060_forward\tFalse\t2\tPARALOG_abcd1234\n"
    )
    proteined = perc_dir / "genome_proteined.tsv"
    proteined.write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tORF Sequence\tIsotopeError\t"
        "PrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
        "run1.mzML\tindex=20\t200\tHCD\t600.3\tMPARALOGPEPKR,MPARALOGPEPKR\t0\t0.8\t2\t-.MPARALOGPEPK.R\t"
        "gORF_c1_1_100-160_forward(pre=-,post=R);gORF_c1_2_5000-5060_forward(pre=-,post=R)\t88\t104\t1e-11\t1e-8\n"
    )

    res02 = perc_dir / "genome_results_02.txt"
    Coordinator(utps=utps, proteined=proteined, qvalue=0.01, pep=1.0).add_information(res02)
    df02 = pd.read_csv(res02, sep="\t")
    assert len(df02) == 2
    assert list(df02["Locus_Multiplicity"]) == [2, 2]

    # Build two distinct paralog ORFCollections keyed by their distinct stop coordinates
    orf1 = ORF(
        name="gORF_c1_1_100-160_forward",
        start=100,
        end=160,
        strand="forward",
        seq="ATGCCCGCCCGCCGC",
        protein_sequence="MPARALOGPEPKR",
    )
    orf1.freeEnergy = -9.2
    orf1.shineDalgarno = "Strong"
    col1 = ORFCollection().add_orfs([orf1])
    col1.set_priority(orf1)

    orf2 = ORF(
        name="gORF_c1_2_5000-5060_forward",
        start=5000,
        end=5060,
        strand="forward",
        seq="ATGCCCGCCCGCCGC",
        protein_sequence="MPARALOGPEPKR",
    )
    orf2.freeEnergy = -8.7
    orf2.shineDalgarno = "Strong"
    col2 = ORFCollection().add_orfs([orf2])
    col2.set_priority(orf2)

    alts = {"160": col1, "5060": col2}
    ext = ExtendedInformation(folder="Genome", filetype="genome", alternatives=alts)
    ext.filter_alternatives(["gORF_c1_1_100-160_forward", "gORF_c1_2_5000-5060_forward"])
    ext.extract_spectra()

    df04 = pd.read_csv(perc_dir / "genome_results_04.txt", sep="\t")
    assert len(df04) == 2
    assert sorted(df04["Extended Coordinates"].tolist()) == ["100-160", "5000-5060"]
    assert sorted(df04["Extended ORF"].tolist()) == [
        "gORF_c1_1_100-160_forward",
        "gORF_c1_2_5000-5060_forward",
    ]
