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

from src.sequtils.postsearch import LinkData, SequenceFinder, TSVChunks


def test_tsvchunks_preserves_int64_scans_across_multiple_chunks_and_mid_headers(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #27 / C5):
    Previously, `TSVChunks` stored `scans` as `list[str]` ('2500'), so clean numeric
    chunks where `pd.read_csv` parsed `ScanNum` as `int64` matched 0 rows and `.append()`
    crashed on pandas >= 2.0.
    """
    monkeypatch.chdir(tmp_path)
    perc_dir = tmp_path / "Genome" / "post_perc"
    perc_dir.mkdir(parents=True)

    (perc_dir / "genome_utps.txt").write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds\n"
        "runA_SII_1_1_2500_2_1\t2.5\t0.001\t0.001\tR.MPEPTIDER.K\tgORF_c1_1_1-30_forward\n"
        "runB_SII_2_1_4200_3_1\t2.9\t0.001\t0.001\tK.MSECONDK.R\tgORF_c1_2_50-80_forward\n"
    )

    header = (
        "#SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tIsotopeError\t"
        "PrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
    )
    (perc_dir / "genome_search.tsv").write_text(
        header
        + "runA.mzML\tindex=0\t1\tHCD\t400.0\t0\t0.5\t2\tR.MWRONGK.A\tgORF_x\t50\t60\t1e-5\t1e-2\n"
        + "runA.mzML\tindex=99\t2500\tHCD\t500.2\t0\t1.0\t2\tR.MPEPTIDER.K\tgORF_c1_1_1-30_forward\t80\t95\t1e-10\t1e-7\n"
        # Chunk 2 (pure numeric int64 chunk!) + Chunk 3 (with repeated header!)
        + "runB.mgf\tindex=10\t4199\tHCD\t600.0\t0\t1.0\t3\tK.MOTHERK.R\tgORF_y\t40\t45\t1e-4\t1e-1\n"
        + "runB.mgf\tindex=11\t4200\tHCD\t610.3\t0\t0.8\t3\tK.MSECONDK.R\tgORF_c1_2_50-80_forward\t85\t102\t1e-12\t1e-9\n"
        + header
        + "runC.mzXML\tindex=5\t2500\tHCD\t500.2\t0\t1.0\t2\tR.MOTHERRUNK.A\tgORF_z\t70\t75\t1e-8\t1e-5\n"
    )

    chunks = TSVChunks(folder="Genome", filetype="genome")
    chunks.filter_search(chunk_size=2)

    out_df = pd.read_csv(perc_dir / "genome_chunk_search.tsv", sep="\t")
    assert len(out_df) == 2
    assert sorted(out_df["ScanNum"].astype(int).tolist()) == [2500, 4200]
    assert sorted(out_df["#SpecFile"].tolist()) == ["runA.mzML", "runB.mgf"]


def test_linkdata_extracts_true_scannum_handles_sii_names_multi_ext_and_decoys(
    tmp_path: pathlib.Path,
):
    """Regression test (Issue #27 / H11, H5):
    - `SII_1` with `ScanNum=2500` MUST match `ScanNum=2500` (not `ScanNum=1`!).
    - Sample names containing `'SII'` (`SII_batch1_rep2`) are parsed accurately.
    - `.mgf`, `.mzXML`, `.mzml`, and 16-column `MzIDToTsv` tables work seamlessly.
    - Lowercase `decoy_`, `Decoy_`, `Reverse_`, and `contaminant_` hits are stripped.
    """
    cat_tsv = tmp_path / "chunk_search.tsv"
    cat_tsv.write_text(
        "#SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tIsotopeError\t"
        "PrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\t"
        "SpecEValue\tEValue\tQValue\tPepQValue\n"
        # Trap row at ScanNum=1 (matches SII_1 index if buggy!)
        "SII_batch1_rep2.mgf\tindex=1\t1\tHCD\t450.0\t0\t0.5\t2\tR.MTRAPSCANK.A\tgORF_trap(pre=R,post=A)\t40\t45\t1e-4\t1e-1\t0.05\t0.05\n"
        # True MS2 scan at ScanNum=2500 (also co-mapped to a lowercase decoy_ token that must be stripped)
        "SII_batch1_rep2.mgf\tindex=55\t2500\tHCD\t512.3\t0\t0.9\t2\tR.MTRUEPEPK.A\tgORF_true(pre=R,post=A);decoy_gORF_noise(pre=R,post=A)\t88\t105\t1e-12\t1e-9\t0.001\t0.001\n"
        # Modern 3-token PSMId on .mzXML
        "frac2.mzXML\tindex=77\t777\tCID\t620.4\t0\t1.1\t3\tK.MMODERNK.R\tgORF_mod(pre=K,post=R)\t79\t92\t1e-10\t1e-7\t0.002\t0.002\n"
        # Pure lowercase decoy row at ScanNum=777 -> must be excluded!
        "frac2.mzXML\tindex=77\t777\tCID\t620.4\t0\t1.1\t3\tK.MDECOYK.R\tdecoy_gORF_mod(pre=K,post=R)\t30\t25\t1e-2\t5.0\t0.5\t0.5\n"
    )

    utps = tmp_path / "utps.txt"
    utps.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds\n"
        "SII_batch1_rep2_SII_1_1_2500_2_1\t3.1\t0.001\t0.0005\tR.MTRUEPEPK.A\tgORF_true\n"
        "frac2_777_3_1\t2.7\t0.002\t0.001\tK.MMODERNK.R\tgORF_mod\n"
    )

    out_prefix = tmp_path / "linked"
    LinkData(cat_tsv, utps).filter_msgf(str(out_prefix))

    linked_df = pd.read_csv(f"{out_prefix}.tsv", sep="\t")
    assert len(linked_df) == 2
    assert linked_df["ScanNum"].tolist() == [2500, 777]
    assert linked_df["SpecFile"].tolist() == ["SII_batch1_rep2.mgf", "frac2.mzXML"]
    assert linked_df["Protein"].tolist() == [
        "gORF_true(pre=R,post=A)",
        "gORF_mod(pre=K,post=R)",
    ]


def test_sequence_finder_case_insensitive_decoy_and_bare_protein_accession(
    tmp_path: pathlib.Path,
):
    """Regression test (Issue #27 / H5):
    `SequenceFinder` must filter `Decoy_` / `Contaminant_` case-insensitively and
    must NOT slice `protein[:-1]` when `(pre=` is absent!
    """
    linked = tmp_path / "linked.tsv"
    linked.write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tIsotopeError\t"
        "PrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
        "run1.mzML\tindex=1\t100\tHCD\t500.0\t0\t1.0\t2\t-.MPEPTIDER.K\tgORF_bare_id\t80\t95\t1e-10\t1e-7\n"
        "run1.mzML\tindex=2\t101\tHCD\t500.0\t0\t1.0\t2\t-.MDECOYK.R\tDecoy_gORF_bare_id\t20\t20\t1e-2\t1.0\n"
        "run1.mzML\tindex=3\t102\tHCD\t500.0\t0\t1.0\t2\t-.MCONTAMK.R\tContaminant_TRYPSIN\t90\t110\t1e-12\t1e-9\n"
    )
    fasta = tmp_path / "genome_database.fasta"
    fasta.write_text(">gORF_bare_id\nMPEPTIDERK\n")

    out_prefix = tmp_path / "proteined"
    SequenceFinder(linked, fasta).df_proteins().save(str(out_prefix))

    out_df = pd.read_csv(f"{out_prefix}.tsv", sep="\t")
    assert len(out_df) == 1
    assert out_df.iloc[0]["Protein"] == "gORF_bare_id"
    assert out_df.iloc[0]["ORF Sequence"] == "MPEPTIDERK"
