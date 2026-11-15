# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression and scientific validity tests for Issue #41 (Metrics stage)."""

from __future__ import annotations

import argparse

from Bio import SeqIO
import numpy as np
import pandas as pd
import pytest

from src.main import uproteins
from src.metrics.metrics import Metrics


def _write_result_tsv(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_metrics_genome_only_auto_detection_and_nan_safety(tmp_path):
    """uProteInS metrics must succeed on genome-only runs (no Transcriptome dir) and handle NaN RBS/energies."""
    g_res = tmp_path / "Genome" / "Results"
    _write_result_tsv(
        g_res / "genome_pre_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_10",
                "ScanNum": 100,
                "Protein sequence": "MAAAAKKKKR",
                "Free Energy": -8.9,
                "Shine Dalgarno": "Present. Strong binding.",
                "Genome Coordinates": "10-42",
                "Peptide": "-.MAAAAKKKKR.-",
            },
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_10",
                "ScanNum": 200,
                "Protein sequence": "MGGGGLLLLR",
                "Free Energy": np.nan,
                "Shine Dalgarno": np.nan,
                "Genome Coordinates": "100-132",
                "Peptide": "-.MGGGGLLLLR.-",
            },
        ],
    )
    _write_result_tsv(
        g_res / "genome_post_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_10",
                "ScanNum": 100,
                "Protein sequence": "MAAAAKKKKR",
                "Free Energy": -8.9,
                "Shine Dalgarno": "Present. Strong binding.",
                "Genome Coordinates": "10-42",
                "Peptide": "-.MAAAAKKKKR.-",
            }
        ],
    )

    uproteins(["metrics", "-o", str(tmp_path)])

    metrics_dir = tmp_path / "metrics"
    assert (metrics_dir / "aa_distribution.pdf").is_file()
    assert (metrics_dir / "identified_microproteins.pdf").is_file()
    assert (metrics_dir / "energies_distribution.pdf").is_file()
    assert (metrics_dir / "shine-dalgarno.pdf").is_file()

    # Check unique FASTA headers even when two distinct sequences shared ORF name 'gORF_10'
    pre_records = list(SeqIO.parse(str(g_res / "Pre-validation.fasta"), "fasta"))
    assert len(pre_records) == 2
    assert len({rec.id for rec in pre_records}) == 2
    assert {rec.id for rec in pre_records} == {"gORF_10|10-42", "gORF_10|100-132"}


def test_metrics_100pct_aa_normalization_leaderless_rbs_non_redundant_and_venns(tmp_path, monkeypatch):
    """Amino acid distributions must sum to 100% per (Database, Subset), preserve Leaderless without double-counting, honor --non-redundant per subset, and render Venns."""
    monkeypatch.chdir(tmp_path)
    g_res = tmp_path / "Genome" / "Results"
    t_res = tmp_path / "Transcriptome" / "Results"

    # Shared bORF sequence 'MPEPTIDERK' + unique sequences in Genome and Transcriptome
    _write_result_tsv(
        g_res / "genome_pre_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_1",
                "ScanNum": 10,
                "Protein sequence": "MPEPTIDERK",
                "Free Energy": -9.1,
                "Shine Dalgarno": "Present. Strong binding.",
                "Genome Coordinates": "10-42",
            },
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_2",
                "ScanNum": 20,
                "Protein sequence": "MWWWYYYRRR",
                "Free Energy": -1.2,
                "Shine Dalgarno": "None (SD absent)",
                "Genome Coordinates": "200-232",
            },
        ],
    )
    _write_result_tsv(
        g_res / "genome_post_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "gORF_1",
                "ScanNum": 10,
                "Protein sequence": "MPEPTIDERK",
                "Free Energy": -9.1,
                "Shine Dalgarno": "Present. Strong binding.",
                "Genome Coordinates": "10-42",
            }
        ],
    )

    _write_result_tsv(
        t_res / "transcriptome_pre_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "tORF_1",
                "ScanNum": 10,
                "Protein sequence": "MPEPTIDERK",
                "Free Energy": 0.0,
                "Shine Dalgarno": "Leaderless",
                "Genome Coordinates": "10-42",
            },
            {
                "SpecFile": "run1.mzML",
                "ORF name": "tORF_2",
                "ScanNum": 30,
                "Protein sequence": "MCCCFFFKKK",
                "Free Energy": -5.0,
                "Shine Dalgarno": "Present. Low to moderate binding.",
                "Genome Coordinates": "500-532",
            },
        ],
    )
    _write_result_tsv(
        t_res / "transcriptome_post_validation_results.txt",
        [
            {
                "SpecFile": "run1.mzML",
                "ORF name": "tORF_1",
                "ScanNum": 10,
                "Protein sequence": "MPEPTIDERK",
                "Free Energy": 0.0,
                "Shine Dalgarno": "Leaderless",
                "Genome Coordinates": "10-42",
            }
        ],
    )

    args = argparse.Namespace(maxsize=100, non_redundant=False, transcriptome=True)
    m = Metrics(args=args)
    m.count_sequences()
    aa_df = m.plot_aa_dist()
    rbs_df = m.plot_rbs()
    m.generate_fasta()
    venn_pngs = m.plot_venn()
    venn2_pdf = m.plot_venn_2()

    # 1. Every (Database, Subset) amino acid frequency distribution MUST sum to 100.0% (never ~50%!)
    for (db, sub), grp in aa_df.groupby(["Database", "Subset"]):
        assert grp["Frequency_Pct"].sum() == pytest.approx(100.0, abs=1e-6), f"{db}/{sub} did not sum to 100%"

    # 2. RBS summary MUST preserve 'Leaderless' vs 'None (SD absent)' and NEVER double-count Validated into Pre-validation
    assert "Leaderless" in set(rbs_df["RBS"])
    assert "None (SD absent)" in set(rbs_df["RBS"])
    assert "Absent" not in set(rbs_df["RBS"])
    assert int(rbs_df[rbs_df["Subset"] == "Pre-validation"]["Count"].sum()) == 4
    assert int(rbs_df[rbs_df["Subset"] == "Validated"]["Count"].sum()) == 2

    # 3. Both Venn helpers MUST succeed and write output files
    assert len(venn_pngs) == 2
    for vp in venn_pngs:
        assert (tmp_path / "metrics" / vp.split("/")[-1]).is_file()
    assert venn2_pdf is not None and (tmp_path / "metrics" / "venn_pre_validation.pdf").is_file()

    # 4. --non-redundant MUST deduplicate shared sequences across DBs per Subset while keeping Validated intact!
    args_nr = argparse.Namespace(maxsize=100, non_redundant=True, transcriptome=True)
    m_nr = Metrics(args=args_nr)
    df_nr = m_nr.count_sequences()
    # Pre-validation had 3 unique sequences (MPEPTIDERK, MWWWYYYRRR, MCCCFFFKKK); Validated had 1 unique sequence (MPEPTIDERK)
    assert len(df_nr[df_nr["Subset"] == "Pre-validation"]) == 3
    assert len(df_nr[df_nr["Subset"] == "Validated"]) == 1
