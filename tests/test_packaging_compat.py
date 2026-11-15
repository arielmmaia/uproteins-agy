# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for Issue #1: modern Python/pandas/sklearn compatibility and non-circular imports."""

import importlib
import subprocess
import sys

import pandas as pd
import pytest

from src.forest.miner import MinePredicted, SpectrumMiner
from src.forest.preforest import FeatureFishing, PreFiltering
from src.postprocess.extinfo import ExtendedInformation
from src.postprocess.specfilt import Coordinator, ProteinFDR
from src.sequtils.orflib import ORF, ORFCollection


def test_non_circular_imports():
    """Importing uProteInS and src in fresh subprocesses must succeed in either order."""
    for stmt in ("import src; import uProteInS", "import uProteInS; import src"):
        proc = subprocess.run(
            [sys.executable, "-c", stmt],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Failed '{stmt}': {proc.stderr}"

    mod = importlib.import_module("src")
    assert callable(mod.uproteins)
    assert callable(mod.run_workflow)


def test_pandas_2_concat_coordinator(tmp_path):
    """Coordinator.add_information must work on pandas >= 2.2 without DataFrame.append."""
    utps = tmp_path / "genome_utps.txt"
    proteined = tmp_path / "genome_proteined.tsv"
    out = tmp_path / "genome_results_02.txt"

    utps.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tGenome Coordinates\tproteinIds\n"
        "run1_SII_1_1_100_2_1\t2.5\t0.005\t0.001\t-.MPEPTIDER.K\t10-70\tGenome_gORF_1_10-70_forward\n"
    )
    proteined.write_text(
        "#SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tORF Sequence\tIsotopeError\tPrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
        "run1.mzML\tindex=1\t100\tHCD\t500.25\tMPEPTIDERK\t0\t1.2\t2\t-.MPEPTIDER.K\tGenome_gORF_1_10-70_forward(pre=-,post=K)\t80\t95\t1e-10\t1e-7\n"
    )

    coord = Coordinator(utps=str(utps), proteined=str(proteined), qvalue=0.01)
    coord.add_information(str(out))

    res = pd.read_csv(out, sep="\t")
    assert len(res) == 1
    assert res.loc[0, "Genome Coordinates"] == "10-70"
    assert res.loc[0, "Protein"] == "Genome_gORF_1_10-70_forward"


def test_pandas_2_prefiltering_and_feature_fishing(tmp_path):
    """PreFiltering and FeatureFishing must work on pandas >= 2.2 without Series.any(level=0) or DataFrame.append."""
    pin_dir = tmp_path / "pins"
    filtered_dir = tmp_path / "filtered_pins"
    pin_dir.mkdir()

    pin_file = pin_dir / "fixed_pin_1.txt"
    pin_file.write_text(
        "SpecId\tLabel\tScanNr\tExpMass\tPeptide\tProteins\n"
        "run1_100_2_1\t1\t100\t1000.5\t-.MPEPTIDER.K\tGenome_gORF_1_10-70_forward\n"
        "run1_200_2_1\t-1\t200\t1000.5\t-.RPEPTIDEM.K\tdecoy_Genome_gORF_9\n"
    )

    results_04 = tmp_path / "genome_results_04.txt"
    results_04.write_text(
        "SpecFile\tSpecID\tScanNum\tExtended ORF\tExtended Sequence\tFree Energy\tShine Dalgarno\tExtended Coordinates\tPeptide\tProtein\n"
        "run1.mzML\tindex=1\t100\tGenome_gORF_1_10-70_forward\tMPEPTIDERK\t-7.5\tStrong\t10-70\t-.MPEPTIDER.K\tGenome_gORF_1_10-70_forward\n"
    )

    pf = PreFiltering(pin_folder=str(pin_dir), results_04=str(results_04))
    pf.filter_proteins()
    pf.filter_pin_files(str(filtered_dir))

    filtered_df = pd.read_csv(filtered_dir / "filtered_fixed_pin_1.txt", sep="\t")
    assert len(filtered_df) == 1
    assert filtered_df.loc[0, "ScanNr"] == 100

    ff = FeatureFishing(results=str(results_04), pin_folder=str(filtered_dir))
    ff.add_features()
    out_features = tmp_path / "features.txt"
    ff.save_table(str(out_features))
    feat_df = pd.read_csv(out_features, sep="\t")
    assert len(feat_df) == 1
    assert feat_df.loc[0, "ScanNr"] == 100


def test_pandas_2_extended_info_and_miners(tmp_path, monkeypatch):
    """ExtendedInformation, ForestCleaning, MinePredicted, and SpectrumMiner must run without pandas 1.x errors."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Genome" / "post_perc").mkdir(parents=True)

    res02 = tmp_path / "Genome" / "post_perc" / "genome_results_02.txt"
    res02.write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tGenome Coordinates\tPeptide\tProtein\n"
        "run1.mzML\t100\t100\tHCD\t500.2\t10-70\t-.MPEPTIDER.K\tGenome_gORF_1_10-70_forward\n"
    )

    orf = ORF(name="Genome_gORF_1_10-70_forward", start=10, end=70, strand="forward")
    orf.proteinSequence = "MPEPTIDERK"
    orf.freeEnergy = -8.9
    orf.shineDalgarno = "Strong binding"
    coll = ORFCollection()
    coll.add_orf(orf)
    coll.set_priority(orf)

    ext = ExtendedInformation("Genome", "genome", {"70": coll})
    ext.extract_spectra()
    res04_path = tmp_path / "Genome" / "post_perc" / "genome_results_04.txt"
    assert res04_path.is_file()
    res04 = pd.read_csv(res04_path, sep="\t")
    assert len(res04) == 1
    assert res04.loc[0, "Extended ORF"] == "Genome_gORF_1_10-70_forward"

    # MinePredicted & SpectrumMiner
    pred_path = tmp_path / "predicted.txt"
    pred_path.write_text(
        "SpecId\tLabel\tPrediction\tScanNr\n"
        "run1_SII_1_1_100_2_1\t1\t1\t100\n"
    )
    mp = MinePredicted(str(pred_path), str(res04_path))
    mp_out = tmp_path / "mp_out.txt"
    mp.filter(str(mp_out))
    assert len(pd.read_csv(mp_out, sep="\t")) == 1

    sm = SpectrumMiner(results=str(res04_path), predicted=str(pred_path))
    sm_out = tmp_path / "sm_out.txt"
    sm.filter_results(str(sm_out))
    assert len(pd.read_csv(sm_out, sep="\t")) == 1

    # ProteinFDR
    (tmp_path / "Genome" / "Percolator").mkdir(parents=True, exist_ok=True)
    (tmp_path / "Genome" / "Percolator" / "protein_results.txt").write_text(
        "ProteinId\tproteinGroupId\tq-value\tposterior_error_prob\tpeptideIds\n"
        "Genome_gORF_1_10-70_forward\t1\t0.005\t0.001\tMPEPTIDER\n"
    )
    (tmp_path / "Genome" / "post_perc" / "genome_proteined.tsv").write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tORF Sequence\tIsotopeError\tPrecursorError(ppm)\tCharge\tPeptide\tProtein\n"
        "run1.mzML\t100\t100\tHCD\t500.2\tMPEPTIDERK\t0\t1.0\t2\t-.MPEPTIDER.K\tGenome_gORF_1_10-70_forward(pre=-,post=K)\n"
    )
    (tmp_path / "Genome" / "post_perc" / "genome_utps.txt").write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tGenome Coordinates\tproteinIds\n"
        "run1_SII_1_1_100_2_1\t2.5\t0.005\t0.001\t-.MPEPTIDER.K\t10-70\tGenome_gORF_1_10-70_forward\n"
    )
    pfdr = ProteinFDR("Genome", "genome")
    pfdr.protein_cutoff()
    pfdr.apply_to_psm()
    pfdr.filter_from_utp()
    assert len(pd.read_csv(tmp_path / "Genome" / "post_perc" / "genome_results_01.txt", sep="\t")) == 1
