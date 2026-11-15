# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for Issue #39 (ValidatePipeline, PreFiltering, FeatureFishing, SpectrumMiner & ResultsSummarizer)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from src.forest.miner import MinePredicted, SpectrumMiner
from src.forest.pinfixer import ProteinFixer
from src.forest.preforest import FeatureFishing, PreFiltering
from src.pipelines.validate import ValidatePipeline
from src.postprocess.results_summarizer import ResultsSummarizer


def _pin_row(
    spec_id: str,
    scan: int,
    protein: str,
    *,
    hc: bool = True,
    label: int = 1,
    peptide: str = "-.MPEPTIDERK.-",
) -> dict:
    if hc:
        return {
            "SpecId": spec_id,
            "Label": label,
            "ScanNr": scan,
            "ExpMass": 527.785,
            "CalcMass": 527.786,
            "RawScore": 134.0,
            "DeNovoScore": 134.0,
            "ScoreRatio": 1.0,
            "Energy": 0.0,
            "lnEValue": 15.5864,
            "IsotopeError": 0,
            "lnExplainedIonCurrentRatio": 0.1672,
            "lnNTermIonCurrentRatio": -0.8809,
            "lnCTermIonCurrentRatio": -0.2644,
            "lnMS2IonCurrent": 11.5406,
            "Mass": 527.785,
            "PepLen": 10,
            "dM": 0.0,
            "absdM": 0.0,
            "MeanErrorTop7": 12.0,
            "sqMeanErrorTop7": 4228.07,
            "StdevErrorTop7": 65.0,
            "Charge1": 0,
            "Charge2": 1,
            "enzN": 1,
            "enzC": 1,
            "enzInt": 0,
            "Peptide": peptide,
            "Proteins": protein,
        }
    return {
        "SpecId": spec_id,
        "Label": label,
        "ScanNr": scan,
        "ExpMass": 685.338,
        "CalcMass": 684.849,
        "RawScore": -38.0,
        "DeNovoScore": 4.0,
        "ScoreRatio": -1.0,
        "Energy": 42.0,
        "lnEValue": -1.6396,
        "IsotopeError": 1,
        "lnExplainedIonCurrentRatio": -2.1355,
        "lnNTermIonCurrentRatio": -2.8792,
        "lnCTermIonCurrentRatio": -2.7789,
        "lnMS2IonCurrent": 7.8959,
        "Mass": 685.338,
        "PepLen": 11,
        "dM": -1.9e-5,
        "absdM": 1.9e-5,
        "MeanErrorTop7": 247.5,
        "sqMeanErrorTop7": 61288.0,
        "StdevErrorTop7": 360.0,
        "Charge1": 0,
        "Charge2": 1,
        "enzN": 1,
        "enzC": 1,
        "enzInt": 1,
        "Peptide": peptide,
        "Proteins": protein,
    }


@pytest.mark.validate
def test_feature_fishing_no_sample1_sample10_collision_and_no_decoy_leak(tmp_path):
    """FeatureFishing must match normalized spectrum run stem by exact equality and drop Label == -1 decoys."""
    pin_dir = tmp_path / "pins"
    pin_dir.mkdir()

    # Same ScanNr=100 across sample_1, sample_10, and a decoy on sample_1
    pin_df = pd.DataFrame(
        [
            _pin_row("sample_10_100_2_1", 100, "Genome_gORF_99_1-60_forward", hc=True),
            _pin_row("sample_1_100_2_1", 100, "Genome_gORF_1_10-70_forward", hc=True),
            _pin_row("decoy_sample_1_100_2_1", 100, "decoy_Genome_gORF_1_10-70_forward", hc=True, label=-1),
        ]
    )
    pin_df.to_csv(pin_dir / "fixed_pin.txt", sep="\t", index=False)

    res04 = tmp_path / "results_04.txt"
    pd.DataFrame(
        [
            {
                "SpecFile": "sample_1.mgf",
                "SpecID": "index=0",
                "ScanNum": 100,
                "Extended ORF": "Genome_gORF_1_10-70_forward",
                "Extended Sequence": "MPEPTIDERK",
                "Free Energy": -8.8,
                "Shine Dalgarno": "Present. Strong binding.",
                "Extended Coordinates": "10-70",
                "Peptide": "-.MPEPTIDERK.-",
                "Protein": "Genome_gORF_1_10-70_forward",
            }
        ]
    ).to_csv(res04, sep="\t", index=False)

    fishing = FeatureFishing(results=str(res04), pin_folder=str(pin_dir)).add_features()
    assert len(fishing.dataWithFeatures) == 1
    assert fishing.dataWithFeatures.loc[0, "SpecId"] == "sample_1_100_2_1"


@pytest.mark.validate
def test_spectrum_miner_modern_msgf2pin_and_mgf_mzxml_extensions_no_renamed_leak(tmp_path):
    """SpectrumMiner and MinePredicted must parse modern 3-suffix msgf2pin SpecIds and .mgf/.mzXML files cleanly."""
    res04 = tmp_path / "genome_results_04.txt"
    pd.DataFrame(
        [
            {
                "SpecFile": "sample_alpha_1.mgf",
                "SpecID": "index=1",
                "ScanNum": 150,
                "Extended ORF": "Genome_gORF_1_10-70_forward",
                "Extended Sequence": "MPEPTIDERK",
                "Free Energy": -9.0,
                "Shine Dalgarno": "Present. Strong binding.",
                "Extended Coordinates": "10-70",
                "Peptide": "-.MPEPTIDERK.-",
                "Protein": "Genome_gORF_1_10-70_forward",
            },
            {
                "SpecFile": "sample_beta_2.mzXML",
                "SpecID": "index=2",
                "ScanNum": 250,
                "Extended ORF": "Genome_gORF_2_100-160_forward",
                "Extended Sequence": "MVVVPEPTIDERK",
                "Free Energy": -6.0,
                "Shine Dalgarno": "Present. Low to moderate binding.",
                "Extended Coordinates": "100-160",
                "Peptide": "-.MVVVPEPTIDERK.-",
                "Protein": "Genome_gORF_2_100-160_forward",
            },
        ]
    ).to_csv(res04, sep="\t", index=False)

    # Modern 3-suffix msgf2pin SpecIds (which legacy split('_')[:-6] wiped to '')
    pred = tmp_path / "predicted.txt"
    pd.DataFrame(
        [
            {"SpecId": "sample_alpha_1_150_2_1", "Label": 1, "Prediction": 1, "ScanNr": 150},
            {"SpecId": "sample_beta_2_target_SII_9_1_250_3_1", "Label": 1, "Prediction": 1, "ScanNr": 250},
        ]
    ).to_csv(pred, sep="\t", index=False)

    out05 = tmp_path / "genome_results_05.txt"
    miner = SpectrumMiner(results=str(res04), predicted=str(pred))
    miner.filter_results(out05)
    df05 = pd.read_csv(out05, sep="\t")
    assert len(df05) == 2
    assert "RenamedFiles" not in df05.columns

    mp_out = tmp_path / "mp_out.txt"
    MinePredicted(str(pred), str(res04)).filter(mp_out)
    assert len(pd.read_csv(mp_out, sep="\t")) == 2


@pytest.mark.validate
def test_validate_pipeline_end_to_end_rescued_lc_wiring_and_summarizer_bool(tmp_path, monkeypatch):
    """ValidatePipeline must execute PreFiltering, wire rescued LC PSMs of HC ORFs into post_validation_results, and summarize Transcriptome when transcriptome=True."""
    monkeypatch.chdir(tmp_path)

    for folder, ftype, prefix in (("Genome", "genome", "gORF"), ("Transcriptome", "transcriptome", "tORF")):
        (tmp_path / folder / "Percolator").mkdir(parents=True)
        (tmp_path / folder / "post_perc").mkdir(parents=True)
        (tmp_path / folder / "Results").mkdir(parents=True)

        orf_hc = f"{folder}_{prefix}_1_10-70_forward"
        orf_lc_only = f"{folder}_{prefix}_2_200-260_forward"

        # Ragged multi-tab Proteins line in raw PIN file + 1 HC scan (101), 1 sibling LC scan (102) on same HC ORF, and 1 LC-only ORF scan (303)
        pin_rows = [
            _pin_row("runA_101_2_1", 101, f"{orf_hc}\t{orf_hc}_alt", hc=True),
            _pin_row("runA_102_2_1", 102, orf_hc, hc=False, peptide="-.KPEPTIDER.-"),
            _pin_row("runA_303_2_1", 303, orf_lc_only, hc=False, peptide="-.MLOWCONF.-"),
            _pin_row("runA_999_2_1", 999, "Rv0001_ANNO", hc=True),
        ]
        pin_path = tmp_path / folder / "Percolator" / "combined_pin.txt"
        # Write raw lines with ragged tab in first data row to exercise ProteinFixer
        df_pin = pd.DataFrame(pin_rows)
        pin_path.write_text(df_pin.to_csv(sep="\t", index=False), encoding="utf-8")

        res04 = pd.DataFrame(
            [
                {
                    "SpecFile": "runA.mzML",
                    "SpecID": "index=1",
                    "ScanNum": 101,
                    "Extended ORF": orf_hc,
                    "Extended Sequence": "MPEPTIDERKKPEPTIDER",
                    "Free Energy": -9.4,
                    "Shine Dalgarno": "Present. Strong binding.",
                    "Extended Coordinates": "10-70",
                    "Genome Coordinates": "10-70",
                    "ORF Sequence": "MPEPTIDERKKPEPTIDER",
                    "Peptide": "-.MPEPTIDERK.-",
                    "Protein": orf_hc,
                },
                {
                    "SpecFile": "runA.mzML",
                    "SpecID": "index=2",
                    "ScanNum": 102,
                    "Extended ORF": orf_hc,
                    "Extended Sequence": "MPEPTIDERKKPEPTIDER",
                    "Free Energy": -9.4,
                    "Shine Dalgarno": "Present. Strong binding.",
                    "Extended Coordinates": "10-70",
                    "Genome Coordinates": "10-70",
                    "ORF Sequence": "MPEPTIDERKKPEPTIDER",
                    "Peptide": "-.KPEPTIDER.-",
                    "Protein": f"{folder}_{prefix}_1b_16-70_forward",
                },
                {
                    "SpecFile": "runA.mzML",
                    "SpecID": "index=3",
                    "ScanNum": 303,
                    "Extended ORF": orf_lc_only,
                    "Extended Sequence": "MLOWCONF",
                    "Free Energy": np.nan,
                    "Shine Dalgarno": np.nan,
                    "Extended Coordinates": "200-260",
                    "Genome Coordinates": "200-260",
                    "ORF Sequence": "MLOWCONF",
                    "Peptide": "-.MLOWCONF.-",
                    "Protein": orf_lc_only,
                },
            ]
        )
        res04.to_csv(tmp_path / folder / "post_perc" / f"{ftype}_results_04.txt", sep="\t", index=False)

        # Pre-validation table (contains both ORFs, including NaN Shine Dalgarno row)
        pre_df = pd.DataFrame(
            [
                {
                    "SpecFile": "runA.mzML",
                    "ORF name": f"{prefix}_1",
                    "ScanNum": 101,
                    "Protein sequence": "MPEPTIDERKKPEPTIDER",
                    "Free Energy": -9.4,
                    "Shine Dalgarno": "Present. Strong binding.",
                    "Genome Coordinates": "10-70",
                    "Peptide": "-.MPEPTIDERK.-",
                },
                {
                    "SpecFile": "runA.mzML",
                    "ORF name": f"{prefix}_2",
                    "ScanNum": 303,
                    "Protein sequence": "MLOWCONF",
                    "Free Energy": np.nan,
                    "Shine Dalgarno": np.nan,
                    "Genome Coordinates": "200-260",
                    "Peptide": "-.MLOWCONF.-",
                },
            ]
        )
        pre_df.to_csv(tmp_path / folder / "Results" / f"{ftype}_pre_validation_results.txt", sep="\t", index=False)

    args = argparse.Namespace(transcriptome=True, model=None)
    vp = ValidatePipeline(args=args)
    vp.validate_genome()
    vp.validate_transcriptome()

    # Confirm PreFiltering actually wrote filtered files (no longer a dead no-op!)
    filtered_files = list((tmp_path / "Genome" / "Percolator" / "for_predicting" / "filtered").glob("filtered_*"))
    assert len(filtered_files) == 1
    filtered_df = pd.read_csv(filtered_files[0], sep="\t")
    assert "Rv0001_ANNO" not in "|".join(filtered_df["Proteins"].astype(str))

    # Post-validation results MUST include both scan 101 (HC) AND rescued sibling scan 102 (Rescued_LC),
    # MUST exclude LC-only ORF scan 303, MUST NOT leak RenamedFiles, and MUST unify ORF name!
    g_post = pd.read_csv(tmp_path / "Genome" / "Results" / "genome_post_validation_results.txt", sep="\t")
    assert set(g_post["ScanNum"]) == {101, 102}
    assert set(g_post["Spectrum Validation"]) == {"HC", "Rescued_LC"}
    assert "RenamedFiles" not in g_post.columns
    assert g_post["ORF name"].nunique() == 1

    # ResultsSummarizer with boolean transcriptome=True MUST emit all 4 subsets and handle NaN Shine Dalgarno!
    summ = ResultsSummarizer(transcriptome=True)
    summ.get_results()
    summ.save()
    summ_df = pd.read_csv(tmp_path / "summarized_results.txt", sep="\t")
    assert list(summ_df["Subset"]) == ["Genome LC", "Genome HC", "Transcriptome LC", "Transcriptome HC"]
    g_hc_row = summ_df[summ_df["Subset"] == "Genome HC"].iloc[0]
    assert int(g_hc_row["ORFs"]) == 1
    assert int(g_hc_row["Unique microprotein sequences"]) == 1
    assert int(g_hc_row["RBS"]) == 1
