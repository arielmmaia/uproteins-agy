# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression and scientific validity tests for Issue #37 (Spectrum Forest predictor & trainer)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.forest.predictor import (
    DEFAULT_MODEL_PATH,
    LEGACY_PICKLE_PATH,
    SPECTRAL_FOREST_V1_FEATURES,
    SPECTRAL_FOREST_V2_FEATURES,
    SpectralForest,
    SpectralForestEngine,
    compute_reference_scaler_bounds,
)
from src.forest.trainer import peptide_group_key, train_spectral_forest
from src.main import uproteins


def _make_hc_row(scan: int = 101, spec_id: str = "run1_101_2_1", protein: str = "Genome_gORF_1") -> dict:
    return {
        "SpecId": spec_id,
        "Label": 1,
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
        "Charge3": 0,
        "enzN": 1,
        "enzC": 1,
        "enzInt": 0,
        "Peptide": "-.MPEPTIDERK.-",
        "Proteins": protein,
    }


def _make_lc_row(scan: int = 202, spec_id: str = "run1_202_2_1", protein: str = "Genome_gORF_2") -> dict:
    return {
        "SpecId": spec_id,
        "Label": 1,
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
        "Charge3": 0,
        "enzN": 1,
        "enzC": 1,
        "enzInt": 1,
        "Peptide": "-.MHVQSSRPRDR.D",
        "Proteins": protein,
    }


@pytest.mark.validate
def test_v1_json_lossless_parity_with_legacy_pickle():
    """Exported JSON forest must match legacy sklearn 0.24.2 model.pkl bit-for-bit across all 200 trees."""
    assert DEFAULT_MODEL_PATH.is_file()
    json_engine = SpectralForestEngine.from_json(DEFAULT_MODEL_PATH)
    pkl_engine = SpectralForestEngine.from_legacy_pickle(LEGACY_PICKLE_PATH)

    assert json_engine.n_estimators == 200
    assert json_engine.n_features == 17
    assert json_engine.feature_names == SPECTRAL_FOREST_V1_FEATURES
    assert sum(len(t["left"]) for t in json_engine.trees) == 45032

    rng = np.random.default_rng(12345)
    X_test = rng.uniform(0.0, 1.0, size=(300, 17))
    X_test[0] = 0.0

    proba_json = json_engine.predict_proba(X_test)
    proba_pkl = pkl_engine.predict_proba(X_test)
    assert np.allclose(proba_json, proba_pkl, atol=1e-15)
    assert proba_json[0, 1] == pytest.approx(0.23783333333333334, abs=1e-12)


@pytest.mark.validate
def test_zero_and_single_row_batches_without_minmax_distortion(tmp_path):
    """0-row and 1-row candidate tables must not crash or collapse to [0.0]*17 -> P(HC)=0.2378."""
    # 0-row candidate file
    empty_df = pd.DataFrame([_make_hc_row()]).iloc[0:0]
    empty_in = tmp_path / "empty_features.txt"
    empty_out = tmp_path / "empty_predicted.txt"
    empty_df.to_csv(empty_in, sep="\t", index=False)

    sf_empty = SpectralForest(model_pickle=str(LEGACY_PICKLE_PATH), results=str(empty_in))
    preds_empty = sf_empty.predict()
    sf_empty.save(empty_out)
    assert len(preds_empty) == 0
    saved_empty = pd.read_csv(empty_out, sep="\t")
    assert len(saved_empty) == 0
    assert "Prediction" in saved_empty.columns
    assert "Prediction_Prob" in saved_empty.columns

    # 1-row high-confidence PSM (legacy MinMaxScaler forced this to [0.0]*17 -> 0!)
    hc_in = tmp_path / "single_hc.txt"
    pd.DataFrame([_make_hc_row()]).to_csv(hc_in, sep="\t", index=False)
    sf_hc = SpectralForest(model_pickle=str(LEGACY_PICKLE_PATH), results=str(hc_in))
    preds_hc = sf_hc.predict()
    assert list(preds_hc) == [1]
    assert len(sf_hc.df) == 1
    assert sf_hc.df.loc[0, "Prediction_Prob"] > 0.85

    # 1-row low-confidence PSM must be rejected
    lc_in = tmp_path / "single_lc.txt"
    pd.DataFrame([_make_lc_row()]).to_csv(lc_in, sep="\t", index=False)
    sf_lc = SpectralForest(model_pickle=str(LEGACY_PICKLE_PATH), results=str(lc_in))
    preds_lc = sf_lc.predict()
    assert list(preds_lc) == [0]
    assert len(sf_lc.df) == 0


@pytest.mark.validate
def test_borf_genome_vs_transcriptome_identical_predictions(tmp_path):
    """Identical MS2 scan (bORF) must receive identical probability in Genome and Transcriptome."""
    borf_scan = _make_hc_row(scan=555, spec_id="run1_555_2_1", protein="Genome_gORF_9")
    borf_rna = dict(borf_scan, Proteins="Transcriptome_tORF_9")

    # Genome batch surrounded by other high-confidence rows; Transcriptome batch surrounded by low-confidence rows
    genome_df = pd.DataFrame([borf_scan, _make_hc_row(scan=556, spec_id="run1_556_2_1")])
    rna_df = pd.DataFrame([borf_rna, _make_lc_row(scan=700, spec_id="run1_700_2_1"), _make_lc_row(scan=701, spec_id="run1_701_2_1")])

    # Shared experiment-wide reference PIN pool (>= 50 target rows)
    rng = np.random.default_rng(7)
    ref_rows = []
    for idx in range(60):
        base = _make_hc_row(scan=1000 + idx, spec_id=f"run1_{1000+idx}_2_1") if idx % 2 == 0 else _make_lc_row(scan=1000 + idx, spec_id=f"run1_{1000+idx}_2_1")
        base["ExpMass"] = float(rng.uniform(420.0, 1450.0))
        base["CalcMass"] = base["ExpMass"]
        base["Mass"] = base["ExpMass"]
        ref_rows.append(base)
    ref_pin = tmp_path / "experiment_ref.pin"
    pd.DataFrame(ref_rows).to_csv(ref_pin, sep="\t", index=False)

    bounds = compute_reference_scaler_bounds([ref_pin])
    assert bounds is not None

    sf_g = SpectralForest(results=genome_df, reference_bounds=bounds)
    sf_t = SpectralForest(results=rna_df, reference_bounds=bounds)
    sf_g.predict()
    sf_t.predict()

    assert sf_g.all_predictions_df is not None and sf_t.all_predictions_df is not None
    p_genome = sf_g.all_predictions_df.loc[0, "Prediction_Prob"]
    p_rna = sf_t.all_predictions_df.loc[0, "Prediction_Prob"]
    assert p_genome == pytest.approx(p_rna, abs=1e-12)


@pytest.mark.validate
def test_missing_sq_mean_error_top7_charge25_and_case_insensitive_decoys():
    """Low-res --inst 0 missing sqMeanErrorTop7, Charge25+, and lowercase decoy_ rows must be handled safely."""
    hc = _make_hc_row(scan=10, spec_id="run1_10_2_1")
    del hc["sqMeanErrorTop7"]
    hc["Charge25"] = 1

    decoy_lower = _make_hc_row(scan=11, spec_id="run1_11_2_1", protein="decoy_Genome_gORF_1")
    del decoy_lower["sqMeanErrorTop7"]
    decoy_lower["Charge25"] = 0

    reverse_prot = _make_hc_row(scan=12, spec_id="run1_12_2_1", protein="Reverse_contaminant_KERATIN")
    del reverse_prot["sqMeanErrorTop7"]
    reverse_prot["Charge25"] = 0

    decoy_label = _make_hc_row(scan=13, spec_id="run1_13_2_1", protein="Genome_gORF_3")
    decoy_label["Label"] = -1
    del decoy_label["sqMeanErrorTop7"]
    decoy_label["Charge25"] = 0

    df = pd.DataFrame([hc, decoy_lower, reverse_prot, decoy_label])
    with pytest.warns(UserWarning, match="sqMeanErrorTop7"):
        sf = SpectralForest(results=df)
    preds = sf.predict()
    assert len(preds) == 1
    assert list(preds) == [1]
    assert list(sf.df["ScanNr"]) == [10]


@pytest.mark.validate
def test_train_validator_cli_unscaled_v2_and_peptide_grouped_cv(tmp_path):
    """uProteInS train-validator must train an unscaled lnSpecEValue model with I==L peptide-grouped CV."""
    assert peptide_group_key("K.MPEPT[UNIMOD:21]IDER.A") == peptide_group_key("R.MPEPTLDER.-")

    rng = np.random.default_rng(99)
    rows = []
    for grp in range(12):
        # Positive peptide group (2 charge states with I/L & modification variation)
        pep_base = f"MPEPT{'I' if grp % 2 == 0 else 'L'}DER" + chr(65 + (grp % 20))
        for chg in (2, 3):
            r = _make_hc_row(scan=100 + grp * 10 + chg, spec_id=f"run1_{100+grp*10+chg}_{chg}_1")
            r["RawScore"] = float(rng.uniform(90.0, 180.0))
            r["DeNovoScore"] = float(rng.uniform(100.0, 190.0))
            r["SpecEValue"] = float(10.0 ** rng.uniform(-16.0, -9.0))
            r["Peptide"] = f"K.{pep_base}.R" if chg == 2 else f"K.{pep_base.replace('I', 'L')}[UNIMOD:4].R"
            r["label_spectra"] = 1
            rows.append(r)

        # Negative peptide group
        neg_pep = f"MGARBA{'I' if grp % 2 == 0 else 'L'}GE" + chr(65 + (grp % 20))
        for chg in (2, 3):
            r = _make_lc_row(scan=500 + grp * 10 + chg, spec_id=f"run1_{500+grp*10+chg}_{chg}_1")
            r["RawScore"] = float(rng.uniform(-50.0, -5.0))
            r["DeNovoScore"] = float(rng.uniform(1.0, 20.0))
            r["SpecEValue"] = float(10.0 ** rng.uniform(-2.0, 1.0))
            r["Peptide"] = f"R.{neg_pep}.K"
            r["label_spectra"] = 0
            rows.append(r)

    train_tsv = tmp_path / "labeled_training_pin.tsv"
    pd.DataFrame(rows).to_csv(train_tsv, sep="\t", index=False)

    out_model = tmp_path / "custom_v2_forest.json"
    uproteins(
        [
            "train-validator",
            "--training_pin",
            str(train_tsv),
            "--output_model",
            str(out_model),
            "--n_estimators",
            "50",
            "--cv_folds",
            "4",
            "--seed",
            "17",
        ]
    )

    assert out_model.is_file()
    metrics_file = out_model.with_suffix(".metrics.json")
    assert metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert metrics["scaling"] == "none"
    assert metrics["cross_validation"]["cv_scheme"] == "StratifiedGroupKFold(peptide_I==L)"
    assert metrics["cross_validation"]["roc_auc"] > 0.95

    engine = SpectralForestEngine.from_json(out_model)
    assert engine.scaling == "none"
    assert engine.feature_names == SPECTRAL_FOREST_V2_FEATURES
    assert "CalcMass" not in engine.feature_names
    assert "Mass" not in engine.feature_names
    assert "lnSpecEValue" in engine.feature_names

    # Evaluate trained v2 model via SpectralForest on 1 positive + 1 negative candidate
    eval_df = pd.DataFrame([rows[0], rows[2]]).drop(columns=["label_spectra"])
    sf_v2 = SpectralForest(model_pickle=str(out_model), results=eval_df)
    preds = sf_v2.predict()
    assert list(preds) == [1, 0]
    assert len(sf_v2.df) == 1
