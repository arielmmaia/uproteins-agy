# Copyright © 2021-2025 Eduardo Vieira de Souza
# Copyright © 2021-2025 Adriana Canedo
# Copyright © 2021-2025 Cristiano Valim Bizarro
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

"""Unscaled Random Forest trainer with peptide-grouped stratified CV for Spectrum Forest."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pathlib
import re
import typing as t
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .predictor import (
    SPECTRAL_FOREST_V2_FEATURES,
    SpectralForestEngine,
    clean_pin_candidates,
    extract_feature_matrix,
)

_FLANK_RE = re.compile(r"^[A-Za-z\-]\.(.+)\.[A-Za-z\-]$")
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|[+\-]?\d+(?:\.\d+)?")
_NON_AA_RE = re.compile(r"[^A-Z]")


def peptide_group_key(raw_peptide: t.Any, fallback_index: int = 0) -> str:
    """Normalize a PIN ``Peptide`` string to an unmodified ``I==L`` stripped sequence for grouped CV."""
    text = str(raw_peptide or "").strip()
    if not text:
        return f"__row_{fallback_index}"
    m = _FLANK_RE.match(text)
    core = m.group(1) if m else text
    core = _MOD_BRACKET_RE.sub("", core).upper()
    core = _NON_AA_RE.sub("", core).replace("I", "L")
    return core if core else f"__row_{fallback_index}"


def resolve_v2_training_features(df: pd.DataFrame) -> list[str]:
    """Select unscaled v2 training features, preferring DB-size-independent ``lnSpecEValue``."""
    features: list[str] = []
    for feat in SPECTRAL_FOREST_V2_FEATURES:
        if feat == "lnSpecEValue":
            if "lnSpecEValue" in df.columns or "SpecEValue" in df.columns:
                features.append("lnSpecEValue")
            elif "lnEValue" in df.columns:
                warnings.warn(
                    "Training table lacks 'lnSpecEValue' and 'SpecEValue'; falling back to "
                    "database-size-dependent 'lnEValue'.",
                    UserWarning,
                    stacklevel=2,
                )
                features.append("lnEValue")
            else:
                raise KeyError(
                    "Training table must include 'lnSpecEValue', 'SpecEValue', or 'lnEValue'."
                )
        else:
            features.append(feat)
    return features


@dataclass
class ValidatorTrainingResult:
    """Artifacts and cross-validation metrics returned by ``train_spectral_forest``."""

    engine: SpectralForestEngine
    model_path: pathlib.Path
    metrics_path: pathlib.Path
    metrics: dict[str, t.Any]


def train_spectral_forest(
    training_pin: t.Union[str, os.PathLike[str], pd.DataFrame],
    output_model: t.Union[str, os.PathLike[str]],
    *,
    label_column: str = "label_spectra",
    n_estimators: int = 200,
    max_depth: t.Optional[int] = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    metrics_output: t.Optional[t.Union[str, os.PathLike[str]]] = None,
) -> ValidatorTrainingResult:
    """Train an unscaled ``SpectralForestEngine`` with peptide-grouped stratified CV."""
    if isinstance(training_pin, pd.DataFrame):
        raw_df = training_pin.copy()
    else:
        raw_df = pd.read_csv(training_pin, sep="\t", low_memory=False)

    df = clean_pin_candidates(raw_df)
    if label_column not in df.columns:
        raise KeyError(
            f"Label column {label_column!r} not found in training PIN columns {list(df.columns)}."
        )

    y_series = pd.to_numeric(df[label_column], errors="coerce")
    valid_mask = y_series.isin([0, 1])
    df = df[valid_mask].reset_index(drop=True)
    y = y_series[valid_mask].to_numpy(dtype=np.int64)

    if len(y) < 4 or len(np.unique(y)) < 2:
        raise ValueError(
            "Training dataset must contain at least 4 labeled rows spanning both classes (0 and 1)."
        )

    feature_names = resolve_v2_training_features(df)
    X = extract_feature_matrix(df, feature_names)

    if "Peptide" in df.columns:
        groups = np.asarray(
            [peptide_group_key(p, idx) for idx, p in enumerate(df["Peptide"])],
            dtype=object,
        )
    else:
        groups = np.asarray([f"__row_{idx}" for idx in range(len(df))], dtype=object)

    # Determine valid number of grouped CV splits without leaking peptides across folds
    pos_groups = len(set(groups[y == 1]))
    neg_groups = len(set(groups[y == 0]))
    effective_folds = min(int(cv_folds), pos_groups, neg_groups)

    cv_metrics: dict[str, t.Any] = {"cv_folds_requested": int(cv_folds), "cv_folds_used": 0}
    if effective_folds >= 2:
        oof_probs = np.zeros(len(y), dtype=np.float64)
        oof_covered = np.zeros(len(y), dtype=bool)
        splitter: t.Any
        if len(set(groups)) < len(y):
            splitter = StratifiedGroupKFold(
                n_splits=effective_folds, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(X, y, groups)
            cv_scheme = "StratifiedGroupKFold(peptide_I==L)"
        else:
            splitter = StratifiedKFold(
                n_splits=effective_folds, shuffle=True, random_state=random_state
            )
            split_iter = splitter.split(X, y)
            cv_scheme = "StratifiedKFold"

        folds_evaluated = 0
        for train_idx, test_idx in split_iter:
            if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
                continue
            # Enforce strict zero-overlap peptide groups when Peptide column exists
            assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
            fold_rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features="sqrt",
                random_state=random_state + folds_evaluated,
                n_jobs=-1,
            )
            fold_rf.fit(X[train_idx], y[train_idx])
            oof_probs[test_idx] = fold_rf.predict_proba(X[test_idx])[:, 1]
            oof_covered[test_idx] = True
            folds_evaluated += 1

        if folds_evaluated >= 2 and np.all(oof_covered) and len(np.unique(y)) == 2:
            oof_pred = (oof_probs > 0.5).astype(np.int64)
            cv_metrics = {
                "cv_scheme": cv_scheme,
                "cv_folds_requested": int(cv_folds),
                "cv_folds_used": folds_evaluated,
                "roc_auc": float(roc_auc_score(y, oof_probs)),
                "pr_auc": float(average_precision_score(y, oof_probs)),
                "accuracy": float(accuracy_score(y, oof_pred)),
                "precision": float(precision_score(y, oof_pred, zero_division=0)),
                "recall": float(recall_score(y, oof_pred, zero_division=0)),
                "f1": float(f1_score(y, oof_pred, zero_division=0)),
            }

    final_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    final_rf.fit(X, y)

    importances = {
        feat: float(imp)
        for feat, imp in zip(feature_names, final_rf.feature_importances_)
    }
    metadata: dict[str, t.Any] = {
        "n_samples": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "n_unique_peptide_groups": int(len(set(groups))),
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "random_state": int(random_state),
        "scaling": "none",
        "feature_importances": importances,
        "cross_validation": cv_metrics,
    }

    engine = SpectralForestEngine.from_sklearn_rf(
        final_rf,
        feature_names=feature_names,
        scaling="none",
        default_scaler_bounds=None,
        schema_version=2,
        metadata=metadata,
    )
    model_path = engine.to_json(output_model)

    if metrics_output is not None:
        metrics_path = pathlib.Path(metrics_output)
    else:
        metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return ValidatorTrainingResult(
        engine=engine,
        model_path=model_path,
        metrics_path=metrics_path,
        metrics=metadata,
    )
