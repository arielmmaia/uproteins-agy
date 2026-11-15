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

"""Version-independent Spectrum Forest predictor and reference feature scaling."""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import re
import typing as t
import warnings

import numpy as np
import pandas as pd

from ..utils import REPO_ROOT

SPECTRAL_FOREST_V1_FEATURES: list[str] = [
    "ExpMass",
    "CalcMass",
    "RawScore",
    "DeNovoScore",
    "ScoreRatio",
    "Energy",
    "lnEValue",
    "IsotopeError",
    "lnExplainedIonCurrentRatio",
    "lnNTermIonCurrentRatio",
    "lnCTermIonCurrentRatio",
    "lnMS2IonCurrent",
    "Mass",
    "PepLen",
    "dM",
    "absdM",
    "sqMeanErrorTop7",
]

SPECTRAL_FOREST_V2_FEATURES: list[str] = [
    "ExpMass",
    "RawScore",
    "DeNovoScore",
    "ScoreRatio",
    "Energy",
    "lnSpecEValue",
    "IsotopeError",
    "lnExplainedIonCurrentRatio",
    "lnNTermIonCurrentRatio",
    "lnCTermIonCurrentRatio",
    "lnMS2IonCurrent",
    "PepLen",
    "dM",
    "absdM",
    "sqMeanErrorTop7",
]

DEFAULT_MODEL_PATH: pathlib.Path = (
    REPO_ROOT / "src" / "model" / "spectral_forest_v1_trees.json"
)
LEGACY_PICKLE_PATH: pathlib.Path = REPO_ROOT / "src" / "model" / "model.pkl"

_CHARGE_COL_RE = re.compile(r"^Charge\d+$")
_NON_FEATURE_METADATA_COLS = frozenset(
    {
        "SpecFile",
        "SpecId",
        "Label",
        "ScanNr",
        "Peptide",
        "Proteins",
        "proteinIds",
        "enzN",
        "enzC",
        "enzInt",
        "StdevErrorTop7",
        "MeanErrorTop7",
        "Prediction",
        "Prediction_Prob",
        "label_spectra",
    }
)


class _LegacySklearnStub:
    """Minimal state holder to extract legacy scikit-learn 0.24 decision trees safely."""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        self._init_args = args

    def __setstate__(self, state: t.Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _LegacySklearnUnpickler(pickle.Unpickler):
    """Unpickler mapping legacy sklearn classes to pure-Python stubs."""

    def find_class(self, module: str, name: str) -> t.Any:
        if module.startswith("sklearn"):
            return type(name, (_LegacySklearnStub,), {"__module__": module})
        return super().find_class(module, name)


def strip_non_feature_columns(columns: t.Iterable[str]) -> list[str]:
    """Remove PIN metadata and arbitrary ``Charge<N>`` columns."""
    kept: list[str] = []
    for col in columns:
        if col in _NON_FEATURE_METADATA_COLS or _CHARGE_COL_RE.match(str(col)):
            continue
        kept.append(col)
    return kept


def clean_pin_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Filter embedded PIN header rows, ``Label == -1``, and decoy/reverse entries."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "ExpMass" in out.columns:
        out = out[out["ExpMass"].astype(str) != "ExpMass"]
    if "SpecId" in out.columns:
        spec_id_lower = out["SpecId"].fillna("").astype(str).str.lower()
        out = out[spec_id_lower != "defaultdirection"]
        out = out[
            ~(
                spec_id_lower.str.startswith("decoy_")
                | spec_id_lower.str.startswith("reverse_")
            )
        ]
    if "SpecFile" in out.columns:
        out = out[out["SpecFile"].fillna("").astype(str) != "DefaultDirection"]
    if "Label" in out.columns:
        labels = pd.to_numeric(out["Label"], errors="coerce")
        out = out[labels != -1]
    if "Proteins" in out.columns:
        prot_lower = out["Proteins"].fillna("").astype(str).str.lower()
        is_decoy = prot_lower.str.contains("decoy", regex=False) | prot_lower.str.contains(
            "reverse_", regex=False
        )
        out = out[~is_decoy]

    return out.reset_index(drop=True)


def extract_feature_matrix(
    df: pd.DataFrame,
    feature_names: t.Sequence[str],
    *,
    warn_missing_top7: bool = True,
) -> np.ndarray:
    """Extract an ``(n_rows, n_features)`` float64 matrix matching ``feature_names``."""
    n_rows = len(df)
    n_features = len(feature_names)
    if n_rows == 0:
        return np.empty((0, n_features), dtype=np.float64)

    cols: list[np.ndarray] = []
    for feat in feature_names:
        if feat in df.columns:
            series = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)
            arr = series.to_numpy(dtype=np.float64, copy=True)
        elif feat == "sqMeanErrorTop7":
            if warn_missing_top7:
                warnings.warn(
                    "Column 'sqMeanErrorTop7' is missing (MS-GF+ low-resolution --inst 0 mode); "
                    "imputing neutral 0.0.",
                    UserWarning,
                    stacklevel=2,
                )
            arr = np.zeros(n_rows, dtype=np.float64)
        elif feat == "CalcMass" and ("Mass" in df.columns or "ExpMass" in df.columns):
            fallback_col = "Mass" if "Mass" in df.columns else "ExpMass"
            arr = (
                pd.to_numeric(df[fallback_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64, copy=True)
            )
        elif feat == "Mass" and ("CalcMass" in df.columns or "ExpMass" in df.columns):
            fallback_col = "CalcMass" if "CalcMass" in df.columns else "ExpMass"
            arr = (
                pd.to_numeric(df[fallback_col], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=np.float64, copy=True)
            )
        elif feat == "lnSpecEValue" and "SpecEValue" in df.columns:
            spec_eval = pd.to_numeric(df["SpecEValue"], errors="coerce").fillna(1.0)
            arr = -np.log(np.clip(spec_eval.to_numpy(dtype=np.float64), 1e-300, None))
        else:
            raise KeyError(
                f"Required Spectrum Forest feature '{feat}' is missing from input columns "
                f"{list(df.columns)}."
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(arr)

    return np.column_stack(cols) if cols else np.empty((n_rows, 0), dtype=np.float64)


def compute_reference_scaler_bounds(
    pin_files: t.Sequence[t.Union[str, os.PathLike[str]]],
    feature_names: t.Sequence[str] = SPECTRAL_FOREST_V1_FEATURES,
    *,
    min_target_psms: int = 50,
    fallback_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
    chunksize: int = 100_000,
) -> t.Optional[tuple[np.ndarray, np.ndarray]]:
    """Compute experiment-wide ``(min, max)`` feature bounds across target PIN PSMs."""
    global_min: t.Optional[np.ndarray] = None
    global_max: t.Optional[np.ndarray] = None
    total_targets = 0

    for pin_path in pin_files:
        path = pathlib.Path(pin_path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, low_memory=False):
            cleaned = clean_pin_candidates(chunk)
            if "Label" in cleaned.columns:
                labels = pd.to_numeric(cleaned["Label"], errors="coerce")
                cleaned = cleaned[labels == 1]
            if cleaned.empty:
                continue
            mat = extract_feature_matrix(
                cleaned, feature_names, warn_missing_top7=False
            )
            if mat.shape[0] == 0:
                continue
            total_targets += mat.shape[0]
            c_min = mat.min(axis=0)
            c_max = mat.max(axis=0)
            if global_min is None:
                global_min = c_min
                global_max = c_max
            else:
                global_min = np.minimum(global_min, c_min)
                assert global_max is not None
                global_max = np.maximum(global_max, c_max)

    if global_min is not None and global_max is not None and total_targets >= min_target_psms:
        return global_min, global_max
    return fallback_bounds


class SpectralForestEngine:
    """Pure-NumPy vectorized decision-forest evaluator independent of scikit-learn versions."""

    def __init__(
        self,
        trees: list[dict[str, np.ndarray]],
        feature_names: t.Sequence[str],
        *,
        scaling: str = "none",
        default_scaler_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
        schema_version: int = 2,
        metadata: t.Optional[dict[str, t.Any]] = None,
    ) -> None:
        if scaling not in ("none", "minmax"):
            raise ValueError(f"Unsupported scaling mode: {scaling!r}")
        self.trees = trees
        self.feature_names = list(feature_names)
        self.scaling = scaling
        self.default_scaler_bounds = default_scaler_bounds
        self.schema_version = schema_version
        self.metadata = metadata or {}

    @property
    def n_estimators(self) -> int:
        return len(self.trees)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ``[P(nHC), P(HC)]`` probabilities with ``float32`` split semantics."""
        X_arr = np.asarray(X)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected 2D array with {self.n_features} features, got shape {X_arr.shape}."
            )
        n_samples = X_arr.shape[0]
        if n_samples == 0:
            return np.empty((0, 2), dtype=np.float64)

        X_f32 = X_arr.astype(np.float32, copy=False)
        prob_hc = np.zeros(n_samples, dtype=np.float64)

        for tree in self.trees:
            left = tree["left"]
            right = tree["right"]
            feat = tree["feature"]
            thr = tree["threshold"]
            leaf_p1 = tree["leaf_p1"]

            nodes = np.zeros(n_samples, dtype=np.int64)
            active = left[nodes] != -1
            while np.any(active):
                idx = np.flatnonzero(active)
                nd = nodes[idx]
                go_left = X_f32[idx, feat[nd]] <= thr[nd]
                nodes[idx] = np.where(go_left, left[nd], right[nd])
                active = left[nodes] != -1
            prob_hc += leaf_p1[nodes]

        prob_hc /= max(len(self.trees), 1)
        return np.column_stack([1.0 - prob_hc, prob_hc])

    def predict(self, X: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
        """Predict binary ``0`` / ``1`` labels (matching ``np.argmax`` at ``threshold=0.5``)."""
        proba = self.predict_proba(X)
        if proba.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if threshold == 0.5:
            return (proba[:, 1] > 0.5).astype(np.int64)
        return (proba[:, 1] >= threshold).astype(np.int64)

    def to_dict(self) -> dict[str, t.Any]:
        """Serialize engine into a portable JSON-compatible dictionary."""
        bounds_payload = None
        if self.default_scaler_bounds is not None:
            bounds_payload = {
                "min": [float(v) for v in self.default_scaler_bounds[0]],
                "max": [float(v) for v in self.default_scaler_bounds[1]],
            }
        serialized_trees = []
        for tree in self.trees:
            entry: dict[str, t.Any] = {
                "left": tree["left"].astype(int).tolist(),
                "right": tree["right"].astype(int).tolist(),
                "feature": tree["feature"].astype(int).tolist(),
                "threshold": [float(v) for v in tree["threshold"]],
            }
            if "values" in tree:
                entry["values"] = [
                    [float(c0), float(c1)] for c0, c1 in tree["values"]
                ]
            else:
                entry["leaf_p1"] = [float(v) for v in tree["leaf_p1"]]
            serialized_trees.append(entry)

        return {
            "schema_version": self.schema_version,
            "scaling": self.scaling,
            "feature_names": self.feature_names,
            "default_scaler_bounds": bounds_payload,
            "metadata": self.metadata,
            "trees": serialized_trees,
        }

    def to_json(self, path: t.Union[str, os.PathLike[str]]) -> pathlib.Path:
        """Write compact JSON model file."""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))
        return out_path

    @classmethod
    def from_dict(cls, payload: dict[str, t.Any]) -> "SpectralForestEngine":
        """Reconstruct ``SpectralForestEngine`` from parsed JSON payload."""
        bounds_raw = payload.get("default_scaler_bounds")
        bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None
        if bounds_raw is not None:
            bounds = (
                np.asarray(bounds_raw["min"], dtype=np.float64),
                np.asarray(bounds_raw["max"], dtype=np.float64),
            )

        trees: list[dict[str, np.ndarray]] = []
        for raw_tree in payload["trees"]:
            left = np.asarray(raw_tree["left"], dtype=np.int64)
            right = np.asarray(raw_tree["right"], dtype=np.int64)
            feat = np.asarray(raw_tree["feature"], dtype=np.int64)
            thr = np.asarray(raw_tree["threshold"], dtype=np.float64)
            if "values" in raw_tree:
                vals = np.asarray(raw_tree["values"], dtype=np.float64)
                denom = vals.sum(axis=1)
                leaf_p1 = np.where(denom > 0.0, vals[:, 1] / denom, 0.0)
                trees.append(
                    {
                        "left": left,
                        "right": right,
                        "feature": feat,
                        "threshold": thr,
                        "values": vals,
                        "leaf_p1": leaf_p1,
                    }
                )
            else:
                leaf_p1 = np.asarray(raw_tree["leaf_p1"], dtype=np.float64)
                trees.append(
                    {
                        "left": left,
                        "right": right,
                        "feature": feat,
                        "threshold": thr,
                        "leaf_p1": leaf_p1,
                    }
                )

        return cls(
            trees=trees,
            feature_names=payload["feature_names"],
            scaling=payload.get("scaling", "none"),
            default_scaler_bounds=bounds,
            schema_version=int(payload.get("schema_version", 2)),
            metadata=payload.get("metadata"),
        )

    @classmethod
    def from_json(cls, path: t.Union[str, os.PathLike[str]]) -> "SpectralForestEngine":
        """Load portable JSON model file."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_sklearn_rf(
        cls,
        rf: t.Any,
        feature_names: t.Sequence[str],
        *,
        scaling: str = "none",
        default_scaler_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
        schema_version: int = 2,
        metadata: t.Optional[dict[str, t.Any]] = None,
    ) -> "SpectralForestEngine":
        """Convert a fitted modern ``scikit-learn`` ``RandomForestClassifier`` to ``SpectralForestEngine``."""
        classes = list(getattr(rf, "classes_", [0, 1]))
        pos_idx = classes.index(1) if 1 in classes else len(classes) - 1

        trees: list[dict[str, np.ndarray]] = []
        for est in rf.estimators_:
            t_obj = est.tree_
            left = np.asarray(t_obj.children_left, dtype=np.int64)
            right = np.asarray(t_obj.children_right, dtype=np.int64)
            feat = np.asarray(t_obj.feature, dtype=np.int64)
            thr = np.asarray(t_obj.threshold, dtype=np.float64)
            raw_vals = np.asarray(t_obj.value, dtype=np.float64)[:, 0, :]
            if raw_vals.shape[1] == 2 and pos_idx == 1:
                vals = raw_vals
            else:
                p1_counts = raw_vals[:, pos_idx]
                p0_counts = raw_vals.sum(axis=1) - p1_counts
                vals = np.column_stack([p0_counts, p1_counts])
            denom = vals.sum(axis=1)
            leaf_p1 = np.where(denom > 0.0, vals[:, 1] / denom, 0.0)
            trees.append(
                {
                    "left": left,
                    "right": right,
                    "feature": feat,
                    "threshold": thr,
                    "values": vals,
                    "leaf_p1": leaf_p1,
                }
            )

        return cls(
            trees=trees,
            feature_names=feature_names,
            scaling=scaling,
            default_scaler_bounds=default_scaler_bounds,
            schema_version=schema_version,
            metadata=metadata,
        )

    @classmethod
    def from_legacy_pickle(
        cls,
        pickle_path: t.Union[str, os.PathLike[str]],
        feature_names: t.Sequence[str] = SPECTRAL_FOREST_V1_FEATURES,
        *,
        scaling: str = "minmax",
        default_scaler_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
        metadata: t.Optional[dict[str, t.Any]] = None,
    ) -> "SpectralForestEngine":
        """Losslessly extract trees from legacy scikit-learn 0.24 pickle without sklearn version errors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with open(pickle_path, "rb") as fh:
                rf = _LegacySklearnUnpickler(fh).load()

        trees: list[dict[str, np.ndarray]] = []
        for est in rf.estimators_:
            t_obj = est.tree_
            nodes = t_obj.nodes
            vals = np.asarray(t_obj.values[:, 0, :], dtype=np.float64)
            denom = vals.sum(axis=1)
            leaf_p1 = np.where(denom > 0.0, vals[:, 1] / denom, 0.0)
            trees.append(
                {
                    "left": np.asarray(nodes["left_child"], dtype=np.int64),
                    "right": np.asarray(nodes["right_child"], dtype=np.int64),
                    "feature": np.asarray(nodes["feature"], dtype=np.int64),
                    "threshold": np.asarray(nodes["threshold"], dtype=np.float64),
                    "values": vals,
                    "leaf_p1": leaf_p1,
                }
            )

        return cls(
            trees=trees,
            feature_names=feature_names,
            scaling=scaling,
            default_scaler_bounds=default_scaler_bounds,
            schema_version=1,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls, model_path: t.Optional[t.Union[str, os.PathLike[str]]] = None
    ) -> "SpectralForestEngine":
        """Load a JSON or legacy pickle model, preferring ``spectral_forest_v1_trees.json`` for ``model.pkl``."""
        target = pathlib.Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if target.suffix.lower() == ".json":
            return cls.from_json(target)
        if target.name == "model.pkl":
            sibling_json = target.with_name("spectral_forest_v1_trees.json")
            if sibling_json.is_file():
                return cls.from_json(sibling_json)
        return cls.from_legacy_pickle(target)


class SpectralForest(object):
    """Spectrum Forest MS2 validator with deterministic scaling and portable tree inference."""

    def __init__(
        self,
        model_pickle: t.Optional[t.Union[str, os.PathLike[str]]] = None,
        results: t.Union[str, os.PathLike[str], pd.DataFrame] = "",
        *,
        reference_pin_files: t.Optional[
            t.Sequence[t.Union[str, os.PathLike[str]]]
        ] = None,
        reference_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
        probability_threshold: float = 0.5,
    ) -> None:
        self.engine = SpectralForestEngine.load(model_pickle)
        self.probability_threshold = float(probability_threshold)

        if isinstance(results, pd.DataFrame):
            raw_df = results.copy()
        else:
            raw_df = pd.read_csv(results, sep="\t", low_memory=False)

        self.df = clean_pin_candidates(raw_df)
        self.properties = list(self.engine.feature_names)
        self.raw_x = extract_feature_matrix(self.df, self.properties)
        self.scaler_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None
        self.x = self._scale_data(
            reference_pin_files=reference_pin_files,
            reference_bounds=reference_bounds,
        )
        self.all_predictions_df: t.Optional[pd.DataFrame] = None

    def _scale_data(
        self,
        *,
        reference_pin_files: t.Optional[
            t.Sequence[t.Union[str, os.PathLike[str]]]
        ] = None,
        reference_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        if self.raw_x.shape[0] == 0 or self.engine.scaling == "none":
            return self.raw_x.copy()

        bounds = reference_bounds
        if bounds is None and reference_pin_files:
            bounds = compute_reference_scaler_bounds(
                reference_pin_files,
                feature_names=self.properties,
                fallback_bounds=self.engine.default_scaler_bounds,
            )
        if bounds is None:
            bounds = self.engine.default_scaler_bounds

        if bounds is not None:
            fmin = np.asarray(bounds[0], dtype=np.float64)
            fmax = np.asarray(bounds[1], dtype=np.float64)
            self.scaler_bounds = (fmin, fmax)
            span = np.where(fmax > fmin, fmax - fmin, 1.0)
            scaled = np.where(fmax > fmin, (self.raw_x - fmin) / span, 0.5)
            return np.clip(scaled, 0.0, 1.0)

        # Fallback only if an external pickle has neither default_scaler_bounds nor reference bounds
        fmin = self.raw_x.min(axis=0)
        fmax = self.raw_x.max(axis=0)
        self.scaler_bounds = (fmin, fmax)
        span = np.where(fmax > fmin, fmax - fmin, 1.0)
        scaled = np.where(fmax > fmin, (self.raw_x - fmin) / span, 0.5)
        return np.clip(scaled, 0.0, 1.0)

    def predict_proba(self) -> np.ndarray:
        """Return ``(n_samples, 2)`` class probability matrix ``[P(nHC), P(HC)]``."""
        return self.engine.predict_proba(self.x)

    def predict(self) -> np.ndarray:
        """Score candidate PSMs, annotate ``Prediction`` & ``Prediction_Prob``, and retain ``Prediction == 1``."""
        proba = self.predict_proba()
        if proba.shape[0] == 0:
            predicted = np.empty(0, dtype=np.int64)
            probs_hc = np.empty(0, dtype=np.float64)
        else:
            probs_hc = proba[:, 1]
            if self.probability_threshold == 0.5:
                predicted = (probs_hc > 0.5).astype(np.int64)
            else:
                predicted = (probs_hc >= self.probability_threshold).astype(np.int64)

        self._insert_predicted(predicted, probs_hc)
        return predicted

    def _insert_predicted(
        self, predicted: np.ndarray, probs_hc: np.ndarray
    ) -> "SpectralForest":
        scored = self.df.copy()
        insert_pos = min(2, len(scored.columns))
        if "Prediction" in scored.columns:
            scored = scored.drop(columns=["Prediction"])
        if "Prediction_Prob" in scored.columns:
            scored = scored.drop(columns=["Prediction_Prob"])
        scored.insert(insert_pos, "Prediction", predicted)
        scored.insert(insert_pos + 1, "Prediction_Prob", np.round(probs_hc, 6))
        self.all_predictions_df = scored
        self.df = scored[scored["Prediction"] == 1].reset_index(drop=True)
        return self

    def save(self, output: t.Union[str, os.PathLike[str]]) -> pathlib.Path:
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out_path, sep="\t", index=False)
        return out_path
