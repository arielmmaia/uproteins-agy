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

"""Stage 5 Spectrum Forest validation pipeline across Genome and Transcriptome."""

from __future__ import annotations

import pathlib
import typing as t

import numpy as np

from ..forest import (
    DEFAULT_MODEL_PATH,
    FeatureFishing,
    PreFiltering,
    ProteinFixer,
    SpectralForest,
    SpectralForestEngine,
    SpectrumMiner,
)
from ..forest.predictor import compute_reference_scaler_bounds
from ..postprocess import ResultsWrapper
from ..utils import REPO_ROOT, ensure_dir


def _is_enabled(flag: t.Any) -> bool:
    if isinstance(flag, str):
        return flag.strip().upper() in {"YES", "Y", "TRUE", "1"}
    return bool(flag)


class ValidatePipeline(object):
    def __init__(self, args, testing: bool = False) -> None:
        self.args = args
        self.pypath = str(REPO_ROOT)
        self.testing = testing
        self.model_path = getattr(args, "model", None) or DEFAULT_MODEL_PATH
        self._shared_bounds: t.Optional[tuple[np.ndarray, np.ndarray]] = None
        self._bounds_computed: bool = False

    def _get_experiment_reference_bounds(
        self,
    ) -> t.Optional[tuple[np.ndarray, np.ndarray]]:
        if self._bounds_computed:
            return self._shared_bounds
        self._bounds_computed = True

        engine = SpectralForestEngine.load(self.model_path)
        if engine.scaling != "minmax":
            self._shared_bounds = None
            return None

        pin_candidates: list[pathlib.Path] = []
        folders = ["Genome"]
        if _is_enabled(getattr(self.args, "transcriptome", False)):
            folders.append("Transcriptome")

        for folder in folders:
            perc_dir = pathlib.Path(folder) / "Percolator"
            if not perc_dir.is_dir():
                continue
            fixed_files = ProteinFixer(pin_folder=perc_dir).fix_files(outdir=perc_dir)
            pin_candidates.extend(fixed_files)

        self._shared_bounds = compute_reference_scaler_bounds(
            pin_candidates,
            feature_names=engine.feature_names,
            fallback_bounds=engine.default_scaler_bounds,
        )
        return self._shared_bounds

    def _run_subset(self, folder: str, filetype: str) -> None:
        for_pred_dir = ensure_dir(f"{folder}/Percolator/for_predicting")
        filtered_dir = ensure_dir(f"{folder}/Percolator/for_predicting/filtered")

        fixer = ProteinFixer(pin_folder=f"{folder}/Percolator")
        fixer.fix_files(outdir=f"{folder}/Percolator")

        bounds = self._get_experiment_reference_bounds()

        results_04 = f"{folder}/post_perc/{filetype}_results_04.txt"
        prefiltering = PreFiltering(
            pin_folder=str(for_pred_dir),
            results_04=results_04,
            testing=self.testing,
        )
        prefiltering.filter_proteins()
        prefiltering.filter_pin_files(filtered_dir)

        fishing = FeatureFishing(
            results=results_04,
            pin_folder=str(filtered_dir),
            testing=self.testing,
        )
        fishing.add_features()
        features_path = f"{folder}/Percolator/for_predicting/{filetype}_results_with_features.txt"
        fishing.save_table(features_path)

        forest = SpectralForest(
            model_pickle=self.model_path,
            results=features_path,
            reference_bounds=bounds,
        )
        forest.predict()
        ensure_dir(f"{folder}/predicted")
        predicted_path = f"{folder}/predicted/{filetype}_predicted.txt"
        forest.save(predicted_path)

        mining = SpectrumMiner(
            predicted=predicted_path,
            results=results_04,
            testing=self.testing,
        )
        results_05 = f"{folder}/post_perc/{filetype}_results_05.txt"
        mining.filter_results(results_05)
        rescued_results = mining.add_lc_spectra_for_hc_orf(
            filetype=filetype, folder=folder
        )

        wrapper = ResultsWrapper(
            df=rescued_results,
            folder=folder,
            filetype=filetype,
        )
        wrapper.reformat(pre_validation=False)

    def validate_genome(self) -> None:
        self._run_subset(folder="Genome", filetype="genome")

    def validate_transcriptome(self) -> None:
        if _is_enabled(getattr(self.args, "transcriptome", False)):
            self._run_subset(folder="Transcriptome", filetype="transcriptome")
