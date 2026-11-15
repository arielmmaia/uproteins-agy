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

"""Match Spectrum Forest high-confidence PSMs back to novel smORFs and rescue sibling LC PSMs."""

from __future__ import annotations

import os
import pathlib

import pandas as pd

from ..utils.psmid import normalize_pin_spec_id
from .preforest import strip_ms_extension


class SpecMiner(object):
    def __init__(self, results, pin_dir) -> None:
        self.df = pd.read_csv(results, sep="\t", low_memory=False).drop_duplicates()
        self.pinFolder = str(pin_dir)
        self.fixedFileNames = self.df["SpecId"].astype(str).tolist()
        self.scanNums = pd.to_numeric(self.df["ScanNum"], errors="coerce").astype(int).tolist()
        self.extendedORFs = self.df["Extended ORF"].tolist()
        self.extendedSequences = self.df["Extended Sequence"].tolist()

    def mine_spectra(self, output: str | os.PathLike[str]) -> pathlib.Path:
        files = sorted(
            entry.name
            for entry in pathlib.Path(self.pinFolder).iterdir()
            if entry.is_file()
        )
        extended_orfs: list[object] = []
        extended_seqs: list[object] = []
        frames: list[pd.DataFrame] = []
        cols = None

        for i, (scan, target_spec) in enumerate(zip(self.scanNums, self.fixedFileNames)):
            target_stem = normalize_pin_spec_id(str(target_spec))[1]
            for fname in files:
                full_df = pd.read_csv(
                    pathlib.Path(self.pinFolder) / fname, sep="\t", low_memory=False
                )
                if cols is None:
                    cols = full_df.columns
                stems = [
                    normalize_pin_spec_id(str(s))[1]
                    for s in full_df["SpecId"].astype(str)
                ]
                scans = pd.to_numeric(full_df["ScanNr"], errors="coerce")
                mask = [(st == target_stem and sc == scan) for st, sc in zip(stems, scans)]
                intersection = full_df[mask]
                if not intersection.empty:
                    frames.append(intersection)
                    for _ in range(len(intersection)):
                        extended_orfs.append(self.extendedORFs[i])
                        extended_seqs.append(self.extendedSequences[i])

        df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=cols if cols is not None else [])
        )
        insert_orf = min(3, len(df.columns))
        df.insert(insert_orf, "Extended ORF", extended_orfs)
        df.insert(insert_orf + 1, "Extended Sequence", extended_seqs)
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        return out_path


class MinePredicted(object):
    def __init__(self, predicted, results) -> None:
        self.predicted = pd.read_csv(predicted, sep="\t", low_memory=False)
        self.df = pd.read_csv(results, sep="\t", low_memory=False)
        self.hc_pairs: set[tuple[str, int]] = set()
        if not self.predicted.empty and "SpecId" in self.predicted.columns and "ScanNr" in self.predicted.columns:
            for sid, sc in zip(
                self.predicted["SpecId"].astype(str), self.predicted["ScanNr"]
            ):
                try:
                    self.hc_pairs.add((normalize_pin_spec_id(sid)[1], int(float(sc))))
                except (TypeError, ValueError):
                    continue

    def filter(self, output: str | os.PathLike[str]) -> pathlib.Path:
        if self.df.empty or not self.hc_pairs:
            filtered = self.df.iloc[0:0].copy()
        else:
            stems = [strip_ms_extension(f) for f in self.df["SpecFile"].astype(str)]
            scans = pd.to_numeric(self.df["ScanNum"], errors="coerce")
            mask = [
                (pd.notna(sc) and (st, int(sc)) in self.hc_pairs)
                for st, sc in zip(stems, scans)
            ]
            filtered = self.df[mask].drop_duplicates().reset_index(drop=True)
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(out_path, sep="\t", index=False)
        return out_path


class SpectrumMiner(object):
    def __init__(self, results, predicted, testing: bool = False) -> None:
        self.results = pd.read_csv(results, sep="\t", low_memory=False)
        self.testing = testing
        self.predicted = pd.read_csv(predicted, sep="\t", low_memory=False)

        self.result_stems = (
            [strip_ms_extension(f) for f in self.results["SpecFile"].astype(str)]
            if "SpecFile" in self.results.columns
            else []
        )
        self.HCs = self.__get_hc()

    def __get_hc(self) -> dict[str, set[int]]:
        hcs: dict[str, set[int]] = {}
        if (
            self.predicted.empty
            or "SpecId" not in self.predicted.columns
            or "ScanNr" not in self.predicted.columns
        ):
            return hcs

        pred_df = self.predicted
        if "Prediction" in pred_df.columns:
            pred_df = pred_df[pd.to_numeric(pred_df["Prediction"], errors="coerce") == 1]

        for raw_sid, raw_scan in zip(
            pred_df["SpecId"].astype(str), pred_df["ScanNr"]
        ):
            try:
                scan_int = int(float(raw_scan))
            except (TypeError, ValueError):
                continue
            _, stem = normalize_pin_spec_id(raw_sid)
            hcs.setdefault(stem, set()).add(scan_int)
        return hcs

    def _is_hc_row(self, spec_file: object, scan_num: object) -> bool:
        try:
            scan_int = int(float(scan_num))
        except (TypeError, ValueError):
            return False
        stem = strip_ms_extension(str(spec_file))
        return scan_int in self.HCs.get(stem, set())

    def filter_results(self, output: str | os.PathLike[str]) -> pathlib.Path:
        if self.results.empty or not self.HCs:
            df = self.results.iloc[0:0].drop(columns=["RenamedFiles"], errors="ignore")
        else:
            mask = [
                self._is_hc_row(sf, sc)
                for sf, sc in zip(self.results["SpecFile"], self.results["ScanNum"])
            ]
            df = (
                self.results[mask]
                .drop(columns=["RenamedFiles"], errors="ignore")
                .drop_duplicates()
                .reset_index(drop=True)
            )
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, sep="\t", index=False)
        return out_path

    def add_lc_spectra_for_hc_orf(
        self, filetype: str, folder: str | os.PathLike[str]
    ) -> pathlib.Path:
        """Retain all FDR-validated PSMs for every extended smORF supported by >=1 HC spectrum."""
        results_dir = pathlib.Path(folder) / "Results"
        results_dir.mkdir(parents=True, exist_ok=True)

        four_path = pathlib.Path(folder) / "post_perc" / f"{filetype}_results_04.txt"
        five_path = pathlib.Path(folder) / "post_perc" / f"{filetype}_results_05.txt"
        four = pd.read_csv(four_path, sep="\t", low_memory=False).drop(
            columns=["RenamedFiles"], errors="ignore"
        )
        five = pd.read_csv(five_path, sep="\t", low_memory=False).drop(
            columns=["RenamedFiles"], errors="ignore"
        )

        out_path = results_dir / f"{filetype}_results.txt"
        if four.empty or five.empty or "Extended Sequence" not in five.columns:
            empty_out = four.iloc[0:0].copy()
            if "Spectrum Validation" not in empty_out.columns:
                empty_out["Spectrum Validation"] = pd.Series(dtype="object")
            empty_out.to_csv(out_path, sep="\t", index=False)
            return out_path

        hc_sequences = set(five["Extended Sequence"].dropna().astype(str))
        passed = four[four["Extended Sequence"].astype(str).isin(hc_sequences)].copy()
        tiers = [
            "HC" if self._is_hc_row(sf, sc) else "Rescued_LC"
            for sf, sc in zip(passed["SpecFile"], passed["ScanNum"])
        ]
        if "Spectrum Validation" in passed.columns:
            passed["Spectrum Validation"] = tiers
        else:
            passed["Spectrum Validation"] = tiers
        passed = passed.drop_duplicates().reset_index(drop=True)
        passed.to_csv(out_path, sep="\t", index=False)
        return out_path