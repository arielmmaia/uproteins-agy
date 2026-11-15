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

"""Pre-filtering and exact (run_stem, ScanNr) feature extraction for Spectrum Forest."""

from __future__ import annotations

import os
import pathlib
import re

import pandas as pd

from ..utils.psmid import normalize_pin_spec_id
from .predictor import clean_pin_candidates

_MS_EXT_RE = re.compile(r"\.(?:mzml|mzxml|mgf|mzid|raw)$", re.IGNORECASE)
_PROT_SPLIT_RE = re.compile(r"[;\t]+|,(?![^()]*\))")


def strip_ms_extension(filename: str) -> str:
    """Strip common MS spectrum/identification extensions without hardcoding 5-char ``.mzML``."""
    base = os.path.basename(str(filename).strip())
    stripped = _MS_EXT_RE.sub("", base)
    if stripped != base:
        return stripped
    stem, ext = os.path.splitext(base)
    return stem if ext else base


def _clean_protein_token(raw_token: str) -> str:
    tok = raw_token.strip()
    if "(pre" in tok:
        tok = tok[: tok.rfind("(pre")]
    return tok.strip()


class PreFiltering(object):
    def __init__(self, pin_folder, results_04, testing: bool = False) -> None:
        self.pinFolder = str(pin_folder)
        self.pinFiles = sorted(
            entry.name
            for entry in pathlib.Path(self.pinFolder).iterdir()
            if entry.is_file()
        )
        self.testing = testing
        self.results = pd.read_csv(results_04, sep="\t", low_memory=False)
        self.__check_testing()
        self.percolatorProteins = (
            self.results["Protein"].dropna().astype(str).tolist()
            if "Protein" in self.results.columns
            else []
        )
        self.proteins: list[str] = []

    def __check_testing(self) -> None:
        if self.testing:
            self.results = self.results.head(20)

    def filter_proteins(self) -> list[str]:
        seen: set[str] = set()
        for raw_entry in self.percolatorProteins:
            for part in _PROT_SPLIT_RE.split(str(raw_entry)):
                tok = _clean_protein_token(part)
                if tok and tok not in seen:
                    seen.add(tok)
                    self.proteins.append(tok)
        return self.proteins

    def filter_pin_files(self, outdir: str | os.PathLike[str]) -> list[pathlib.Path]:
        out_path = pathlib.Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        written: list[pathlib.Path] = []

        escaped_pattern = (
            "|".join(re.escape(p) for p in self.proteins) if self.proteins else ""
        )

        for file in self.pinFiles:
            lower_file = file.lower()
            if "pin" not in lower_file and "pep_results" not in lower_file:
                continue
            src_file = pathlib.Path(self.pinFolder) / file
            if not src_file.is_file():
                continue

            df = pd.read_csv(src_file, sep="\t", low_memory=False)
            df = clean_pin_candidates(df)
            if "Label" in df.columns:
                df = df[pd.to_numeric(df["Label"], errors="coerce") == 1]

            if escaped_pattern and not df.empty and "Proteins" in df.columns:
                mask = df["Proteins"].fillna("").astype(str).str.contains(
                    escaped_pattern, regex=True, na=False
                )
                df = df[mask].reset_index(drop=True)
            else:
                df = df.iloc[0:0]

            dest = out_path / f"filtered_{file}"
            df.to_csv(dest, sep="\t", index=False)
            written.append(dest)

        return written


class FeatureFishing(object):
    def __init__(self, results, pin_folder, testing: bool = False) -> None:
        self.results = pd.read_csv(results, sep="\t", low_memory=False)
        self.testing = testing
        self.__check_testing()
        self.specFiles = (
            self.results["SpecFile"].astype(str).tolist()
            if "SpecFile" in self.results.columns
            else []
        )
        self.scanNum = (
            pd.to_numeric(self.results["ScanNum"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
            if "ScanNum" in self.results.columns
            else []
        )
        self.pinFolder = str(pin_folder)
        self.pinFiles = sorted(
            entry.name
            for entry in pathlib.Path(self.pinFolder).iterdir()
            if entry.is_file()
        )
        self.dataWithFeatures = pd.DataFrame()

    def __check_testing(self) -> None:
        if self.testing:
            return

    def add_features(self) -> "FeatureFishing":
        target_pairs: set[tuple[str, int]] = set()
        if "SpecFile" in self.results.columns and "ScanNum" in self.results.columns:
            for spec, raw_scan in zip(
                self.results["SpecFile"].astype(str), self.results["ScanNum"]
            ):
                try:
                    scan_int = int(float(raw_scan))
                except (TypeError, ValueError):
                    continue
                target_pairs.add((strip_ms_extension(spec), scan_int))

        scan_set = {scan for _, scan in target_pairs}
        frames: list[pd.DataFrame] = []
        cols = None

        for file in self.pinFiles:
            lower_file = file.lower()
            if "pin" not in lower_file and "pep_results" not in lower_file:
                continue
            pin_path = pathlib.Path(self.pinFolder) / file
            if not pin_path.is_file():
                continue

            chunk = pd.read_csv(pin_path, sep="\t", low_memory=False)
            if cols is None:
                cols = chunk.columns
            chunk = clean_pin_candidates(chunk)
            if "Label" in chunk.columns:
                chunk = chunk[pd.to_numeric(chunk["Label"], errors="coerce") == 1]
            if chunk.empty or not target_pairs:
                continue

            scan_series = pd.to_numeric(chunk["ScanNr"], errors="coerce")
            chunk = chunk[scan_series.isin(scan_set)].copy()
            if chunk.empty:
                continue

            chunk_scans = pd.to_numeric(chunk["ScanNr"], errors="coerce").astype(int)
            chunk_stems = [
                normalize_pin_spec_id(str(sid))[1]
                for sid in chunk["SpecId"].astype(str)
            ]
            keep_mask = [
                (stem, sc) in target_pairs
                for stem, sc in zip(chunk_stems, chunk_scans)
            ]
            matched = chunk[keep_mask]
            if not matched.empty:
                frames.append(matched)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if "SpecId" in combined.columns:
                combined = combined.drop_duplicates(subset=["SpecId"], keep="first")
            self.dataWithFeatures = combined.reset_index(drop=True)
        else:
            self.dataWithFeatures = pd.DataFrame(columns=cols if cols is not None else [])
        return self

    def save_table(self, output: str | os.PathLike[str]) -> "FeatureFishing":
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataWithFeatures.to_csv(out_path, sep="\t", index=False)
        return self
