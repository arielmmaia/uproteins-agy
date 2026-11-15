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


import pathlib
import pandas as pd

from ...utils.psmid import parse_percolator_psmid


def _normalize_spec_stem(spec_file: str) -> str:
    name = pathlib.Path(str(spec_file).strip()).name
    stem = pathlib.Path(name).stem
    low = stem.lower()
    if low.endswith("_target"):
        return stem[:-7]
    if low.endswith("_decoy"):
        return stem[:-6]
    return stem


class TSVChunks(object):
    def __init__(self, folder, filetype):
        self.folder = folder
        self.filetype = filetype
        self.percDir = f"{self.folder}/post_perc"
        self.df = pd.read_csv(f"{self.percDir}/{self.filetype}_utps.txt", sep="\t")
        if not self.df.empty and "PSMId" in self.df.columns:
            self.df = self.df[self.df["PSMId"] != "PSMId"].copy()
        self.psm_keys: set[tuple[str, int]] = set()
        self.scans: set[int] = self._extract_scans_and_keys()

    def _extract_scans_and_keys(self) -> set[int]:
        scans: set[int] = set()
        if self.df.empty or "PSMId" not in self.df.columns:
            return scans
        for raw_id in self.df["PSMId"].tolist():
            try:
                spec_stem, scan_num, _, _ = parse_percolator_psmid(str(raw_id))
                scans.add(int(scan_num))
                self.psm_keys.add((spec_stem, int(scan_num)))
            except ValueError:
                parts = str(raw_id).split("_")
                if len(parts) >= 3 and parts[-3].isdigit():
                    scans.add(int(parts[-3]))
        return scans

    def _filter_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        if chunk.empty or not self.scans:
            return chunk.iloc[0:0].copy()

        spec_col = "#SpecFile" if "#SpecFile" in chunk.columns else "SpecFile"
        # Strip repeated mid-file headers created when concatenating multiple TSVs
        clean = chunk[chunk["ScanNum"].astype(str) != "ScanNum"].copy()
        if clean.empty:
            return clean

        scan_series = pd.to_numeric(clean["ScanNum"], errors="coerce")
        clean = clean[scan_series.notna()].copy()
        if clean.empty:
            return clean

        clean["ScanNum"] = scan_series.loc[clean.index].astype("int64")
        clean = clean[clean["ScanNum"].isin(self.scans)].copy()
        if clean.empty or not self.psm_keys or spec_col not in clean.columns:
            return clean

        stems = clean[spec_col].map(_normalize_spec_stem)
        pair_mask = [
            (st, int(sc)) in self.psm_keys
            for st, sc in zip(stems, clean["ScanNum"].tolist())
        ]
        return clean[pair_mask].copy()

    def filter_search(self, chunk_size: int = 10**6):
        search_path = pathlib.Path(f"{self.percDir}/{self.filetype}_search.tsv")
        out_path = pathlib.Path(f"{self.percDir}/{self.filetype}_chunk_search.tsv")

        if not search_path.is_file() or search_path.stat().st_size == 0:
            pd.DataFrame(
                columns=[
                    "#SpecFile", "SpecID", "ScanNum", "FragMethod", "Precursor",
                    "IsotopeError", "PrecursorError(ppm)", "Charge", "Peptide",
                    "Protein", "DeNovoScore", "MSGFScore", "SpecEValue", "EValue",
                ]
            ).to_csv(out_path, sep="\t", index=False)
            return self

        frames: list[pd.DataFrame] = []
        columns: pd.Index | None = None
        for chunk in pd.read_csv(
            search_path, sep="\t", chunksize=chunk_size, low_memory=False
        ):
            if columns is None:
                columns = chunk.columns
            filtered = self._filter_chunk(chunk)
            if not filtered.empty:
                frames.append(filtered)

        if frames:
            result = pd.concat(frames, ignore_index=True)
        else:
            result = pd.DataFrame(columns=columns if columns is not None else [])
        result.to_csv(out_path, sep="\t", index=False)
        return self
