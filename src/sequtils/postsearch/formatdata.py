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
import re

import pandas as pd

from ...utils.psmid import parse_percolator_psmid


STANDARD_MSGF_COLUMNS: list[str] = [
    "SpecFile",
    "SpecID",
    "ScanNum",
    "FragMethod",
    "Precursor",
    "IsotopeError",
    "PrecursorError(ppm)",
    "Charge",
    "Peptide",
    "Protein",
    "DeNovoScore",
    "MSGFScore",
    "SpecEValue",
    "EValue",
]

_PROTEIN_DELIM_RE = re.compile(r"[;\t]+|,(?![^()]*\))")


def _normalize_spec_stem(spec_file: str) -> str:
    name = pathlib.Path(str(spec_file).strip()).name
    stem = pathlib.Path(name).stem
    low = stem.lower()
    if low.endswith("_target"):
        return stem[:-7]
    if low.endswith("_decoy"):
        return stem[:-6]
    return stem


def _clean_novel_target_proteins(protein_field: str) -> str:
    """Filter semicolon/comma-separated MS-GF+ protein hits, stripping case-insensitive
    `decoy_`, `reverse_`, and `contaminant_` entries."""
    if pd.isna(protein_field):
        return ""
    kept: list[str] = []
    for raw_tok in _PROTEIN_DELIM_RE.split(str(protein_field)):
        tok = raw_tok.strip()
        if not tok:
            continue
        base = tok.split("(")[0].strip().lower()
        if base.startswith(("decoy_", "reverse_", "contaminant_")):
            continue
        kept.append(tok)
    return ";".join(kept)


class LinkData(object):
    def __init__(self, cat_msgf, peptide):
        """Join concatenated MS-GF+ TSV records (`cat_msgf`) with FDR-filtered
        Percolator PSMs (`peptide`) on `(spec_stem, ScanNum, Charge)`."""
        raw_cat = pd.read_csv(cat_msgf, sep="\t", low_memory=False)
        if "#SpecFile" in raw_cat.columns:
            raw_cat = raw_cat.rename(columns={"#SpecFile": "SpecFile"})
        elif len(raw_cat.columns) >= 14 and "SpecFile" not in raw_cat.columns:
            renamed = list(raw_cat.columns)
            renamed[:14] = STANDARD_MSGF_COLUMNS
            raw_cat.columns = renamed

        # Keep standard 14 MS-GF+ columns even when newer MzIDToTsv adds QValue/PepQValue
        present_std = [c for c in STANDARD_MSGF_COLUMNS if c in raw_cat.columns]
        self.catDataFrame = raw_cat[present_std].copy()
        if not self.catDataFrame.empty:
            self.catDataFrame = self.catDataFrame[
                self.catDataFrame["ScanNum"].astype(str) != "ScanNum"
            ].copy()
            self.catDataFrame["ScanNum"] = pd.to_numeric(
                self.catDataFrame["ScanNum"], errors="coerce"
            )
            self.catDataFrame = self.catDataFrame[
                self.catDataFrame["ScanNum"].notna()
            ].copy()
            self.catDataFrame["ScanNum"] = self.catDataFrame["ScanNum"].astype("int64")

        self.peptideDataFrame = pd.read_csv(peptide, sep="\t")
        if not self.peptideDataFrame.empty and "PSMId" in self.peptideDataFrame.columns:
            self.peptideDataFrame = self.peptideDataFrame[
                self.peptideDataFrame["PSMId"] != "PSMId"
            ].copy()
        self.pepIds = (
            self.peptideDataFrame["PSMId"].tolist()
            if "PSMId" in self.peptideDataFrame.columns
            else []
        )
        self._parse_peptide_scans()
        self.joinedDataFrame = pd.DataFrame(columns=STANDARD_MSGF_COLUMNS)

    def _parse_peptide_scans(self):
        """Extract `(spec_stem, ScanNum, Charge)` from Percolator `PSMId` using
        `parse_percolator_psmid` (never extracting `SII_<index>` as `ScanNum`!)."""
        stems: list[str] = []
        scans: list[int] = []
        charges: list[int] = []
        for psm in self.pepIds:
            spec_stem, scan_num, charge, _ = parse_percolator_psmid(str(psm))
            stems.append(spec_stem)
            scans.append(int(scan_num))
            charges.append(int(charge))

        if not self.peptideDataFrame.empty or len(self.pepIds) == 0:
            self.peptideDataFrame.insert(0, "spec_stem", stems)
            self.peptideDataFrame.insert(2, "scanNumber", scans)
            self.peptideDataFrame["psm_charge"] = charges

    def filter_msgf(self, output):
        """Filter MS-GF+ TSV rows to exact Percolator `(spec_stem, ScanNum)` PSMs,
        supporting `.mzML`/`.mzml`/`.mzXML`/`.mgf` and case-insensitive decoy removal."""
        if self.catDataFrame.empty or self.peptideDataFrame.empty:
            self.joinedDataFrame = pd.DataFrame(columns=STANDARD_MSGF_COLUMNS)
            self.joinedDataFrame.to_csv(f"{output}.tsv", sep="\t", index=False)
            return self

        df = self.catDataFrame.copy()
        df["Protein"] = df["Protein"].map(_clean_novel_target_proteins)
        df = df[df["Protein"] != ""].copy()

        if df.empty:
            self.joinedDataFrame = pd.DataFrame(columns=STANDARD_MSGF_COLUMNS)
            self.joinedDataFrame.to_csv(f"{output}.tsv", sep="\t", index=False)
            return self

        df["_spec_stem"] = df["SpecFile"].map(_normalize_spec_stem)
        target_pairs = set(
            zip(
                self.peptideDataFrame["spec_stem"].tolist(),
                self.peptideDataFrame["scanNumber"].tolist(),
            )
        )
        pair_mask = [
            (st, int(sc)) in target_pairs
            for st, sc in zip(df["_spec_stem"].tolist(), df["ScanNum"].tolist())
        ]
        matched = df[pair_mask].drop(columns=["_spec_stem"]).drop_duplicates()
        self.joinedDataFrame = matched.reset_index(drop=True)
        self.joinedDataFrame.to_csv(f"{output}.tsv", sep="\t", index=False)
        return self