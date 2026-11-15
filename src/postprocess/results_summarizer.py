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

"""Summarize pre-validation and post-validation microprotein and RBS counts."""

from __future__ import annotations

import os
import pathlib
import typing as t

import pandas as pd


def _transcriptome_enabled(value: t.Any) -> bool:
    if isinstance(value, str):
        return value.strip().upper() in {"YES", "Y", "TRUE", "1"}
    return bool(value)


class ResultsSummarizer:
    def __init__(self, transcriptome: t.Any = False) -> None:
        self.transcriptomeResults = "Transcriptome/Results"
        self.genomeResults = "Genome/Results"
        self.transcriptome = transcriptome

        self.outfile: dict[str, list[t.Any]] = {
            "ORFs": [],
            "Unique microprotein sequences": [],
            "RBS": [],
            "Subset": [],
        }

    def get_results(self) -> "ResultsSummarizer":
        subsets: dict[str, str] = {
            "Genome LC": f"{self.genomeResults}/genome_pre_validation_results.txt",
            "Genome HC": f"{self.genomeResults}/genome_post_validation_results.txt",
        }
        tx_pre = f"{self.transcriptomeResults}/transcriptome_pre_validation_results.txt"
        tx_post = f"{self.transcriptomeResults}/transcriptome_post_validation_results.txt"
        if _transcriptome_enabled(self.transcriptome) and (
            os.path.isfile(tx_pre) or os.path.isfile(tx_post)
        ):
            subsets["Transcriptome LC"] = tx_pre
            subsets["Transcriptome HC"] = tx_post

        for subset_label, filepath in subsets.items():
            if not os.path.isfile(filepath):
                continue
            counter = self.count_orfs(file=filepath)
            self.outfile["ORFs"].append(len(counter["orfs"]))
            self.outfile["Unique microprotein sequences"].append(len(counter["seqs"]))
            self.outfile["RBS"].append(len(counter["shines"]))
            self.outfile["Subset"].append(subset_label)
        return self

    @staticmethod
    def count_orfs(file: str | os.PathLike[str]) -> dict[str, list[str]]:
        counter: dict[str, list[str]] = {"orfs": [], "seqs": [], "shines": []}
        df = pd.read_csv(file, sep="\t", low_memory=False)
        if df.empty:
            return counter

        orfs = df["ORF name"].astype(str).tolist() if "ORF name" in df.columns else []
        seqs = (
            df["Protein sequence"].astype(str).tolist()
            if "Protein sequence" in df.columns
            else []
        )
        shines = (
            df["Shine Dalgarno"].tolist()
            if "Shine Dalgarno" in df.columns
            else [None] * len(orfs)
        )
        coords = (
            df["Genome Coordinates"].astype(str).tolist()
            if "Genome Coordinates" in df.columns
            else [""] * len(orfs)
        )

        seen_loci: set[tuple[str, str]] = set()
        seen_seqs: set[str] = set()
        for i, orf in enumerate(orfs):
            seq = seqs[i] if i < len(seqs) else ""
            coord = coords[i] if i < len(coords) else ""
            locus_key = (coord, seq) if coord else (orf, seq)
            if locus_key not in seen_loci:
                seen_loci.add(locus_key)
                counter["orfs"].append(orf)
                sd_raw = shines[i] if i < len(shines) else None
                sd_text = "" if pd.isna(sd_raw) else str(sd_raw)
                if "Present" in sd_text:
                    counter["shines"].append(sd_text)
            if seq and seq not in seen_seqs:
                seen_seqs.add(seq)
                counter["seqs"].append(seq)
        return counter

    def save(self, output: str | os.PathLike[str] = "summarized_results.txt") -> pathlib.Path:
        out_path = pathlib.Path(output)
        df = pd.DataFrame(data=self.outfile)
        df.to_csv(out_path, sep="\t", index=False)
        return out_path