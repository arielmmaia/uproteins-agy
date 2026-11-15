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

"""Normalize ragged tab-delimited Proteins columns in Percolator PIN files."""

from __future__ import annotations

import os
import pathlib


class ProteinFixer(object):
    def __init__(self, pin_folder: str | os.PathLike[str]) -> None:
        self.folder = str(pin_folder)
        self.files = sorted(
            str(entry)
            for entry in pathlib.Path(self.folder).iterdir()
            if entry.is_file()
        )

    def fix_files(self, outdir: str | os.PathLike[str]) -> list[pathlib.Path]:
        target_dir = pathlib.Path(outdir) / "for_predicting"
        target_dir.mkdir(parents=True, exist_ok=True)
        written: list[pathlib.Path] = []

        for file in self.files:
            basename = os.path.basename(file)
            lower_name = basename.lower()
            if lower_name.startswith(("fixed_", "filtered_")):
                continue
            if (
                "pin" not in lower_name
                or "uncompeted" in lower_name
                or "pep_results" in lower_name
            ):
                continue

            with open(file, "r", encoding="utf-8") as probe:
                first_line = probe.readline().rstrip("\r\n")
            if not first_line or first_line.split("\t")[0] != "SpecId":
                continue

            out_path = target_dir / f"fixed_{basename}"
            with open(file, "r", encoding="utf-8") as unfixed, open(
                out_path, "w", encoding="utf-8"
            ) as out:
                proteins_col: int | None = None
                for line in unfixed:
                    if "DefaultDirection" in line or line.startswith("Default"):
                        continue
                    stripped = line.rstrip("\r\n")
                    if not stripped:
                        continue
                    cols = stripped.split("\t")
                    if proteins_col is None:
                        proteins_col = len(cols) - 1
                    new_proteins = ",".join(cols[proteins_col:])
                    new_line = "\t".join(cols[:proteins_col]) + "\t" + new_proteins + "\n"
                    out.write(new_line)
            written.append(out_path)

        return written