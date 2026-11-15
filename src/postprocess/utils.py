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

from .percolator import classify_mzid_file
from ..utils import REPO_ROOT, build_mzid_to_tsv_cmd, ensure_dir, run_cmd


class TSVConverter(object):
    def __init__(self, folder, args=None):
        self.path = str(REPO_ROOT)
        self.folder = folder
        self.args = args

    def convert_files(self):
        folder_path = pathlib.Path(self.folder)
        tsv_dir = ensure_dir(folder_path / "tsv_msgf")
        msgf_jar = getattr(self.args, "msgf_jar", None)
        java_mem = getattr(self.args, "java_mem", None) or "8G"

        for entry in sorted(folder_path.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or not entry.name.lower().endswith(".mzid"):
                continue
            # Only convert target/concatenated .mzid files to TSV; decoy-only .mzid
            # files contain zero target proteins and double Java conversion cost.
            if classify_mzid_file(entry) == "decoy":
                continue
            cmd = build_mzid_to_tsv_cmd(
                entry,
                tsv_dir / f"{entry.name}.tsv",
                msgf_jar=msgf_jar,
                java_mem=java_mem,
            )
            run_cmd(cmd)
        return self
