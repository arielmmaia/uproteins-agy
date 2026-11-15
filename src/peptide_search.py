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
import shutil

from .utils import REPO_ROOT, build_msgf_search_cmd, ensure_dir, run_cmd


SPECTRUM_EXTENSIONS: tuple[str, ...] = (".mzml", ".mzxml", ".mgf")

# Explicit allowlist mapping uProteInS parser attribute names -> MS-GF+ CLI flags
MSGF_PARAM_MAP: tuple[tuple[str, str], ...] = (
    ("t", "-t"),
    ("ti", "-ti"),
    ("thread", "-thread"),
    ("tasks", "-tasks"),
    ("m", "-m"),
    ("inst", "-inst"),
    ("e", "-e"),
    ("protocol", "-protocol"),
    ("ntt", "-ntt"),
    ("mod", "-mod"),
    ("minLength", "-minLength"),
    ("maxLength", "-maxLength"),
    ("minCharge", "-minCharge"),
    ("maxCharge", "-maxCharge"),
    ("n", "-n"),
    ("ccm", "-ccm"),
    ("maxMissedCleavages", "-maxMissedCleavages"),
    ("numMods", "-numMods"),
)


class PeptideSearch(object):
    def __init__(self, database_type, ms_files_folder, orf_file, args, decoy=False):
        self.database_type = database_type
        self.ms_files_folder = pathlib.Path(ms_files_folder).resolve()
        self.orf_file = pathlib.Path(orf_file)
        self.args = args
        self.path = str(REPO_ROOT)
        self.decoy = bool(decoy)
        self._staged_db_path: pathlib.Path | None = None

    def _stage_database(self) -> pathlib.Path:
        """Stage the FASTA inside {database_type}/db_index/ so MS-GF+ suffix-array
        index files (.cseq, .cnlcp, .csarr, .canno) are kept inside --outdir and
        never pollute --Mass_spec."""
        db_dir = ensure_dir(self.database_type)
        index_dir = ensure_dir(db_dir / "db_index")
        orf_src = self.orf_file.resolve()
        staged = (index_dir / self.orf_file.name).resolve()
        if orf_src != staged:
            shutil.copy2(orf_src, staged)
        self._staged_db_path = staged
        return staged

    def discover_spectrum_files(self) -> list[str]:
        """Return sorted spectrum filenames (.mzML, .mzXML, .mgf, case-insensitive)."""
        return sorted(
            entry.name
            for entry in self.ms_files_folder.iterdir()
            if entry.is_file() and entry.name.lower().endswith(SPECTRUM_EXTENSIONS)
        )

    def peptide_identification(self):
        print("\nPerforming peptide search using %s database\n" % self.database_type)
        ensure_dir(self.database_type)
        self._stage_database()
        self.loop_search()
        return self

    def loop_search(self):
        for file in self.discover_spectrum_files():
            self.run_msgf(file)
        return self

    def build_msgf_args(self, file: str) -> list[str]:
        db_dir = ensure_dir(self.database_type)
        staged_db = self._staged_db_path or self._stage_database()
        stem = pathlib.Path(file).stem
        role_suffix = "decoy" if self.decoy else "target"
        out_mzid = (db_dir / f"{stem}_{role_suffix}.mzid").resolve()

        msgf_args = [
            "-d", str(staged_db),
            "-tda", "0",
            "-s", str((self.ms_files_folder / file).resolve()),
            "-o", str(out_mzid),
            "-addFeatures", "1",
        ]

        for attr_name, cli_flag in MSGF_PARAM_MAP:
            val = getattr(self.args, attr_name, None)
            if val is not None:
                msgf_args.extend([cli_flag, str(val)])

        return msgf_args

    def run_msgf(self, file: str):
        self._check_folder()
        msgf_args = self.build_msgf_args(file)
        java_mem = getattr(self.args, "java_mem", None) or "8G"
        cmd = build_msgf_search_cmd(
            msgf_args,
            msgf_jar=getattr(self.args, "msgf_jar", None),
            java_mem=java_mem,
        )
        run_cmd(cmd)
        return self

    def _check_folder(self):
        ensure_dir("mzid/Genome")
        if getattr(self.args, "transcriptome", False):
            ensure_dir("mzid/Transcriptome")
