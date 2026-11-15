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


import os
import pathlib
import shutil

from .utils import REPO_ROOT, build_msgf_search_cmd, ensure_dir, run_cmd


class PeptideSearch(object):
    def __init__(self, database_type, ms_files_folder, orf_file, args, decoy=False):
        self.database_type = database_type
        self.ms_files_folder = pathlib.Path(ms_files_folder).resolve()
        self.orf_file = pathlib.Path(orf_file)
        self.args = args
        self.path = str(REPO_ROOT)
        self.decoy = decoy

    def peptide_identification(self):
        print("\nPerforming peptide search using %s database\n" % self.database_type)
        db_dir = ensure_dir(self.database_type)
        orf_src = self.orf_file.resolve()
        orf_dst = (db_dir / self.orf_file.name).resolve()
        if orf_src != orf_dst:
            shutil.copy2(orf_src, orf_dst)
        ms_db_dst = (self.ms_files_folder / self.orf_file.name).resolve()
        if orf_src != ms_db_dst:
            shutil.copy2(orf_src, ms_db_dst)

        self.loop_search()
        for mzid_file in sorted(self.ms_files_folder.glob("*.mzid")):
            shutil.move(str(mzid_file), str(db_dir / mzid_file.name))

    def loop_search(self):
        files = sorted(
            entry.name
            for entry in self.ms_files_folder.iterdir()
            if entry.is_file() and entry.name.endswith('mzML')
        )
        for file in files:
            self.run_msgf(file)
        return self

    def run_msgf(self, file):
        self._check_folder()
        msgf_args = [
            '-d', str(self.orf_file.resolve()),
            '-tda', '0',
            '-s', str(self.ms_files_folder / file),
            '-addFeatures', '1',
        ]
        if self.decoy:
            msgf_args.extend(['-o', str(self.ms_files_folder / f'{file}_decoy.mzid')])

        excluded = {
            None, "mass_spec", "outdir", "transcriptome", "mode", "msgf_jar",
            "skip_assembly", "skip_db", "skip_ms", "skip_postms",
            "skip_validation", "gtf", "single", "reads1", "reads2",
            "strandness", "gffcompare_path", "gffread_path", "genome",
            "proteome", "minsize", "maxsize", "starts", "stops", "threads",
        }
        for key, value in vars(self.args).items():
            if key not in excluded and value is not None:
                msgf_args.extend([f'-{key}', str(value)])

        cmd = build_msgf_search_cmd(
            msgf_args,
            msgf_jar=getattr(self.args, 'msgf_jar', None),
        )
        run_cmd(cmd)
        return self

    def _check_folder(self):
        ensure_dir('mzid/Genome')
        if getattr(self.args, 'transcriptome', False):
            ensure_dir('mzid/Transcriptome')

