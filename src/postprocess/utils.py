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

from ..utils import REPO_ROOT, build_mzid_to_tsv_cmd, ensure_dir, run_cmd


class TSVConverter(object):
    def __init__(self, folder, args=None):
        self.path = str(REPO_ROOT)
        self.folder = folder
        self.args = args

    def convert_files(self):
        tsv_dir = ensure_dir(f'{self.folder}/tsv_msgf')
        files = sorted(os.listdir(f'{self.folder}'))
        msgf_jar = getattr(self.args, 'msgf_jar', None)
        for file in files:
            if '.mzid' in file:
                cmd = build_mzid_to_tsv_cmd(
                    f'{self.folder}/{file}',
                    tsv_dir / f'{file}.tsv',
                    msgf_jar=msgf_jar,
                )
                run_cmd(cmd)
        return self
