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

from ..utils import ensure_dir, resolve_executable, run_cmd


class PercolatorProcessing(object):
    def __init__(self, folder, filetype, args=None):
        self.folder = folder
        self.args = args
        self.target = None
        self.decoy = None
        self.filetype = filetype

    def create_metafiles(self):
        """ Creates a metafile containing the paths to the mzid files, created during the peptide search step. """
        files = sorted(os.listdir(self.folder))
        real = []
        decoy = []
        for file in files:
            if "mzid" in file:
                if "decoy" in file:
                    decoy.append(os.path.abspath(f"{self.folder}/{file}") + "\n")
                else:
                    real.append(os.path.abspath(f"{self.folder}/{file}") + "\n")
        with open(f"{self.folder}/{self.folder}_target_metafile.txt", 'w') as out:
            out.writelines(real)
        with open(f"{self.folder}/{self.folder}_decoy_metafile.txt", 'w') as decoy_out:
            decoy_out.writelines(decoy)
        self.target = f"{self.folder}/{self.folder}_target_metafile.txt"
        self.decoy = f"{self.folder}/{self.folder}_decoy_metafile.txt"
        return self

    def convert_to_pin(self):
        """ Converts the mzid files to a percolator input file using the metafiles created with create_metafiles()
        method. """
        msgf2pin_bin = resolve_executable(
            getattr(self.args, 'msgf2pin_path', None),
            'msgf2pin',
            env_var='UPROTEINS_MSGF2PIN',
            cli_flag='--msgf2pin_path',
        )
        ensure_dir(f"{self.folder}/Percolator")
        cmd_pin = [
            str(msgf2pin_bin),
            str(self.target),
            str(self.decoy),
            '-o', f'{self.folder}/Percolator/{self.folder}_pin.txt',
            '-F', f'{self.filetype}_database.fasta,{self.folder}/Percolator/{self.folder}_decoy.fasta',
            '-c', '2',
        ]
        run_cmd(cmd_pin)
        return self

    def percolate(self):
        """ Runs percolator on the converted mzid files. """
        perc_bin = resolve_executable(
            getattr(self.args, 'percolator_path', None),
            'percolator',
            env_var='UPROTEINS_PERCOLATOR',
            cli_flag='--percolator_path',
        )
        enz = getattr(self.args, 'enzyme', None) or "trypsinp"
        folder = f"{self.folder}/Percolator"
        cmd_perc = [
            str(perc_bin),
            '-X', f'{folder}/{self.filetype}_percolator_out.xml',
            '--tab-out', f'{folder}/comp_features.txt',
            '-w', f'{folder}/feature_weights.txt',
            '-v', '3',
            '-r', f'{folder}/{self.filetype}_pep_results.txt',
            '-m', f'{folder}/{self.filetype}_results_psm.txt',
            '--search-input', 'separate',
            '--results-proteins', f'{folder}/{self.filetype}_protein_results.txt',
            '--protein-enzyme', str(enz),
            '--protein-report-duplicates',
            '--protein-report-fragments',
            '--spectral-counting-fdr', '0.01',
            '-Y',
            '-P', 'decoy_',
            '--picked-protein', 'auto',
            f'{folder}/{self.folder}_pin.txt',
        ]
        run_cmd(cmd_perc)
        return self
