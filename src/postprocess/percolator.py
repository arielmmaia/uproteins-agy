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


class PercolatorProcessing(object):
    def __init__(self, folder, filetype, args=None):
        self.folder = folder
        self.args = args
        self.target = None
        self.decoy = None
        self.filetype = filetype

    def create_metafiles(self):
        """ Creates a metafile containing the paths to the mzid files, created during the peptide search step. """
        files = os.listdir(self.folder)
        real = []
        decoy = []
        for file in files:
            if "mzid" in file:
                if "decoy" in file:
                    decoy.append(os.path.abspath(f"{self.folder}/{file}\n"))
                else:
                    real.append(os.path.abspath(f"{self.folder}/{file}\n"))
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
        msgf2pin_path = "msgf2pin"
        if self.args is not None:
            if self.args.msgf2pin_path is not None:
                msgf2pin_path = self.args.msgf2pin_path
        if not os.path.exists(f"{self.folder}/Percolator"):
            cmd_dir = f'mkdir {self.folder}/Percolator'
            os.system(cmd_dir)
        cmd_pin = f'{msgf2pin_path} {self.target} {self.decoy} -o {self.folder}/Percolator/{self.folder}_pin.txt' \
                  f' -F {self.filetype}_database.fasta,{self.folder}/Percolator/{self.folder}_decoy.fasta -c 2'
        os.system(cmd_pin)
        return self

    def percolate(self):
        """ Runs percolator on the converted mzid files. """
        perc_path = "percolator"
        enz = "trypsinp"
        folder = f"{self.folder}/Percolator"
        if self.args is not None:
            if self.args.enzyme is not None:
                enz = self.args.enzyme
            if self.args.percolator_path is not None:
                perc_path = self.args.percolator_path
        cmd_perc = f'{perc_path} -X {self.folder}/Percolator/{self.filetype}_percolator_out.xml --tab-out ' \
                   f'{folder}/comp_features.txt -w {folder}/feature_weights.txt -v 3 -r {folder}/{self.filetype}_pep_results.txt -m ' \
                   f'{folder}/{self.filetype}_results_psm.txt --search-input separate --results-proteins {folder}/{self.filetype}_protein_results.txt' \
                   f' --protein-enzyme {enz} --protein-report-duplicates --protein-report-fragments ' \
                   f'--spectral-counting-fdr 0.01 -Y -P decoy_ --picked-protein auto {folder}/{self.folder}_pin.txt'
        os.system(cmd_perc)
        return self
