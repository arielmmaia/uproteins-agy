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


import pandas as pd

from ..sequtils.orflib import ORF, ORFCollection


class ExtendedInformation(object):
    def __init__(self, folder, filetype, alternatives):
        """
        Adds spectrum information for each of the extended ORFs based on the peptides that match them.
        :param folder: either Genome or Transcriptome
        :param filetype: either genome or transcriptome
        :param alternatives: the dictionary containing ORFCollection() instances generated with AltCodons() class
        methods
        """
        self.folder = folder
        self.filetype = filetype
        self.alternatives = alternatives
        self.results = pd.read_csv(f'{self.folder}/post_perc/{self.filetype}_results_02.txt', sep='\t')
        self.peptides = self.__fix_peptides()

    def filter_alternatives(self, priorities):
        """
        Excludes all but the top priority START codon based on the list of priorities.
        :param priorities: list containing ORF names. Generated with get_priorities() method from AltCodons().
        :return:
        """
        alternatives = {}
        for stop in self.alternatives:
            i = 0
            alternatives[stop] = self.alternatives[stop]
            for alt in self.alternatives[stop]:
                # if alt.name in priorities:
                if i == 0:
                    alternatives[stop].set_priority(alt)
                    # alternatives[stop] = ORFCollection()
                    # alternatives[stop].add_orf(alt)
                    i += 1
                else:
                    break
        # for stop in alternatives:
        #     for alt in alternatives[stop]:
        #         print(alt.name)
        self.alternatives = alternatives

    def __fix_peptides(self):
        peptides = self.results["Peptide"].tolist()

        def format_pep(pep):
            fixed = ""
            for i in pep:
                if i.isalpha():
                    fixed += i
            return fixed

        fixed_peptides = []
        for peptide in peptides:
            fixed = format_pep(peptide)
            fixed_peptides.append(fixed)
        self.results.insert(5, "Fixed Peptides", fixed_peptides)
        return fixed_peptides

    def extract_spectra(self):
        print('extracting spectra')
        frames = []
        extended = []
        ext_seqs = []
        energies = []
        coordinates = []
        sds = []
        dfdf = self.results.drop_duplicates(subset=["SpecFile", "ScanNum"], keep='last')
        k = 0
        for stop in self.alternatives:
            k += 1
            print(k, len(self.alternatives), end='\r')
            real_stop = self.alternatives[stop].priority.end
            df = dfdf[dfdf["Protein"].str.contains(str(real_stop), na=False)]
            priority = self.alternatives[stop].priority
            for alt in self.alternatives[stop]:
                ndf = df[df["Protein"].str.contains(alt.name, na=False)]
                frames.append(ndf)
                prots = ndf["Protein"].tolist()
                for _ in prots:
                    priority_coords = f'{priority.start}-{priority.end}'
                    extended.append(priority.name)
                    ext_seqs.append(priority.proteinSequence)
                    energies.append(priority.freeEnergy)
                    sds.append(priority.shineDalgarno)
                    coordinates.append(priority_coords)
        new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=self.results.columns)
        new_df.insert(3, "Extended ORF", extended)
        new_df.insert(4, "Extended Sequence", ext_seqs)
        new_df.insert(5, "Free Energy", energies)
        new_df.insert(6, "Shine Dalgarno", sds)
        new_df.insert(7, "Extended Coordinates", coordinates)
        new_df = new_df.drop_duplicates(subset=["SpecFile", "ScanNum"])
        new_df.to_csv(f'{self.folder}/post_perc/{self.filetype}_results_04.txt', sep='\t', index=False)
        return self




