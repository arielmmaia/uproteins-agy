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
        """Attach extended start-codon, RBS free-energy, and Shine-Dalgarno metadata
        to FDR-validated novel smORF PSMs while preserving paralogous genomic loci."""
        self.folder = folder
        self.filetype = filetype
        self.alternatives = alternatives
        self.results = pd.read_csv(
            f"{self.folder}/post_perc/{self.filetype}_results_02.txt", sep="\t"
        )
        self.peptides = self.__fix_peptides()

    def filter_alternatives(self, priorities):
        alternatives = {}
        for stop in self.alternatives:
            alternatives[stop] = self.alternatives[stop]
            for alt in self.alternatives[stop]:
                alternatives[stop].set_priority(alt)
                break
        self.alternatives = alternatives
        return self

    def __fix_peptides(self):
        peptides = self.results["Peptide"].tolist() if "Peptide" in self.results.columns else []
        fixed_peptides = [
            "".join(ch for ch in str(peptide) if ch.isalpha())
            for peptide in peptides
        ]
        self.results.insert(5, "Fixed Peptides", fixed_peptides)
        return fixed_peptides

    def extract_spectra(self):
        print("extracting spectra")
        spec_col = "SpecFile" if "SpecFile" in self.results.columns else "#SpecFile"
        # Preserve paralogous smORF loci matched by the same spectrum: only drop
        # exact duplicate (SpecFile, ScanNum, Protein, Genome Coordinates) rows!
        dedup_cols = [
            c
            for c in (spec_col, "ScanNum", "Protein", "Genome Coordinates")
            if c in self.results.columns
        ]
        dfdf = self.results.drop_duplicates(subset=dedup_cols).copy()

        frames = []
        extended = []
        ext_seqs = []
        energies = []
        coordinates = []
        sds = []

        for k, stop in enumerate(self.alternatives, start=1):
            print(k, len(self.alternatives), end="\r")
            collection = self.alternatives[stop]
            priority = collection.priority
            if priority is None:
                continue

            alt_names = {alt.name for alt in collection if getattr(alt, "name", None)}
            if not alt_names:
                continue

            ndf = dfdf[dfdf["Protein"].isin(alt_names)].copy()
            if ndf.empty:
                continue

            frames.append(ndf)
            lo_c, hi_c = (
                (int(priority.start), int(priority.end))
                if int(priority.start) <= int(priority.end)
                else (int(priority.end), int(priority.start))
            )
            priority_coords = (
                f"{priority.contig}:{lo_c}-{hi_c}({priority.strand_sign})"
                if getattr(priority, "contig_formatted", False)
                else f"{lo_c}-{hi_c}"
            )
            for _ in range(len(ndf)):
                extended.append(priority.name)
                ext_seqs.append(priority.proteinSequence)
                energies.append(priority.freeEnergy)
                sds.append(priority.shineDalgarno)
                coordinates.append(priority_coords)

        new_df = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=self.results.columns)
        )
        new_df.insert(3, "Extended ORF", extended)
        new_df.insert(4, "Extended Sequence", ext_seqs)
        new_df.insert(5, "Free Energy", energies)
        new_df.insert(6, "Shine Dalgarno", sds)
        new_df.insert(7, "Extended Coordinates", coordinates)

        post_dedup_cols = [
            c
            for c in (spec_col, "ScanNum", "Extended ORF", "Extended Coordinates")
            if c in new_df.columns
        ]
        new_df = new_df.drop_duplicates(subset=post_dedup_cols)
        new_df.to_csv(
            f"{self.folder}/post_perc/{self.filetype}_results_04.txt",
            sep="\t",
            index=False,
        )
        return self
