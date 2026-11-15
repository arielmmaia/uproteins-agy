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


from Bio import SeqIO


class ORF(object):
    def __init__(
        self,
        name=None,
        seq=None,
        start=None,
        end=None,
        cds=None,
        strand=None,
        chromosome=None,
        transcript=None,
        origin=None,
        appearances=1,
        experiment=None,
        protein_sequence=None,
        exons=None,
    ):
        self.name = name
        self.seq = seq
        self.start = start
        self.end = end
        self.cds = cds
        self.strand = strand
        self.chromosome = chromosome
        self.transcript = transcript
        self.origin = origin
        self.exons = list(exons) if exons is not None else None
        self.genomic_coordinates = None
        self.transcript_coordinates = None
        self.start_codon = None
        self.shineDalgarno = None
        self.freeEnergy = None
        self.proteinSequence = protein_sequence

        self.MSPeptides = []

        # for spectral counting
        self.appearances = appearances
        self.nsaf = None
        self.experiment = experiment
        self.normalizedSpec = None

        # after using msprocess
        self.peptides = {}
        self.closestToStart = self.find_ms_peptides() if self.end is not None else None

    def set_coordinates(self, genome):
        return self

    def __len__(self):
        return len(self.seq) if self.seq is not None else 0

    def filter_peptides(self):
        if self.proteinSequence is None:
            return
        self.MSPeptides = [
            pep for pep in self.MSPeptides if pep in self.proteinSequence
        ]

    def find_ms_peptides(self):
        if self.end is None:
            self.closestToStart = None
            return None
        closest_to_start = int(self.end)
        target_seq = self.proteinSequence if self.proteinSequence else self.seq
        if not target_seq or self.start is None:
            self.closestToStart = closest_to_start
            return closest_to_start

        use_aa_offset = bool(self.proteinSequence)
        for peptide in self.MSPeptides:
            idx = str(target_seq).find(peptide)
            if idx < 0:
                continue
            nt_offset = idx * 3 if use_aa_offset else idx
            if self.strand == "reverse":
                genome_start = int(self.start) - nt_offset
                if genome_start > closest_to_start:
                    closest_to_start = genome_start
            else:
                genome_start = int(self.start) + nt_offset
                if genome_start <= closest_to_start:
                    closest_to_start = genome_start
        self.closestToStart = closest_to_start
        return closest_to_start


class ORFCollection(object):
    def __init__(self):
        self.entries = None
        self.seqs = None
        self.starts = None
        self.ends = None
        self.coordinates = None
        self.orfs = []
        self.priority = None

    def __iter__(self):
        return iter(self.orfs)

    def set_priority(self, orf):
        self.priority = orf

    def read_fasta(self, file):
        """Adds all entries and sequences in a fasta file to this class instance."""
        records = SeqIO.parse(file, "fasta")
        for record in records:
            self.orfs.append(ORF(name=record.name, seq=str(record.seq)))
        return self

    def add_orfs(self, orfs):
        """Add ORFs translated with GenomeTranslator or TranscriptomeTranslator method."""
        for orf in orfs:
            self.orfs.append(orf)
        return self

    def add_orf(self, orf):
        """Add a single ORF to collection."""
        self.orfs.append(orf)

    def __len__(self):
        return len(self.orfs)

    def to_fasta(self, filename="orfs.fasta"):
        """Generates a fasta file containing all ORFs in this instance."""
        to_write = [
            f">{orf.name}_{orf.start}-{orf.end}_{orf.strand}\n{orf.seq}\n"
            for orf in self.orfs
        ]
        with open(filename, "w") as fa:
            fa.writelines(to_write)
