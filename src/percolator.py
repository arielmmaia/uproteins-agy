# Copyright © 2021-2025 Eduardo Vieira de Souza
# Copyright © 2021-2025 Adriana Canedo
# Copyright © 2021-2025 Cristiano Valim Bizarro
# Copyright © 2025 Bruno Maestri A Becker
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

from .utils import REPO_ROOT, ensure_dir


class Decoy(object):
    def __init__(self, db, db_type):
        self.df = db
        self.seqs, self.entries = self.__get_seqs()
        self.reversed = []
        self.type = db_type
        self.__create_dir()
        self.path = str(REPO_ROOT)

    def __create_dir(self):
        ensure_dir(f"{self.type}/Percolator")

    def __get_seqs(self):
        seqs = []
        entries = []
        records = SeqIO.parse(self.df, 'fasta')
        for record in records:
            seqs.append(record.seq)
            entries.append(str(record.description))
        return seqs, entries

    def reverse_sequences(self):
        """Reverses the amino acid sequence of a protein, except for the aa in
        the c-terminal.
        """
        for seq in self.seqs:
            cterminus = seq[-1]
            to_reverse = seq[:-1]
            reversed_seq = to_reverse[::-1]
            reversed_seq += cterminus
            self.reversed.append(reversed_seq)
        return self

    def add_contaminants(self):
        seqs = []
        records = SeqIO.parse(REPO_ROOT / 'seqlib' / 'contaminants.txt', 'fasta')
        for record in records:
            seqs.append(f">{record.id}\n{record.seq}\n")
        return seqs

    def to_fasta(self):
        out = []
        for i in range(len(self.reversed)):
            string = f">decoy_{self.entries[i]}\n{self.reversed[i]}\n"
            out.append(string)
        seqs = self.add_contaminants()
        for seq in seqs:
            to_add = seq.replace("B", "")
            to_add = to_add.replace("X", "")
            to_add = to_add.replace("Z", "")
            out.append(to_add)
        with open(f"{self.type}/Percolator/{self.type}_decoy.fasta", 'w') as fa:
            fa.writelines(out)
        return self
