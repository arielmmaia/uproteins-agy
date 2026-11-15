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


import os
from Bio import SeqIO


class Database(object):
    def __init__(self, ob, p, filetype):
        self.orf_to_blast = ob
        self.proteome = p
        self.filetype = filetype

    def mark_annotated(self):
        with open('annotated.fasta', 'w') as handle:
            for record in SeqIO.parse(self.proteome, 'fasta'):
                record.id += '_ANNO'
                # Stopping the sequences from being written
                # >name name extra-info
                old_desc = record.description
                if old_desc.split()[0] == record.id:
                    # Old desc starts with id
                    record.description = ' '.join(old_desc.split()[1:])
                SeqIO.write(record, handle, 'fasta')
        self.annotated = 'annotated.fasta'

    def unify(self):
        cmd_cat = f'cat {self.orf_to_blast} {self.annotated} > {self.filetype}_database.fasta'
        os.system(cmd_cat)
