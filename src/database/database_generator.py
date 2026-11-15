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
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..utils import concat_files


class Database(object):
    def __init__(self, ob, p, filetype):
        self.orf_to_blast = ob
        self.proteome = p
        self.filetype = filetype
        self.annotated = 'annotated.fasta'

    def mark_annotated(self):
        with open('annotated.fasta', 'w') as handle:
            for record in SeqIO.parse(self.proteome, 'fasta'):
                orig_id = record.id
                new_id = orig_id if orig_id.endswith('_ANNO') else f"{orig_id}_ANNO"
                desc_tokens = record.description.split()
                while desc_tokens and desc_tokens[0] in (orig_id, new_id):
                    desc_tokens.pop(0)
                extra_desc = " ".join(desc_tokens)
                clean_seq = "".join(str(record.seq).split()).rstrip("*").upper()
                if not clean_seq:
                    continue

                out_record = SeqRecord(
                    Seq(clean_seq),
                    id=new_id,
                    name=new_id,
                    description=extra_desc,
                )
                SeqIO.write(out_record, handle, 'fasta')
        self.annotated = 'annotated.fasta'

    def unify(self):
        concat_files(
            [self.orf_to_blast, self.annotated],
            f'{self.filetype}_database.fasta',
        )
