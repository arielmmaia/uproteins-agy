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


import pathlib

from ..sequtils import TranscriptomeTranslator, GenomeTranslator, DatabaseGenerator
from .proteome_matcher import CanonicalProteomeMatcher


class Database(object):
    def __init__(self, args):
        self.args = args
        self.genome = args.genome

    def translate(self):
        circular = getattr(self.args, "circular", True)
        genetic_code = getattr(self.args, "genetic_code", 11)
        minsize = int(self.args.minsize)
        maxsize = int(self.args.maxsize)
        proteome_path = getattr(self.args, "proteome", None)
        matcher = CanonicalProteomeMatcher(
            proteome_fasta=proteome_path,
            max_orf_aa=max(1, maxsize // 3),
            min_orf_aa=max(1, minsize // 3),
        )

        if self.args.transcriptome:
            gtf_candidate = pathlib.Path("assembled.gtf")
            rna = TranscriptomeTranslator(
                sequence="HISAT/transcripts.fasta",
                form='fasta',
                minsize=minsize,
                maxsize=maxsize,
                circular=False,
                genetic_code=genetic_code,
                gtf_path=gtf_candidate if gtf_candidate.is_file() else None,
            )
            rna_orfs = rna.parse_frames(
                starts=self.args.starts,
                stops=self.args.stops,
                seqtype='aa',
                entry='full',
            )
            tx_lengths = {
                idx + 1: len(seq)
                for idx, seq in enumerate(rna.seqsToTranslate)
            }
            rna_orfs = matcher.filter_orfs(rna_orfs, contig_lengths=tx_lengths)

            db = DatabaseGenerator(name="transcriptome", db_type="sql")
            db.add_orfs(rna_orfs)
            db.to_fasta(filename="transcriptome_ORFs.fasta", identifier='t')

        dna = GenomeTranslator(
            sequence=self.args.genome,
            form='fasta',
            minsize=minsize,
            maxsize=maxsize,
            circular=circular,
            genetic_code=genetic_code,
        )
        dna_orfs = dna.parse_frames(
            starts=self.args.starts,
            stops=self.args.stops,
            seqtype='aa',
            entry='full',
        )
        contig_lengths = {
            idx + 1: len(seq)
            for idx, seq in enumerate(dna.seqsToTranslate)
        }
        dna_orfs = matcher.filter_orfs(dna_orfs, contig_lengths=contig_lengths)

        genome_db = DatabaseGenerator(name="genome", db_type="sql")
        genome_db.add_orfs(dna_orfs)
        genome_db.to_fasta(filename="genome_ORFs.fasta", identifier='g')
