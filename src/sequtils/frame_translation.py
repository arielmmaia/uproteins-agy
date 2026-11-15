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

from .conversion.translate import Translator
from .conversion import StrandConverter
from .orflib import ORF, ORFCollection
from .__helpers import FormatError


class FrameTranslator(object):
    def __init__(
        self,
        sequence,
        form,
        frames,
        minsize=18,
        maxsize=300,
        circular=False,
        genetic_code=11,
    ):
        self.sequence = sequence
        self.form = form
        self.frames = frames
        self.minSize = int(minsize)
        self.maxsize = int(maxsize)
        self.circular = bool(circular)
        self.genetic_code = int(genetic_code)
        self.orfs = []

    def read_genome(self):
        if self.format == 'fasta':
            records = SeqIO.parse(self.sequence, 'fasta')
            entries = []
            seqs = []
            for record in records:
                entries.append(record.id)
                seqs.append(str(record.seq))
            return seqs, entries
        elif self.format == 'list':
            if isinstance(self.sequence, tuple) and len(self.sequence) == 2:
                return [str(s) for s in self.sequence[0]], list(self.sequence[1])
            seqs = [str(s) for s in self.sequence]
            entries = [f"seq_{idx + 1}" for idx in range(len(seqs))]
            return seqs, entries
        else:
            raise FormatError

    def parse_frames(self, starts=None, stops=None, seqtype='both', **kwargs):
        """Specify a list of start and stop codons. By default, only 'ATG' is included as a start. TGA, TAA and TAG
        are the default stop codons. 'seqtype' accepts 'aa', 'cds', or 'both'. It changes which type of sequence the
        function adds to the 'ORF' object. To save memory, you may exclude one of them."""
        if starts is None:
            starts = ['ATG']
        if stops is None:
            stops = ['TGA', 'TAA', 'TAG']

        start_set = {c.upper() for c in starts}
        stop_tuple = tuple(c.upper() for c in stops)
        stop_set = set(stop_tuple)

        sequences = [str(s) for s in self.seqsToTranslate]
        if kwargs.get("entry") == "full" and self.form == "fasta":
            entries = self.entries
        else:
            entries = ["" for _ in range(len(sequences))]

        forward_orfs = self.__get_cds(
            sequences,
            strand='forward',
            start_set=start_set,
            stop_set=stop_set,
            stop_tuple=stop_tuple,
            seqtype=seqtype,
            entries=entries,
        )
        self.__add_orfs(forward_orfs)

        if self.frames == 6:
            reverse_sequences = StrandConverter(sequences)
            rev = reverse_sequences.complement().reverse()
            reverse_orfs = self.__get_cds(
                rev,
                strand='reverse',
                start_set=start_set,
                stop_set=stop_set,
                stop_tuple=stop_tuple,
                seqtype=seqtype,
                entries=entries,
            )
            self.__add_orfs(reverse_orfs)

        return ORFCollection().add_orfs(self.orfs)

    def __add_orfs(self, orf_list):
        """Appends all ORFs in a list to the 'orfs' attribute of this class' instance."""
        self.orfs.extend(orf_list)
        return self

    def __get_cds(
        self,
        sequences,
        strand,
        start_set,
        stop_set,
        stop_tuple,
        seqtype,
        entries,
    ):
        """Returns all possible CDS for the three frames of a nucleotide sequence using a deterministic codon walker."""
        orfs = []
        orf_number = 0

        for seq_idx, raw_seq in enumerate(sequences):
            seq_str = str(raw_seq)
            seq_len = len(seq_str)
            if seq_len < 3:
                continue

            upper_seq = seq_str.upper().replace("U", "T")
            if self.circular:
                max_orf_nt = min(self.maxsize, seq_len) if self.maxsize > 0 else seq_len
                wrap_nt = min(seq_len, max_orf_nt + 3)
                scan_seq = upper_seq + upper_seq[:wrap_nt]
                scan_raw = seq_str + seq_str[:wrap_nt]
            else:
                max_orf_nt = self.maxsize if self.maxsize > 0 else seq_len
                scan_seq = upper_seq
                scan_raw = seq_str

            scan_len = len(scan_seq)
            candidates = []

            for frame in (0, 1, 2):
                open_starts = []
                for pos in range(frame, scan_len - 2, 3):
                    codon = scan_seq[pos : pos + 3]
                    if codon in stop_set:
                        if open_starts:
                            for s_pos in open_starts:
                                orf_len = pos - s_pos
                                if self.minSize <= orf_len <= max_orf_nt:
                                    candidates.append((s_pos, pos))
                            open_starts = []
                    else:
                        if pos < seq_len and codon in start_set:
                            open_starts.append(pos)
                        if max_orf_nt > 0 and open_starts:
                            min_valid_start = (pos + 3) - max_orf_nt
                            if open_starts[0] < min_valid_start:
                                open_starts = [s for s in open_starts if s >= min_valid_start]

            candidates.sort()
            for s_pos, e_pos in candidates:
                cds_seq = scan_raw[s_pos:e_pos]
                start_coord, end_coord = self.__compute_coordinates(
                    s_pos, e_pos, seq_len, strand
                )
                orf_number += 1
                orf_i = self.__check_seqtype(
                    seqtype=seqtype,
                    start=start_coord,
                    end=end_coord,
                    orf_number=f'{entries[seq_idx]}_{orf_number}',
                    strand=strand,
                    cds=cds_seq,
                    chromosome=seq_idx + 1,
                    stops=stop_tuple,
                )
                if orf_i is not None:
                    orfs.append(orf_i)

        return orfs

    @staticmethod
    def __compute_coordinates(s_pos: int, e_pos: int, seq_len: int, strand: str) -> tuple[int, int]:
        """Convert 0-based [s_pos, e_pos) strand slice (excluding stop codon) to 1-based inclusive coordinates."""
        if strand == "forward":
            start_1b = s_pos + 1
            end_1b = ((e_pos - 1) % seq_len) + 1
            return start_1b, end_1b
        else:
            start_1b = seq_len - (s_pos % seq_len)
            end_1b = seq_len - ((e_pos - 1) % seq_len)
            return start_1b, end_1b

    def __check_seqtype(self, seqtype, **kwargs):
        """Check which type of sequence should be added to the 'ORF' object."""
        start = kwargs.get("start")
        end = kwargs.get("end")
        orf_number = kwargs.get("orf_number")
        strand = kwargs.get("strand")
        cds = None
        orf = kwargs.get("cds")
        stops = kwargs.get("stops")
        protein = None
        chromosome = None
        transcript = None
        origin = None
        if self.frames == 6:
            chromosome = kwargs.get('chromosome')
            origin = "Genome"
        elif self.frames == 3:
            strand = ""
            transcript = kwargs.get('chromosome')
            origin = "Transcriptome"
        if seqtype == "cds" or seqtype == "both":
            cds = orf
        if seqtype == "aa" or seqtype == "both":
            aa = Translator(orf, stops=stops, genetic_code=self.genetic_code)
            protein = aa.translate()
        if protein is None or "*" not in protein:
            orf_i = ORF(
                name=f'ORF_{orf_number}',
                start=start,
                end=end,
                cds=cds,
                seq=protein if protein is not None else cds,
                strand=strand,
                chromosome=chromosome,
                transcript=transcript,
                origin=origin,
            )
            return orf_i
        return None

    def __within_limits(self, orf):
        """Checks if the predicted ORF is within the size constraints specified by 'minsize' and 'maxsize'."""
        return self.minSize <= len(orf) <= self.maxsize


class GenomeTranslator(FrameTranslator):
    def __init__(
        self,
        sequence=None,
        form='fasta',
        minsize=18,
        maxsize=300,
        circular=True,
        genetic_code=11,
    ):
        """Sequence must be either a list or the path to a fasta file. The format must be specified by 'form', which is
        set to 'fasta' by default. If it is a list, each element must be a chromosome and thus contain all its
        nucleotide sequences. minsize and maxsize refer to the ORF length in nucleotides, and are set to 18 and 300
        by default, respectively, in order to detect only small ORFs."""
        self.frames = 6
        self.format = form
        self.sequence = sequence
        super().__init__(
            self.sequence,
            form,
            self.frames,
            minsize=minsize,
            maxsize=maxsize,
            circular=circular,
            genetic_code=genetic_code,
        )
        self.seqsToTranslate, self.entries = self.read_genome()


class TranscriptomeTranslator(FrameTranslator):
    def __init__(
        self,
        sequence=None,
        form='fasta',
        minsize=18,
        maxsize=300,
        circular=False,
        genetic_code=11,
    ):
        """Sequence must be either a list or the path to a fasta file. The format must be specified by 'form', which is
        set to 'fasta' by default. If it is a list, each element must be a transcript and thus contain all its
        nucleotide sequences. minsize and maxsize refer to the ORF length in nucleotides, and are set to 18 and 300
        by default, respectively, in order to detect only small ORFs."""
        self.frames = 3
        self.format = form
        self.sequence = sequence
        super().__init__(
            self.sequence,
            form,
            self.frames,
            minsize=minsize,
            maxsize=maxsize,
            circular=circular,
            genetic_code=genetic_code,
        )
        self.seqsToTranslate, self.entries = self.read_genome()