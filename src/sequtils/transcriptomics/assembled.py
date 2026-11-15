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
import re
import typing as t

from Bio import SeqIO


_TRANSCRIPT_ID_RE = re.compile(r'transcript_id\s+"([^"]+)"')
_ID_ATTR_RE = re.compile(r'(?:^|;)\s*ID=([^;\s]+)')


def parse_gtf_transcript_strands(gtf_path: t.Union[str, pathlib.Path]) -> dict[str, str]:
    """Parse transcript_id -> genomic strand ('+', '-', or '.') from a GTF/GFF file without mutating it."""
    path = pathlib.Path(gtf_path)
    strands: dict[str, str] = {}
    if not path.is_file():
        return strands

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("transcript", "mRNA", "exon"):
                continue
            strand = parts[6].strip() or "."
            attrs = parts[8]
            tx_id: t.Optional[str] = None
            m_gtf = _TRANSCRIPT_ID_RE.search(attrs)
            if m_gtf:
                tx_id = m_gtf.group(1)
            else:
                m_gff = _ID_ATTR_RE.search(attrs)
                if m_gff:
                    tx_id = m_gff.group(1)
            if tx_id and (tx_id not in strands or feature in ("transcript", "mRNA")):
                strands[tx_id] = strand
    return strands


class TranscriptExtractor(object):
    def __init__(self, assembly):
        self.assembly = SeqIO.parse(assembly, 'fasta')

    def get_transcripts(self) -> dict[str, str]:
        """Return transcript_id -> sequence preserving full isoform IDs (e.g. uproteins.10.2)."""
        rnas: dict[str, str] = {}
        fallback_aliases: dict[str, str] = {}
        for record in self.assembly:
            full_id = str(record.id)
            seq_str = str(record.seq)
            if full_id not in rnas:
                rnas[full_id] = seq_str

            # Keep legacy stripped prefixes only as non-colliding fallback aliases
            if full_id.startswith("gene-"):
                fallback_aliases.setdefault(full_id[5:], seq_str)
            elif full_id.startswith("rna-"):
                fallback_aliases.setdefault(full_id[4:], seq_str)
            elif full_id.startswith("uproteins."):
                parts = full_id.split(".")
                if len(parts) >= 2:
                    fallback_aliases.setdefault(f"{parts[0]}.{parts[1]}", seq_str)

        for alias, seq_str in fallback_aliases.items():
            rnas.setdefault(alias, seq_str)
        return rnas
