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


import re
import typing as t

import pandas as pd

from .orflib import ORF, ORFCollection


_TRANSCRIPT_ID_RE = re.compile(r'transcript_id\s+"([^"]+)"')
_LEGACY_TX_FIXTURE_RE = re.compile(
    r"^(?:Transcriptome_)?(?P<tx>.+)_ORF_\d+_(?P<start>\d+)-(?P<end>\d+)(?P<sign>[+-])?$"
)


def parse_gorf_header(protein_id: str) -> t.Optional[tuple[t.Optional[str], int, int, str]]:
    """Parse a gORF protein identifier into (contig_id, start, end, strand).

    Supports both new contig-preserving headers (`gORF_NC_000962.3_1_15-62_forward`)
    and legacy headers (`Genome_gORF_1_15-62_forward`, `gORF__1_15-62_forward`).
    """
    token = protein_id.strip()
    if "(pre" in token:
        token = token[: token.rfind("(pre")]
    if token.startswith("decoy_"):
        token = token[len("decoy_") :]
    if token.startswith("Genome_"):
        token = token[len("Genome_") :]
    if not token.startswith("gORF_"):
        return None

    body = token[len("gORF_") :]
    parts = body.rsplit("_", 3)
    if len(parts) == 4:
        contig_raw, _orf_num, coords, strand = parts
        contig_id = contig_raw if contig_raw else None
    elif len(parts) == 3:
        _orf_num, coords, strand = parts
        contig_id = None
    else:
        return None

    if "-" not in coords:
        return None
    start_s, end_s = coords.split("-", 1)
    return contig_id, int(start_s), int(end_s), (strand or "forward")


def parse_torf_header(protein_id: str) -> t.Optional[tuple[str, int, int, str]]:
    """Parse a tORF protein identifier into (transcript_id, tx_start, tx_end, orf_strand).

    Supports:
    - New headers: `tORF_uproteins.10.2_1_15-62_forward` / `..._reverse`
    - Legacy empty-strand headers: `tORF_uproteins.10.1_1_15-62_`
    - Legacy fixture headers: `Transcriptome_uproteins.1.1_ORF_1_15-62+`
    """
    token = protein_id.strip()
    if "(pre" in token:
        token = token[: token.rfind("(pre")]
    if token.startswith("decoy_"):
        token = token[len("decoy_") :]

    m_leg = _LEGACY_TX_FIXTURE_RE.match(token)
    if m_leg:
        sign = m_leg.group("sign")
        strand = "reverse" if sign == "-" else "forward"
        return (
            m_leg.group("tx"),
            int(m_leg.group("start")),
            int(m_leg.group("end")),
            strand,
        )

    if token.startswith("Transcriptome_"):
        token = token[len("Transcriptome_") :]
    if not token.startswith("tORF_"):
        return None

    body = token[len("tORF_") :]
    parts = body.rsplit("_", 3)
    if len(parts) != 4:
        return None
    tx_id, _orf_num, coords, strand_raw = parts
    if "-" not in coords:
        return None
    start_s, end_s = coords.split("-", 1)
    strand = strand_raw if strand_raw in ("forward", "reverse") else "forward"
    return tx_id, int(start_s), int(end_s), strand


class RefSeqGFF(object):
    def __init__(self, gff):
        self.gff = gff

    def __read_gff(self):
        df = pd.read_csv(self.gff, sep='\t', header=1)
        df.columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes']
        df = df[df["source"].isin(["RefSeq", "ena"])]
        starts = df["start"].tolist()
        ends = df["end"].tolist()
        strands = df["strand"].tolist()
        attrs = df["attributes"].tolist()
        orf_list = []
        for i in range(len(attrs)):
            name = attrs[i].split(";")[1][12:]
            start = starts[i]
            end = ends[i]
            strand = strands[i]
            orf = ORF(name=name, start=start, end=end, strand=strand)
            orf_list.append(orf)
        return ORFCollection().add_orfs(orf_list)

    def get_dict(self):
        orfs = self.__read_gff()
        orf_dict = {}
        for orf in orfs:
            orf_dict[orf.name] = orf
        return orf_dict


class StringTieGFF(object):
    def __init__(self, gff):
        self.gff = gff

    def __read_gff(self):
        df = pd.read_csv(
            self.gff,
            sep='\t',
            comment='#',
            header=None,
            names=['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes'],
        )
        df = df[df["feature"] == "transcript"]
        starts = df["start"].tolist()
        ends = df["end"].tolist()
        strands = df["strand"].tolist()
        seqnames = df["seqname"].tolist()
        attrs = df["attributes"].tolist()

        orf_list = []
        for i in range(len(attrs)):
            m_tx = _TRANSCRIPT_ID_RE.search(str(attrs[i]))
            if m_tx:
                name = m_tx.group(1)
            else:
                name = str(attrs[i]).split(";")[1].split(" ")[-1].replace('"', "")
            orf = ORF(
                name=name,
                start=int(starts[i]),
                end=int(ends[i]),
                strand=str(strands[i]),
                chromosome=str(seqnames[i]),
            )
            orf_list.append(orf)
        return ORFCollection().add_orfs(orf_list)

    def get_dict(self):
        orfs = self.__read_gff()
        orf_dict = {}
        fallback_aliases = {}
        for orf in orfs:
            orf_dict[orf.name] = orf
            if orf.name.startswith("uproteins."):
                parts = orf.name.split(".")
                if len(parts) >= 2:
                    fallback_aliases.setdefault(f"{parts[0]}.{parts[1]}", orf)
            elif orf.name.startswith("gene-"):
                fallback_aliases.setdefault(orf.name[5:], orf)
            elif orf.name.startswith("rna-"):
                fallback_aliases.setdefault(orf.name[4:], orf)
        for alias, orf in fallback_aliases.items():
            orf_dict.setdefault(alias, orf)
        return orf_dict


class GenomeCoordinatesRNA(object):
    """For Transcriptome"""

    def __init__(self, psm_table, string_dict, ref_dict=None):
        self.psm = pd.read_csv(psm_table, sep='\t')
        self.ids = self.psm["proteinIds"].tolist()
        self.stringTieDict = string_dict
        self.refSeqDict = ref_dict or {}
        self.coordinates = []

    def _lookup_transcript(self, tx_id: str):
        if tx_id in self.stringTieDict:
            return self.stringTieDict[tx_id]
        if tx_id.startswith("gene-") and tx_id[5:] in self.stringTieDict:
            return self.stringTieDict[tx_id[5:]]
        if tx_id.startswith("rna-") and tx_id[4:] in self.stringTieDict:
            return self.stringTieDict[tx_id[4:]]
        if tx_id.startswith("uproteins."):
            short_id = ".".join(tx_id.split(".")[:2])
            if short_id in self.stringTieDict:
                return self.stringTieDict[short_id]
        return None

    def get_proteins(self):
        for raw_ids in self.ids:
            proteins = str(raw_ids).split(",")
            coord_items = []
            for protein in proteins:
                parsed = parse_torf_header(protein)
                if parsed is not None:
                    tx_id, orf_start, orf_end, _orf_strand = parsed
                    tx_entry = self._lookup_transcript(tx_id)
                    if tx_entry is not None:
                        genome_start = int(tx_entry.start) + int(orf_start)
                        genome_end = int(tx_entry.start) + int(orf_end)
                        lo, hi = (
                            (genome_end, genome_start)
                            if genome_start > genome_end
                            else (genome_start, genome_end)
                        )
                        coord_items.append(f"{lo}-{hi}")
                    else:
                        coord_items.append("not found")
                else:
                    if protein in self.refSeqDict:
                        orf = self.refSeqDict[protein]
                        coord_items.append(f"{int(orf.start) - 1}-{int(orf.end)}")
                    else:
                        coord_items.append("not found")
            self.coordinates.append(",".join(coord_items))
        return self

    def save_table(self, output):
        df = self.psm
        df.insert(5, "Genome Coordinates", self.coordinates)
        df.to_csv(f'{output}.txt', sep="\t", index=False)
        return self


def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)


class GenomeCoordinates(object):
    """For genome ORFs"""

    def __init__(self, psm_table, ref_dict=None):
        self.psm = pd.read_csv(psm_table, sep='\t')
        self.ids = self.psm["proteinIds"].tolist()
        self.refSeqDict = ref_dict or {}
        self.coordinates = []

    def get_coords(self):
        for raw_ids in self.ids:
            proteins = str(raw_ids).split(",")
            coord_items = []
            for protein in proteins:
                parsed = parse_gorf_header(protein)
                if parsed is not None:
                    _contig_id, start_c, end_c, _strand = parsed
                    lo, hi = (end_c, start_c) if start_c > end_c else (start_c, end_c)
                    coord_items.append(f"{lo}-{hi}")
                else:
                    if protein in self.refSeqDict:
                        orf = self.refSeqDict[protein]
                        coord_items.append(f"{orf.start}-{orf.end}")
                    else:
                        coord_items.append("not found")
            self.coordinates.append(",".join(coord_items))
        return self

    def save_table(self, output):
        df = self.psm
        df.insert(5, "Genome Coordinates", self.coordinates)
        df.to_csv(f'{output}.txt', sep="\t", index=False)
        return self
