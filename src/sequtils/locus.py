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
import pandas as pd

from .orflib import ORF, ORFCollection


GFF_COLUMNS: list[str] = [
    "seqname",
    "source",
    "feature",
    "start",
    "end",
    "score",
    "strand",
    "frame",
    "attributes",
]

_TRANSCRIPT_ID_RE = re.compile(r'transcript_id\s+"([^"]+)"')
_GTF_KV_RE = re.compile(r'([A-Za-z0-9_.-]+)\s+"([^"]*)"')
_PROTEIN_SPLIT_RE = re.compile(r"[;\t]+|,(?![^()]*\))")
_COORD_TOKEN_RE = re.compile(
    r"^(?:(?P<contig>[^:\s]+):)?(?P<start>\d+)-(?P<end>\d+)(?:\((?P<strand>[+-])\))?$"
)
_LEGACY_TX_FIXTURE_RE = re.compile(
    r"^(?:Transcriptome_)?(?P<tx>.+)_ORF_(?P<num>\d+)_(?P<start>\d+)-(?P<end>\d+)(?P<sign>[+-])?$"
)
_IGNORED_GFF_FEATURES = frozenset(
    {
        "region",
        "databank",
        "remark",
        "chromosome",
        "contig",
        "supercontig",
        "biological_region",
        "sequence_feature",
    }
)


def load_genome_contigs(
    genome_fasta: str | pathlib.Path,
) -> tuple[dict[str, str], str]:
    """Load all reference FASTA contigs/plasmids into `{record.id: UPPER_SEQ}`
    together with the primary (first) contig ID."""
    contigs: dict[str, str] = {}
    default_contig = ""
    for record in SeqIO.parse(str(genome_fasta), "fasta"):
        seq_str = str(record.seq).upper()
        if not default_contig:
            default_contig = record.id
        contigs[record.id] = seq_str
    return contigs, default_contig


def resolve_contig_sequence(
    contigs: dict[str, str],
    contig_id: str | None,
    default_contig: str = "",
) -> str:
    """Resolve the DNA sequence for `contig_id` across exact ID, version-stripped ID,
    or `default_contig` fallback."""
    if not contigs:
        return ""
    if contig_id:
        if contig_id in contigs:
            return contigs[contig_id]
        bare = contig_id.split(".")[0]
        for k, v in contigs.items():
            if k == bare or k.split(".")[0] == bare:
                return v
    if default_contig and default_contig in contigs:
        return contigs[default_contig]
    return next(iter(contigs.values()))


def parse_coordinate_token(
    coord_str: str,
) -> t.Optional[tuple[t.Optional[str], int, int, t.Optional[str]]]:
    """Parse `'114-161'` or `'NC_000962.3:114-161(-)'` into `(contig, lo, hi, strand)`
    with `lo <= hi` guaranteed."""
    if coord_str is None:
        return None
    m = _COORD_TOKEN_RE.match(str(coord_str).strip())
    if not m:
        return None
    a = int(m.group("start"))
    b = int(m.group("end"))
    lo, hi = (a, b) if a <= b else (b, a)
    return m.group("contig"), lo, hi, m.group("strand")


def format_orf_display_name(protein_id: str) -> str:
    """Convert an ORF FASTA identifier into its clean locus display name
    without coordinate/strand suffixes."""
    token = str(protein_id).strip()
    if "(pre" in token:
        token = token[: token.rfind("(pre")]
    m_leg = _LEGACY_TX_FIXTURE_RE.match(token)
    if m_leg:
        prefix = "Transcriptome_" if token.startswith("Transcriptome_") else ""
        return f"{prefix}{m_leg.group('tx')}_ORF_{m_leg.group('num')}"

    g_parsed = parse_gorf_header(token)
    if g_parsed is not None:
        clean = token
        if clean.startswith("decoy_"):
            clean = clean[6:]
        if clean.startswith("Genome_"):
            clean = clean[len("Genome_") :]
        body = clean[len("gORF_") :]
        parts = body.rsplit("_", 3)
        if len(parts) == 4:
            contig_raw, orf_num, _coords, _strand = parts
            return f"gORF_{contig_raw}_{orf_num}" if contig_raw else f"gORF_{orf_num}"
        if len(parts) == 3:
            orf_num, _coords, _strand = parts
            return f"gORF_{orf_num}"

    t_parsed = parse_torf_header(token)
    if t_parsed is not None:
        clean = token
        if clean.startswith("decoy_"):
            clean = clean[6:]
        if clean.startswith("Transcriptome_"):
            clean = clean[len("Transcriptome_") :]
        if clean.startswith("tORF_"):
            body = clean[len("tORF_") :]
            parts = body.rsplit("_", 3)
            if len(parts) == 4:
                tx_id, orf_num, _coords, _strand = parts
                return f"tORF_{tx_id}_{orf_num}"

    parts = token.split("_")
    if len(parts) > 2:
        return "_".join(parts[:-2]).replace("__", "_")
    return token


def read_gff_dataframe(gff_path: str | pathlib.Path | None) -> pd.DataFrame:
    """Non-destructively parse any GFF3 or GTF file into a 9-column DataFrame.

    - Gracefully returns an empty DataFrame when `gff_path` is `None`.
    - Skips `#` comment/directive lines (`##gff-version`, `##sequence-region`, etc.)
      without ever modifying the user's file on disk.
    - Stops cleanly at `##FASTA` or embedded FASTA records (`>`), as produced by
      Prokka, Bakta, and Prodigal.
    """
    if gff_path is None or str(gff_path).strip() == "":
        return pd.DataFrame(columns=GFF_COLUMNS)
    p = pathlib.Path(gff_path)
    if not p.is_file():
        return pd.DataFrame(columns=GFF_COLUMNS)

    rows: list[tuple[str, str, str, int, int, str, str, str, str]] = []
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("##FASTA") or line.startswith(">"):
                break
            if line.startswith("#"):
                continue
            cols = line.split("\t", 8)
            if len(cols) < 9:
                continue
            try:
                start_i = int(cols[3])
                end_i = int(cols[4])
            except ValueError:
                continue
            lo, hi = (start_i, end_i) if start_i <= end_i else (end_i, start_i)
            rows.append(
                (
                    cols[0],
                    cols[1],
                    cols[2],
                    lo,
                    hi,
                    cols[5],
                    cols[6],
                    cols[7],
                    cols[8],
                )
            )
    if not rows:
        return pd.DataFrame(columns=GFF_COLUMNS)
    return pd.DataFrame(rows, columns=GFF_COLUMNS)


def parse_gff_attributes(attr_str: str) -> dict[str, str]:
    """Parse both GFF3 (`key=value;...`) and GTF (`key "value"; ...`) attribute strings."""
    attrs: dict[str, str] = {}
    if not attr_str or pd.isna(attr_str):
        return attrs
    text = str(attr_str).strip()
    for m in _GTF_KV_RE.finditer(text):
        attrs.setdefault(m.group(1), m.group(2))
    for field in text.split(";"):
        part = field.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"')
        if k:
            attrs[k] = v
    return attrs


def project_transcript_interval_to_genome(
    exons: t.Sequence[tuple[int, int]],
    tx_strand: str,
    tx_orf_start: int,
    tx_orf_end: int,
    orf_strand: str = "forward",
) -> tuple[int, int, str]:
    """Project a 1-based inclusive local RNA interval `[tx_orf_start, tx_orf_end]`
    onto 1-based inclusive genomic coordinates `(genome_start, genome_end, genome_strand)`
    across single- or multi-exon transcripts on `+`, `-`, or `.` strands.

    Guarantees `genome_start <= genome_end` universally!
    """
    if not exons:
        raise ValueError("Cannot project transcript coordinates without at least one exon.")

    lo_tx = max(1, min(int(tx_orf_start), int(tx_orf_end)))
    hi_tx = max(1, max(int(tx_orf_start), int(tx_orf_end)))

    norm_exons = [
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b in exons
    ]
    strand_clean = str(tx_strand).strip() if tx_strand else "+"

    def _map_offset(offset_1based: int) -> int:
        rem = offset_1based
        if strand_clean == "-":
            ordered = sorted(norm_exons, key=lambda e: e[1], reverse=True)
            for e_start, e_end in ordered:
                exon_len = e_end - e_start + 1
                if rem <= exon_len:
                    return e_end - (rem - 1)
                rem -= exon_len
            last_start, _last_end = ordered[-1]
            return max(1, last_start - rem)
        else:
            ordered = sorted(norm_exons, key=lambda e: e[0])
            for e_start, e_end in ordered:
                exon_len = e_end - e_start + 1
                if rem <= exon_len:
                    return e_start + (rem - 1)
                rem -= exon_len
            _last_start, last_end = ordered[-1]
            return last_end + rem

    g_a = _map_offset(lo_tx)
    g_b = _map_offset(hi_tx)
    genome_start = min(g_a, g_b)
    genome_end = max(g_a, g_b)

    is_rev_orf = str(orf_strand).lower() in ("reverse", "-")
    if strand_clean == "-":
        genome_strand = "+" if is_rev_orf else "-"
    elif strand_clean == "+":
        genome_strand = "-" if is_rev_orf else "+"
    else:
        genome_strand = "-" if is_rev_orf else "+"

    return genome_start, genome_end, genome_strand


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
    try:
        return contig_id, int(start_s), int(end_s), (strand or "forward")
    except ValueError:
        return None


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
    try:
        return tx_id, int(start_s), int(end_s), strand
    except ValueError:
        return None


class RefSeqGFF(object):
    """Non-destructive GFF3/GTF reader supporting NCBI RefSeq, Protein Homology,
    Prodigal, Prokka, Bakta, GenBank, ENA, and graceful `gff=None` omission."""

    def __init__(self, gff):
        self.gff = gff

    @staticmethod
    def _extract_candidate_keys(attrs: dict[str, str], raw_attr: str) -> list[str]:
        keys: list[str] = []
        for field_name in (
            "Parent",
            "ID",
            "locus_tag",
            "Name",
            "gene_id",
            "protein_id",
            "gene",
            "transcript_id",
        ):
            val = attrs.get(field_name)
            if not val:
                continue
            for item in val.split(","):
                item = item.strip()
                if not item:
                    continue
                keys.append(item)
                for prefix in ("gene-", "cds-", "rna-", "transcript-"):
                    if item.startswith(prefix):
                        keys.append(item[len(prefix) :])

        if not keys and raw_attr:
            parts = [p.strip() for p in str(raw_attr).split(";") if p.strip()]
            if parts:
                keys.append(parts[0])

        expanded: list[str] = []
        seen: set[str] = set()
        for k in keys:
            for alias in (k, f"{k}_ANNO" if not k.endswith("_ANNO") else k[:-5]):
                if alias and alias not in seen:
                    seen.add(alias)
                    expanded.append(alias)
        return expanded

    def _read_entries(self) -> tuple[ORFCollection, dict[str, ORF]]:
        df = read_gff_dataframe(self.gff)
        collection = ORFCollection()
        orf_dict: dict[str, ORF] = {}
        if df.empty:
            return collection, orf_dict

        # Filter out whole-chromosome / region meta-features while keeping CDS/gene/RNA rows
        feat_lower = df["feature"].astype(str).str.lower()
        df = df[~feat_lower.isin(_IGNORED_GFF_FEATURES)].copy()
        if df.empty:
            return collection, orf_dict

        # Prefer CDS rows over gene rows when both share the same identifier
        priority_rank = {"cds": 0, "gene": 1, "mrna": 2, "transcript": 3}
        df["_prio"] = df["feature"].astype(str).str.lower().map(
            lambda f: priority_rank.get(f, 4)
        )
        df = df.sort_values("_prio", kind="mergesort")

        orf_list: list[ORF] = []
        for row in df.itertuples(index=False):
            attrs = parse_gff_attributes(row.attributes)
            keys = self._extract_candidate_keys(attrs, row.attributes)
            if not keys:
                continue
            primary_name = keys[0]
            orf = ORF(
                name=primary_name,
                start=int(row.start),
                end=int(row.end),
                strand=str(row.strand),
                chromosome=str(row.seqname),
            )
            orf_list.append(orf)
            for alias in keys:
                orf_dict.setdefault(alias, orf)

        collection.add_orfs(orf_list)
        return collection, orf_dict

    def get_dict(self) -> dict[str, ORF]:
        _, orf_dict = self._read_entries()
        return orf_dict


class StringTieGFF(object):
    """Non-destructive StringTie/GTF reader preserving multi-exon coordinates and
    full isoform IDs (`uproteins.10.2`) without modifying input files on disk."""

    def __init__(self, gff):
        self.gff = gff

    @staticmethod
    def _extract_tx_id(raw_attr: str) -> str:
        m_tx = _TRANSCRIPT_ID_RE.search(str(raw_attr))
        if m_tx:
            return m_tx.group(1)
        attrs = parse_gff_attributes(raw_attr)
        for key in ("transcript_id", "ID", "Parent", "gene_id", "Name"):
            if key in attrs and attrs[key]:
                return attrs[key]
        parts = [p.strip() for p in str(raw_attr).split(";") if p.strip()]
        if len(parts) >= 2:
            return parts[1].split(" ")[-1].replace('"', "")
        return parts[0].replace('"', "") if parts else ""

    def __read_gff(self) -> ORFCollection:
        df = read_gff_dataframe(self.gff)
        if df.empty:
            return ORFCollection()

        tx_meta: dict[str, tuple[str, int, int, str]] = {}
        tx_exons: dict[str, list[tuple[int, int]]] = {}

        for row in df.itertuples(index=False):
            feat = str(row.feature).lower()
            if feat not in ("transcript", "mrna", "exon", "cds"):
                continue
            tx_id = self._extract_tx_id(row.attributes)
            if not tx_id:
                continue
            start_i = int(row.start)
            end_i = int(row.end)
            seqname = str(row.seqname)
            strand = str(row.strand)

            if feat in ("transcript", "mrna"):
                tx_meta[tx_id] = (seqname, start_i, end_i, strand)
            elif feat == "exon":
                tx_exons.setdefault(tx_id, []).append((start_i, end_i))
                if tx_id not in tx_meta:
                    tx_meta[tx_id] = (seqname, start_i, end_i, strand)
            elif feat == "cds" and tx_id not in tx_meta:
                tx_exons.setdefault(tx_id, []).append((start_i, end_i))
                tx_meta[tx_id] = (seqname, start_i, end_i, strand)

        orf_list: list[ORF] = []
        for tx_id, (seqname, meta_start, meta_end, strand) in tx_meta.items():
            exons = sorted(tx_exons.get(tx_id, []), key=lambda e: e[0])
            if exons:
                span_start = min(e[0] for e in exons)
                span_end = max(e[1] for e in exons)
            else:
                span_start, span_end = meta_start, meta_end
                exons = [(span_start, span_end)]

            orf = ORF(
                name=tx_id,
                start=span_start,
                end=span_end,
                strand=strand,
                chromosome=seqname,
                exons=exons,
            )
            orf_list.append(orf)

        return ORFCollection().add_orfs(orf_list)

    def get_dict(self) -> dict[str, ORF]:
        orfs = self.__read_gff()
        orf_dict: dict[str, ORF] = {}
        fallback_aliases: dict[str, ORF] = {}
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


def _split_protein_field(raw_ids: object) -> list[str]:
    if pd.isna(raw_ids):
        return []
    tokens: list[str] = []
    for raw in _PROTEIN_SPLIT_RE.split(str(raw_ids)):
        tok = raw.strip()
        if not tok:
            continue
        if "(pre" in tok:
            tok = tok[: tok.rfind("(pre")].strip()
        if tok:
            tokens.append(tok)
    return tokens


def _format_genomic_coord(
    contig: str | None,
    lo: int,
    hi: int,
    strand_sign: str | None,
    include_contig: bool,
) -> str:
    lo_i, hi_i = (int(lo), int(hi)) if int(lo) <= int(hi) else (int(hi), int(lo))
    if include_contig and contig:
        sign = strand_sign if strand_sign in ("+", "-") else "+"
        return f"{contig}:{lo_i}-{hi_i}({sign})"
    return f"{lo_i}-{hi_i}"


class GenomeCoordinatesRNA(object):
    """Project local 1-based transcript ORF coordinates onto genomic coordinates."""

    def __init__(
        self,
        psm_table,
        string_dict,
        ref_dict=None,
        include_contig: bool | None = None,
    ):
        self.psm = pd.read_csv(psm_table, sep="\t")
        self.ids = (
            self.psm["proteinIds"].tolist() if "proteinIds" in self.psm.columns else []
        )
        self.stringTieDict = string_dict or {}
        self.refSeqDict = ref_dict or {}
        if include_contig is None:
            contigs_seen = {
                str(orf.chromosome)
                for orf in list(self.stringTieDict.values())
                + list(self.refSeqDict.values())
                if getattr(orf, "chromosome", None)
            }
            self.include_contig = len(contigs_seen) > 1
        else:
            self.include_contig = bool(include_contig)
        self.coordinates: list[str] = []

    def _lookup_transcript(self, tx_id: str) -> ORF | None:
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

    def _lookup_refseq(self, protein: str) -> ORF | None:
        if protein in self.refSeqDict:
            return self.refSeqDict[protein]
        if protein.endswith("_ANNO") and protein[:-5] in self.refSeqDict:
            return self.refSeqDict[protein[:-5]]
        return None

    def get_proteins(self):
        for raw_ids in self.ids:
            proteins = _split_protein_field(raw_ids)
            coord_items: list[str] = []
            for protein in proteins:
                parsed = parse_torf_header(protein)
                if parsed is not None:
                    tx_id, orf_start, orf_end, orf_strand = parsed
                    tx_entry = self._lookup_transcript(tx_id)
                    if tx_entry is not None:
                        exons = tx_entry.exons or [(int(tx_entry.start), int(tx_entry.end))]
                        g_lo, g_hi, g_strand = project_transcript_interval_to_genome(
                            exons=exons,
                            tx_strand=str(tx_entry.strand or "+"),
                            tx_orf_start=orf_start,
                            tx_orf_end=orf_end,
                            orf_strand=orf_strand,
                        )
                        coord_items.append(
                            _format_genomic_coord(
                                getattr(tx_entry, "chromosome", None),
                                g_lo,
                                g_hi,
                                g_strand,
                                self.include_contig,
                            )
                        )
                    else:
                        coord_items.append("not found")
                else:
                    ref_orf = self._lookup_refseq(protein)
                    if ref_orf is not None:
                        coord_items.append(
                            _format_genomic_coord(
                                getattr(ref_orf, "chromosome", None),
                                int(ref_orf.start),
                                int(ref_orf.end),
                                str(ref_orf.strand or "+"),
                                self.include_contig,
                            )
                        )
                    else:
                        coord_items.append("not found")
            self.coordinates.append(",".join(coord_items))
        return self

    def save_table(self, output):
        df = self.psm.copy()
        df.insert(5, "Genome Coordinates", self.coordinates)
        df.to_csv(f"{output}.txt", sep="\t", index=False)
        return self


def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)


class GenomeCoordinates(object):
    """Extract 1-based inclusive genomic coordinates for genome ORFs."""

    def __init__(
        self,
        psm_table,
        ref_dict=None,
        include_contig: bool | None = None,
    ):
        self.psm = pd.read_csv(psm_table, sep="\t")
        self.ids = (
            self.psm["proteinIds"].tolist() if "proteinIds" in self.psm.columns else []
        )
        self.refSeqDict = ref_dict or {}
        if include_contig is None:
            contigs_seen: set[str] = {
                str(orf.chromosome)
                for orf in self.refSeqDict.values()
                if getattr(orf, "chromosome", None)
            }
            for raw_ids in self.ids:
                for tok in _split_protein_field(raw_ids):
                    gp = parse_gorf_header(tok)
                    if gp is not None and gp[0]:
                        contigs_seen.add(gp[0])
            self.include_contig = len(contigs_seen) > 1
        else:
            self.include_contig = bool(include_contig)
        self.coordinates: list[str] = []

    def _lookup_refseq(self, protein: str) -> ORF | None:
        if protein in self.refSeqDict:
            return self.refSeqDict[protein]
        if protein.endswith("_ANNO") and protein[:-5] in self.refSeqDict:
            return self.refSeqDict[protein[:-5]]
        return None

    def get_coords(self):
        for raw_ids in self.ids:
            proteins = _split_protein_field(raw_ids)
            coord_items: list[str] = []
            for protein in proteins:
                parsed = parse_gorf_header(protein)
                if parsed is not None:
                    contig_id, start_c, end_c, strand_name = parsed
                    strand_sign = "-" if strand_name == "reverse" else "+"
                    coord_items.append(
                        _format_genomic_coord(
                            contig_id,
                            start_c,
                            end_c,
                            strand_sign,
                            self.include_contig,
                        )
                    )
                else:
                    ref_orf = self._lookup_refseq(protein)
                    if ref_orf is not None:
                        coord_items.append(
                            _format_genomic_coord(
                                getattr(ref_orf, "chromosome", None),
                                int(ref_orf.start),
                                int(ref_orf.end),
                                str(ref_orf.strand or "+"),
                                self.include_contig,
                            )
                        )
                    else:
                        coord_items.append("not found")
            self.coordinates.append(",".join(coord_items))
        return self

    def save_table(self, output):
        df = self.psm.copy()
        df.insert(5, "Genome Coordinates", self.coordinates)
        df.to_csv(f"{output}.txt", sep="\t", index=False)
        return self
