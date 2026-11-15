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


import hashlib
import pathlib
import re

from Bio import SeqIO
import pandas as pd

from .locus import parse_gorf_header, parse_torf_header
from ..utils import REPO_ROOT


_FLANKING_RE = re.compile(r"^[A-Za-z\-]\.(.*)\.[A-Za-z\-]$")
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_AA_RE = re.compile(r"[^A-Za-z]")
_PROTEIN_SPLIT_RE = re.compile(r"[;\t]+|,(?![^()]*\))")
_COORD_PARSE_RE = re.compile(
    r"^(?:(?P<contig>[^:]+):)?(?P<start>\d+)-(?P<end>\d+)(?:\((?P<strand>[+\-.])\))?$"
)
_MET_EXCISE_NEXT: frozenset[str] = frozenset({"A", "C", "G", "P", "S", "T", "V"})


def normalize_isobaric_peptide(raw_peptide: str) -> str:
    """Strip flanking cleavage/PTMs and normalize mass-isobaric `I -> L`."""
    pep = str(raw_peptide).strip()
    m = _FLANKING_RE.match(pep)
    if m:
        pep = m.group(1)
    pep = _MOD_BRACKET_RE.sub("", pep)
    clean = _NON_AA_RE.sub("", pep).upper()
    return clean.replace("I", "L")


class CanonicalPeptideExclusionIndex(object):
    """Fast `I == L` mass-isobaric peptide exclusion index over `--proteome` and
    forward standard contaminants (`src/seqlib/contaminants.txt`)."""

    def __init__(
        self,
        proteome_fasta: str | pathlib.Path | None = None,
        *,
        include_contaminants: bool = True,
    ):
        chunks: list[str] = []
        if proteome_fasta is not None and pathlib.Path(proteome_fasta).is_file():
            for rec in SeqIO.parse(str(proteome_fasta), "fasta"):
                seq = str(rec.seq).upper().rstrip("*").replace("I", "L")
                if seq:
                    chunks.append(seq)
                    if len(seq) > 2 and seq[0] == "M" and seq[1] in _MET_EXCISE_NEXT:
                        chunks.append(seq[1:])

        if include_contaminants:
            pkg_contam = pathlib.Path(__file__).resolve().parents[1] / "seqlib" / "contaminants.txt"
            contam_path = (
                pkg_contam
                if pkg_contam.is_file()
                else REPO_ROOT / "src" / "seqlib" / "contaminants.txt"
            )
            if contam_path.is_file():
                for rec in SeqIO.parse(str(contam_path), "fasta"):
                    if rec.id.lower().startswith(("reverse_", "decoy_")):
                        continue
                    seq = str(rec.seq).upper().rstrip("*").replace("I", "L")
                    if seq:
                        chunks.append(seq)

        self._corpus: str = "#".join(chunks)
        self._cache: dict[str, bool] = {}

    def contains_peptide(self, raw_peptide: str) -> bool:
        if not self._corpus:
            return False
        norm = normalize_isobaric_peptide(raw_peptide)
        if not norm:
            return False
        cached = self._cache.get(norm)
        if cached is not None:
            return cached
        hit = norm in self._corpus
        self._cache[norm] = hit
        return hit


def _extract_token_locus(
    prot_token: str,
    coord_token: str,
    stringtie_dict: dict | None = None,
) -> tuple[str, str, int, int] | None:
    """Resolve `(contig_id, strand_sign, start, end)` for a candidate ORF token."""
    contig = "chr_default"
    strand = "+"
    lo: int | None = None
    hi: int | None = None

    m_coord = _COORD_PARSE_RE.match(coord_token.strip())
    if m_coord:
        if m_coord.group("contig"):
            contig = m_coord.group("contig")
        c_lo = int(m_coord.group("start"))
        c_hi = int(m_coord.group("end"))
        lo, hi = (c_lo, c_hi) if c_lo <= c_hi else (c_hi, c_lo)
        if m_coord.group("strand"):
            strand = m_coord.group("strand")

    g_parsed = parse_gorf_header(prot_token)
    if g_parsed is not None:
        g_contig, g_start, g_end, g_strand = g_parsed
        if g_contig:
            contig = g_contig
        strand = "-" if g_strand == "reverse" else "+"
        if lo is None or hi is None:
            lo, hi = (g_start, g_end) if g_start <= g_end else (g_end, g_start)
        return contig, strand, lo, hi

    t_parsed = parse_torf_header(prot_token)
    if t_parsed is not None:
        tx_id, tx_start, tx_end, orf_strand = t_parsed
        tx_obj = None
        if stringtie_dict:
            tx_obj = stringtie_dict.get(tx_id)
            if tx_obj is None and tx_id.startswith("uproteins."):
                short_id = ".".join(tx_id.split(".")[:2])
                tx_obj = stringtie_dict.get(short_id)
        if tx_obj is not None:
            if getattr(tx_obj, "chromosome", None):
                contig = str(tx_obj.chromosome)
            tx_strand = str(getattr(tx_obj, "strand", "+"))
            if tx_strand in ("+", "-"):
                strand = (
                    tx_strand
                    if orf_strand == "forward"
                    else ("-" if tx_strand == "+" else "+")
                )
            else:
                strand = "-" if orf_strand == "reverse" else "+"
        else:
            strand = "-" if orf_strand == "reverse" else "+"
        if lo is None or hi is None:
            lo, hi = (tx_start, tx_end) if tx_start <= tx_end else (tx_end, tx_start)
        return contig, strand, lo, hi

    if lo is not None and hi is not None:
        return contig, strand, lo, hi
    return None


def cluster_genomic_loci(
    loci: list[tuple[str, str, int, int]]
) -> list[tuple[str, str, int, int]]:
    """Merge overlapping coding intervals on the same `(contig, strand)` so
    nested start candidates or alternative transcript isoforms (`uproteins.10.1`,
    `uproteins.10.2`) sharing the same genomic ORF count as **1 genomic locus**,
    while duplicated paralogs at disjoint coordinates or on different contigs/strands
    remain distinct loci."""
    if not loci:
        return []
    by_chr_strand: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for contig, strand, lo, hi in loci:
        by_chr_strand.setdefault((contig, strand), []).append((lo, hi))

    merged_all: list[tuple[str, str, int, int]] = []
    for (contig, strand), intervals in sorted(by_chr_strand.items()):
        intervals.sort()
        cur_lo, cur_hi = intervals[0]
        for nxt_lo, nxt_hi in intervals[1:]:
            if nxt_lo <= cur_hi:
                cur_hi = max(cur_hi, nxt_hi)
            else:
                merged_all.append((contig, strand, cur_lo, cur_hi))
                cur_lo, cur_hi = nxt_lo, nxt_hi
        merged_all.append((contig, strand, cur_lo, cur_hi))
    return merged_all


def compute_paralog_group_id(clustered_loci: list[tuple[str, str, int, int]]) -> str:
    if len(clustered_loci) <= 1:
        return "SINGLE_LOCUS"
    sig = ";".join(f"{c}:{s}:{lo}-{hi}" for c, s, lo, hi in clustered_loci)
    digest = hashlib.blake2b(sig.encode("utf-8"), digest_size=4).hexdigest()
    return f"PARALOG_{digest}"


class PercolatorUTP(object):
    """Filter novel ORF PSMs for `I == L` canonical proteome/contaminant exclusion,
    collapse redundant transcript isoforms per genomic coding interval, and annotate
    single-locus vs. paralogous smORF families."""

    def __init__(
        self,
        coord_df=None,
        pep: float = 1.0,
        qvalue: float = 0.01,
        *,
        proteome_fasta: str | pathlib.Path | None = None,
        stringtie_dict: dict | None = None,
        max_paralog_loci: int = 10,
        unique_loci_only: bool = False,
        include_contaminants: bool = True,
    ):
        self.df = (
            pd.read_csv(coord_df, sep="\t")
            if not isinstance(coord_df, pd.DataFrame)
            else coord_df.copy()
        )
        self.pep = float(pep)
        self.qvalue = float(qvalue)
        self.max_paralog_loci = int(max_paralog_loci)
        self.unique_loci_only = bool(unique_loci_only)
        self.stringtie_dict = stringtie_dict or {}
        self.canonical_index = CanonicalPeptideExclusionIndex(
            proteome_fasta, include_contaminants=include_contaminants
        )

        self.unique: list[bool] = []
        self.locus_multiplicity: list[int] = []
        self.paralog_group_ids: list[str] = []
        self._prepare()

    def _prepare(self):
        if self.df.empty:
            return

        self.df = self.df[
            self.df["Genome Coordinates"].astype(str) != "Genome Coordinates"
        ].copy()
        self.df = self.df[
            ~self.df["Genome Coordinates"]
            .fillna("")
            .astype(str)
            .str.contains("not found", case=False, regex=False, na=False)
        ].copy()

        # Filter out decoy/contaminant/canonical proteinIds case-insensitively
        prot_s = self.df["proteinIds"].fillna("").astype(str)
        self.df = self.df[
            ~prot_s.str.contains("decoy_", case=False, regex=False, na=False)
            & ~prot_s.str.contains("reverse_", case=False, regex=False, na=False)
            & ~prot_s.str.contains("contaminant_", case=False, regex=False, na=False)
            & ~prot_s.str.contains("_ANNO", case=False, regex=False, na=False)
        ].copy()

        if self.df.empty:
            return

        # Apply FDR & PEP thresholds if not already filtered upstream
        if (
            "novel_psm_q_value" in self.df.columns
            and "novel_peptide_q_value" in self.df.columns
        ):
            self.df["novel_psm_q_value"] = pd.to_numeric(
                self.df["novel_psm_q_value"], errors="coerce"
            )
            self.df["novel_peptide_q_value"] = pd.to_numeric(
                self.df["novel_peptide_q_value"], errors="coerce"
            )
            q_mask = (self.df["novel_psm_q_value"] <= self.qvalue) & (
                self.df["novel_peptide_q_value"] <= self.qvalue
            )
        else:
            self.df["q-value"] = pd.to_numeric(self.df["q-value"], errors="coerce")
            q_mask = self.df["q-value"] <= self.qvalue

        if "posterior_error_prob" in self.df.columns:
            self.df["posterior_error_prob"] = pd.to_numeric(
                self.df["posterior_error_prob"], errors="coerce"
            )
            q_mask = q_mask & (self.df["posterior_error_prob"] <= self.pep)

        self.df = self.df[q_mask].copy()
        if self.df.empty:
            return

        # Exclude any PSM whose I/L-normalized peptide matches --proteome or contaminants
        is_canonical_pep = self.df["peptide"].map(self.canonical_index.contains_peptide)
        self.df = self.df[~is_canonical_pep].copy()

    def get_utps(self):
        self.unique = []
        self.locus_multiplicity = []
        self.paralog_group_ids = []

        if self.df.empty:
            return self

        deduped_prot_col: list[str] = []
        deduped_coord_col: list[str] = []

        for raw_prots, raw_coords in zip(
            self.df["proteinIds"].tolist(),
            self.df["Genome Coordinates"].tolist(),
        ):
            prot_tokens = [
                p.strip().split("(")[0].strip()
                for p in _PROTEIN_SPLIT_RE.split(str(raw_prots))
                if p.strip()
            ]
            coord_tokens = [c.strip() for c in str(raw_coords).split(",") if c.strip()]

            # Deduplicate multiple transcript isoforms mapping to the EXACT same
            # genomic coding interval `(contig, strand, lo, hi)` while keeping
            # distinct paralogous genomic loci!
            seen_exact_intervals: set[tuple[str, str, int, int]] = set()
            kept_prots: list[str] = []
            kept_coords: list[str] = []
            raw_loci: list[tuple[str, str, int, int]] = []

            for idx, p_tok in enumerate(prot_tokens):
                c_tok = (
                    coord_tokens[idx]
                    if idx < len(coord_tokens)
                    else (coord_tokens[0] if coord_tokens else "")
                )
                loc = _extract_token_locus(p_tok, c_tok, self.stringtie_dict)
                if loc is not None:
                    raw_loci.append(loc)
                    if loc in seen_exact_intervals:
                        continue
                    seen_exact_intervals.add(loc)
                kept_prots.append(p_tok)
                kept_coords.append(c_tok)

            clustered = cluster_genomic_loci(raw_loci)
            multiplicity = max(1, len(clustered))
            is_single_locus = multiplicity == 1
            group_id = compute_paralog_group_id(clustered)

            deduped_prot_col.append(",".join(kept_prots))
            deduped_coord_col.append(",".join(kept_coords))
            self.unique.append(is_single_locus)
            self.locus_multiplicity.append(multiplicity)
            self.paralog_group_ids.append(group_id)

        self.df["proteinIds"] = deduped_prot_col
        self.df["Genome Coordinates"] = deduped_coord_col
        return self

    def save_table(self, output="unique_peptides", keep="paralogs"):
        out_df = self.df.copy()
        out_df["Unique Peptide"] = pd.Series(self.unique, index=out_df.index, dtype=bool)
        out_df["Locus_Multiplicity"] = pd.Series(
            self.locus_multiplicity, index=out_df.index, dtype="int64"
        )
        out_df["Paralog_Group_ID"] = pd.Series(
            self.paralog_group_ids, index=out_df.index, dtype=object
        )

        if keep == "unique" or self.unique_loci_only:
            out_df = out_df[out_df["Locus_Multiplicity"] == 1].copy()
        elif keep == "paralogs":
            out_df = out_df[
                (out_df["Locus_Multiplicity"] >= 1)
                & (out_df["Locus_Multiplicity"] <= self.max_paralog_loci)
            ].copy()

        self.df = out_df
        out_path = str(output)
        if not out_path.endswith((".txt", ".tsv")):
            out_path = f"{out_path}.txt"
        out_df.to_csv(out_path, sep="\t", index=False)
        return self
