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


from copy import copy
import pathlib
import typing as t

from Bio import SeqIO

from ..sequtils.orflib import ORF, ORFCollection


def tryptic_peptides(
    protein: str,
    min_len: int = 6,
    max_missed_cleavages: int = 1,
    include_met_excision: bool = True,
) -> set[str]:
    """Compute in silico tryptic peptides (cleaving after K/R unless followed by P)."""
    seq = protein.upper()
    if not seq:
        return set()

    cut_sites = [0]
    for idx in range(len(seq) - 1):
        if seq[idx] in ("K", "R") and seq[idx + 1] != "P":
            cut_sites.append(idx + 1)
    cut_sites.append(len(seq))

    n_frags = len(cut_sites) - 1
    peptides: set[str] = set()
    for i in range(n_frags):
        for mc in range(max_missed_cleavages + 1):
            j = i + 1 + mc
            if j <= n_frags:
                pep = seq[cut_sites[i] : cut_sites[j]]
                if len(pep) >= min_len:
                    peptides.add(pep)
                if (
                    i == 0
                    and include_met_excision
                    and len(pep) >= min_len + 1
                    and pep[0] == "M"
                    and pep[1] in "ACGPSTV"
                ):
                    peptides.add(pep[1:])
    return peptides


def n_terminal_tryptic_peptides(
    protein: str,
    min_len: int = 6,
    max_missed_cleavages: int = 0,
    include_met_excision: bool = True,
) -> set[str]:
    """Return N-terminal tryptic peptides starting at residue 0 (or 1 after Met excision)."""
    seq = protein.upper()
    if not seq:
        return set()

    cut_sites = [0]
    for idx in range(len(seq) - 1):
        if seq[idx] in ("K", "R") and seq[idx + 1] != "P":
            cut_sites.append(idx + 1)
            if len(cut_sites) > max_missed_cleavages + 1:
                break
    if cut_sites[-1] != len(seq) and len(cut_sites) <= max_missed_cleavages + 1:
        cut_sites.append(len(seq))

    n_frags = len(cut_sites) - 1
    peptides: set[str] = set()
    for mc in range(min(n_frags, max_missed_cleavages + 1)):
        pep = seq[0 : cut_sites[mc + 1]]
        if len(pep) >= min_len:
            peptides.add(pep)
        if (
            include_met_excision
            and len(pep) >= min_len + 1
            and pep[0] == "M"
            and pep[1] in "ACGPSTV"
        ):
            peptides.add(pep[1:])
    return peptides


def first_tryptic_end_at_or_after(protein: str, start_idx: int) -> int:
    """Return slice end right after the first K/R (not before P) at index >= start_idx."""
    for idx in range(start_idx, len(protein) - 1):
        if protein[idx] in ("K", "R") and protein[idx + 1] != "P":
            return idx + 1
    return len(protein)


class CanonicalProteomeMatcher(object):
    """Cross-checks predicted ORFs against a canonical reference proteome (--proteome).

    1. Excludes ORFs identical to an annotated protein or representing an in-frame
       C-terminal truncation (`orf.seq[1:]` is an exact suffix of an annotated protein),
       eliminating internal `GTG`/`TTG`/`ATT`/`CTG` -> `M` fragment bloat from canonical CDSs.
    2. Detects upstream in-frame readthroughs across canonical `GTG`/`TTG`-initiated
       proteins (`orf.seq[-L+1:] == anno_seq[1:]`):
       - If the upstream prefix ends with a tryptic `K`/`R` cleavage right before the
         canonical start codon, trims strictly before the canonical start so zero-upstream
         `V`/`L`-substituted canonical N-terminal peptides cannot masquerade as novel smORFs.
       - Otherwise trims right after the first canonical tryptic `K`/`R` so only the
         upstream-spanning N-terminal-extension junction peptide is retained.
    """

    def __init__(
        self,
        proteome_fasta: t.Optional[t.Union[str, pathlib.Path]],
        max_orf_aa: int = 100,
        min_orf_aa: int = 6,
    ):
        self.max_orf_aa = max(int(max_orf_aa), 1)
        self.min_orf_aa = max(int(min_orf_aa), 1)
        self._exact_seqs: set[str] = set()
        self._cterm_suffixes: set[str] = set()
        self._bodies_by_len: dict[int, set[str]] = {}

        if proteome_fasta is not None and pathlib.Path(proteome_fasta).is_file():
            self._index_proteome(pathlib.Path(proteome_fasta))

    def _index_proteome(self, path: pathlib.Path) -> None:
        max_suffix_len = self.max_orf_aa + 2
        for record in SeqIO.parse(str(path), "fasta"):
            seq = "".join(str(record.seq).split()).rstrip("*").upper()
            if not seq:
                continue
            self._exact_seqs.add(seq)

            # Index C-terminal suffixes of length >= max(4, min_orf_aa - 1) up to max_orf_aa
            min_suf = max(4, self.min_orf_aa - 1)
            upper_suf = min(len(seq), max_suffix_len)
            for k in range(min_suf, upper_suf + 1):
                self._cterm_suffixes.add(seq[-k:])

            # Index post-initiator bodies (seq[1:]) of annotated proteins shorter than max_orf_aa
            # to catch upstream readthroughs across canonical GTG/TTG/ATT/CTG/ATG starts
            if len(seq) >= 5 and (len(seq) - 1) < self.max_orf_aa:
                body = seq[1:]
                self._bodies_by_len.setdefault(len(body), set()).add(body)

    def is_canonical_or_cterm_fragment(self, protein: str) -> bool:
        seq = protein.upper()
        if seq in self._exact_seqs:
            return True
        if len(seq) >= 5 and seq[1:] in self._cterm_suffixes:
            return True
        return False

    def filter_orfs(
        self,
        orfs: t.Iterable[ORF],
        contig_lengths: t.Optional[dict[int, int]] = None,
    ) -> ORFCollection:
        kept: list[ORF] = []
        sorted_body_lens = sorted(self._bodies_by_len.keys(), reverse=True)

        for orf in orfs:
            seq = str(orf.seq or "").upper()
            if not seq:
                continue

            if self.is_canonical_or_cterm_fragment(seq):
                continue

            seq_len = len(seq)
            trimmed_seq: t.Optional[str] = seq

            # Check for upstream readthrough into a canonical protein sharing the stop codon
            for body_len in sorted_body_lens:
                anno_full_len = body_len + 1
                if anno_full_len >= seq_len:
                    continue
                if seq[-body_len:] in self._bodies_by_len[body_len]:
                    upstream_len = seq_len - anno_full_len
                    upstream = seq[:upstream_len]
                    canonical_start_aa = seq[upstream_len]
                    if upstream[-1] in ("K", "R") and canonical_start_aa != "P":
                        # Tryptic cut immediately precedes canonical start codon:
                        # keep ONLY strictly upstream residues so V/L-substituted canonical
                        # N-terminal peptide is never emitted without upstream sequence!
                        trimmed_seq = upstream
                    else:
                        # Genuine spanning N-terminal junction peptide carries >=1 upstream residue
                        cut_end = first_tryptic_end_at_or_after(seq, upstream_len)
                        trimmed_seq = seq[:cut_end]
                    break

            if trimmed_seq is None or len(trimmed_seq) < self.min_orf_aa:
                continue

            if len(trimmed_seq) != seq_len:
                updated_orf = copy(orf)
                keep_nt = len(trimmed_seq) * 3
                updated_orf.seq = trimmed_seq
                if updated_orf.cds:
                    updated_orf.cds = updated_orf.cds[:keep_nt]
                updated_orf.end = self._adjust_3prime_end(
                    orf=updated_orf,
                    trimmed_nt=keep_nt,
                    contig_lengths=contig_lengths,
                )
                kept.append(updated_orf)
            else:
                kept.append(orf)

        return ORFCollection().add_orfs(kept)

    @staticmethod
    def _adjust_3prime_end(
        orf: ORF,
        trimmed_nt: int,
        contig_lengths: t.Optional[dict[int, int]] = None,
    ) -> int:
        chrom_key = orf.chromosome if orf.chromosome is not None else orf.transcript
        seq_len = (
            contig_lengths.get(int(chrom_key))
            if contig_lengths and chrom_key is not None
            else None
        )
        start_1b = int(orf.start)
        if orf.strand == "reverse":
            raw_end = start_1b - trimmed_nt + 1
            if seq_len and raw_end <= 0:
                raw_end += seq_len
            return raw_end
        else:
            raw_end = start_1b + trimmed_nt - 1
            if seq_len and raw_end > seq_len:
                raw_end = ((raw_end - 1) % seq_len) + 1
            return raw_end
