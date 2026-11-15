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

"""Pure-Python Xia (1998) / Mathews & Turner (1999) nearest-neighbor RNA-RNA
bimolecular duplex free-energy (dG_37, kcal/mol) calculator and 21-nt upstream
Shine-Dalgarno sequence extraction utilities."""

from functools import lru_cache
import pathlib
import warnings

from Bio import SeqIO

from ..sequtils.conversion import reverse_complement_dna


# Bimolecular initiation + terminal AU/GU penalties (kcal/mol at 37 C, 1 M Na+, Xia 1998 / Turner 1999)
BIMOLECULAR_INIT_DG: float = 4.09
TERMINAL_AU_GU_PENALTY: float = 0.45

# Thresholds (kcal/mol) matching Starmer et al. (2006) free2bind / uProteInS 2022 paper
STRONG_SD_THRESHOLD: float = -8.4
MODERATE_SD_THRESHOLD: float = -3.4535
MIN_LEADER_UTR_NT: int = 5

_VALID_RNA_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("A", "U"),
        ("U", "A"),
        ("G", "C"),
        ("C", "G"),
        ("G", "U"),
        ("U", "G"),
    }
)

_AU_OR_GU_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {
        ("A", "U"),
        ("U", "A"),
        ("G", "U"),
        ("U", "G"),
    }
)

# Nearest-neighbor stacking free energies dG_37 (kcal/mol) for
#   5' - x1 x2 - 3'
#   3' - y1 y2 - 5'
# Keyed as (x1, x2, y1, y2)
_STACK_DG_37: dict[tuple[str, str, str, str], float] = {
    # Watson-Crick stacks (Xia et al., 1998 Biochemistry 37:14719)
    ("A", "A", "U", "U"): -0.93,
    ("U", "U", "A", "A"): -0.93,
    ("A", "U", "U", "A"): -1.10,
    ("U", "A", "A", "U"): -1.33,
    ("C", "U", "G", "A"): -2.08,
    ("A", "G", "U", "C"): -2.08,
    ("C", "A", "G", "U"): -2.11,
    ("U", "G", "A", "C"): -2.11,
    ("G", "U", "C", "A"): -2.24,
    ("A", "C", "U", "G"): -2.24,
    ("G", "A", "C", "U"): -2.35,
    ("U", "C", "A", "G"): -2.35,
    ("C", "G", "G", "C"): -2.36,
    ("G", "C", "C", "G"): -3.42,
    ("G", "G", "C", "C"): -3.26,
    ("C", "C", "G", "G"): -3.26,
    # G-U wobble stacks (Mathews et al., 1999 J. Mol. Biol. 288:911)
    ("A", "G", "U", "U"): -0.55,
    ("U", "U", "G", "A"): -0.55,
    ("A", "U", "U", "G"): -1.36,
    ("G", "U", "U", "A"): -1.36,
    ("C", "G", "G", "U"): -1.41,
    ("U", "G", "G", "C"): -1.41,
    ("C", "U", "G", "G"): -2.11,
    ("G", "G", "U", "C"): -2.11,
    ("G", "G", "C", "U"): -1.53,
    ("U", "C", "G", "G"): -1.53,
    ("G", "U", "C", "G"): -2.51,
    ("G", "C", "U", "G"): -2.51,
    ("G", "A", "U", "U"): -1.27,
    ("U", "U", "A", "G"): -1.27,
    ("U", "G", "A", "U"): -1.00,
    ("U", "A", "G", "U"): -1.00,
    # Tandem G-U wobble stacks
    ("G", "U", "U", "G"): +1.29,
    ("U", "G", "G", "U"): +0.47,
    ("G", "G", "U", "U"): -0.50,
    ("U", "U", "G", "G"): -0.50,
}


def to_rna(seq: str) -> str:
    """Convert a DNA/RNA string to uppercase RNA (`T -> U`)."""
    return str(seq).upper().replace("T", "U").strip()


def _end_penalty(b1: str, b2: str) -> float:
    return TERMINAL_AU_GU_PENALTY if (b1, b2) in _AU_OR_GU_PAIRS else 0.0


@lru_cache(maxsize=16384)
def compute_rbs_free_energy(upstream_seq_5to3: str, anti_sd_3to5: str) -> float:
    """Compute the minimum bimolecular RNA-RNA hybridization free energy dG_37 (kcal/mol)
    between `upstream_seq_5to3` (5' -> 3' mRNA upstream window) and `anti_sd_3to5`
    (3' -> 5' 16S rRNA tail).

    Returns `0.0` when no thermodynamically favorable (`< 0.0`) duplex exists.
    """
    s1 = to_rna(upstream_seq_5to3)
    s2 = to_rna(anti_sd_3to5)
    m = len(s1)
    n = len(s2)
    if m < 2 or n < 2:
        return 0.0

    inf = 1e9
    # dp_single[i][j]: initiation energy of a single isolated base pair at (i, j)
    # dp_stem[i][j]: best energy ending at base pair (i, j) with >= 1 nearest-neighbor stack
    dp_single = [[inf] * n for _ in range(m)]
    dp_stem = [[inf] * n for _ in range(m)]
    best_dg = 0.0

    for i in range(m):
        b1 = s1[i]
        for j in range(n):
            b2 = s2[j]
            if (b1, b2) not in _VALID_RNA_PAIRS:
                continue

            ep_curr = _end_penalty(b1, b2)
            dp_single[i][j] = BIMOLECULAR_INIT_DG + ep_curr

            best_here = inf
            # 1. Contiguous nearest-neighbor stack from (i - 1, j - 1)
            if i > 0 and j > 0:
                p1 = s1[i - 1]
                p2 = s2[j - 1]
                if (p1, p2) in _VALID_RNA_PAIRS:
                    stack_dg = _STACK_DG_37.get((p1, b1, p2, b2))
                    if stack_dg is not None:
                        prev_best = min(dp_single[i - 1][j - 1], dp_stem[i - 1][j - 1])
                        if prev_best < inf:
                            cand = prev_best + stack_dg
                            if cand < best_here:
                                best_here = cand

            # 2. Small bulges (1..2 nt) and internal loops (1x1, 1x2, 2x2)
            #    extending an already-stacked stem (prev_stem < inf)
            for di, dj in ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
                pi = i - di
                pj = j - dj
                if pi < 0 or pj < 0:
                    continue
                prev_stem = dp_stem[pi][pj]
                if prev_stem >= inf:
                    continue
                p1 = s1[pi]
                p2 = s2[pj]
                if (di == 1 and dj == 2) or (di == 2 and dj == 1):
                    # Single-nucleotide bulge (Turner 1999: +3.8 kcal/mol)
                    loop_pen = 3.80
                elif (di == 1 and dj == 3) or (di == 3 and dj == 1):
                    # 2-nt bulge
                    loop_pen = 2.80 + _end_penalty(p1, p2) + ep_curr
                elif di == 2 and dj == 2:
                    # 1x1 internal loop (stabilized for G-G or U-U mismatches)
                    mm = (s1[i - 1], s2[j - 1])
                    base_1x1 = 0.40 if mm in (("G", "G"), ("U", "U")) else 1.10
                    loop_pen = base_1x1 + _end_penalty(p1, p2) + ep_curr
                elif (di == 2 and dj == 3) or (di == 3 and dj == 2):
                    # 1x2 internal loop
                    loop_pen = 2.10 + _end_penalty(p1, p2) + ep_curr
                else:
                    # 2x2 internal loop
                    loop_pen = 2.20 + _end_penalty(p1, p2) + ep_curr

                cand = prev_stem + loop_pen
                if cand < best_here:
                    best_here = cand

            dp_stem[i][j] = best_here
            if best_here < inf:
                closed = best_here + ep_curr
                if closed < best_dg:
                    best_dg = closed

    return round(best_dg, 4) if best_dg < 0.0 else 0.0


def extract_16s_anti_sd_tail(
    rrna_fasta: str | pathlib.Path,
    tail_length: int = 13,
) -> str:
    """Select the 16S rRNA record from `rrna_fasta` and return its 3' tail in
    `3' -> 5'` orientation (length `tail_length`, default 13 nt).

    Scans all FASTA records so multi-record NCBI `rna_from_genomic.fna` files
    select 16S rRNA containing the conserved `CCUCC` anti-SD core rather than
    blindly taking record 0 (often 5S rRNA or tRNA).
    """
    records = list(SeqIO.parse(str(rrna_fasta), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in --rrna file: {rrna_fasta}")

    scored: list[tuple[int, int, str]] = []
    for idx, rec in enumerate(records):
        seq_rna = to_rna(str(rec.seq))
        if not seq_rna:
            continue
        header = f"{rec.id} {rec.description}".lower()
        is_16s_header = ("16s" in header) or ("rrs" in header) or ("small subunit" in header)
        tail_30 = seq_rna[-30:]
        has_anti_sd = "CCUCC" in tail_30
        rank = (
            (2 if has_anti_sd else 0)
            + (1 if is_16s_header else 0)
        )
        scored.append((rank, -idx, seq_rna))

    if not scored:
        raise ValueError(f"All FASTA records in --rrna file are empty: {rrna_fasta}")

    scored.sort(reverse=True)
    best_rank, _neg_idx, chosen_seq = scored[0]
    if best_rank < 2 and "CCUCC" not in chosen_seq[-30:]:
        warnings.warn(
            f"--rrna sequence in {rrna_fasta} does not contain the canonical "
            "bacterial 3' anti-Shine-Dalgarno 'CCUCC' core in its 3' tail.",
            UserWarning,
            stacklevel=2,
        )

    # Reverse 5'->3' sequence to 3'->5' and take first `tail_length` bases from the 3' end
    return chosen_seq[::-1][:tail_length]


def extract_genomic_upstream_21nt(
    contig_seq: str,
    coord_a: int,
    coord_b: int,
    strand: str,
    *,
    circular: bool = True,
    window: int = 21,
) -> str:
    """Extract the exact `window` (default 21) nucleotides strictly 5' upstream of
    an ORF's start codon on `contig_seq` (1-based inclusive coordinates `[coord_a, coord_b]`).

    - `+` / `forward`: start codon begins at `lo = min(coord_a, coord_b)`.
      Strictly upstream bases are 1-based `[lo - window .. lo - 1]` -> 0-based `[lo - 1 - window : lo - 1]`.
    - `-` / `reverse`: start codon begins at `hi = max(coord_a, coord_b)` (reading 5'->3' on minus strand).
      Strictly upstream bases are 1-based `[hi + 1 .. hi + window]` -> 0-based `[hi : hi + window]`,
      reverse-complemented into 5'->3' mRNA sense.
    """
    seq_len = len(contig_seq)
    if seq_len == 0:
        return ""

    lo = min(int(coord_a), int(coord_b))
    hi = max(int(coord_a), int(coord_b))
    is_rev = str(strand).lower() in ("reverse", "-")

    if not is_rev:
        end_0 = max(0, min(seq_len, lo - 1))
        start_0 = end_0 - window
        if start_0 >= 0:
            return contig_seq[start_0:end_0]
        prefix = contig_seq[0:end_0]
        if circular and seq_len >= window:
            needed = window - len(prefix)
            return contig_seq[seq_len - needed :] + prefix
        return prefix
    else:
        start_0 = max(0, min(seq_len, hi))
        end_0 = start_0 + window
        if end_0 <= seq_len:
            raw_plus = contig_seq[start_0:end_0]
        else:
            suffix = contig_seq[start_0:seq_len]
            if circular and seq_len >= window:
                needed = window - len(suffix)
                raw_plus = suffix + contig_seq[:needed]
            else:
                raw_plus = suffix
        return reverse_complement_dna(raw_plus)


def extract_transcript_upstream(
    transcript_seq: str,
    coord_a: int,
    coord_b: int,
    orf_strand: str = "forward",
    *,
    window: int = 21,
) -> tuple[str, int]:
    """Extract up to `window` (21) nucleotides strictly upstream of a transcript ORF
    together with the true transcribed 5' UTR length (`utr_len`).

    Preserves `5..21` nt 5' UTRs instead of discarding `< 22` nt leaders!
    """
    tx_len = len(transcript_seq)
    if tx_len == 0:
        return "", 0

    lo = min(int(coord_a), int(coord_b))
    hi = max(int(coord_a), int(coord_b))
    is_rev = str(orf_strand).lower() in ("reverse", "-")

    if not is_rev:
        utr_len = max(0, lo - 1)
        if utr_len == 0:
            return "", 0
        end_0 = min(tx_len, lo - 1)
        start_0 = max(0, end_0 - window)
        return transcript_seq[start_0:end_0], utr_len
    else:
        utr_len = max(0, tx_len - hi)
        if utr_len == 0:
            return "", 0
        start_0 = min(tx_len, hi)
        end_0 = min(tx_len, start_0 + window)
        return reverse_complement_dna(transcript_seq[start_0:end_0]), utr_len


def classify_rbs_category(
    free_energy: float,
    *,
    is_transcriptome: bool = False,
    utr_length: int | None = None,
) -> str:
    """Classify Shine-Dalgarno status accurately:

    - True `"Leaderless"` is reserved strictly for assembled transcripts whose
      transcribed 5' leader is `< 5 nt` (`0..4` nt upstream of the start codon).
    - Otherwise (`Genome` mode without TSS coordinates, or transcripts with `>= 5 nt`
      5' UTR):
        - `dG <= -8.4`: `"Present. Strong binding."`
        - `-8.4 < dG <= -3.4535`: `"Present. Low to moderate binding."`
        - `dG > -3.4535`: `"None (SD absent)"`
    """
    if is_transcriptome and utr_length is not None and utr_length < MIN_LEADER_UTR_NT:
        return "Leaderless"

    dg = float(free_energy)
    if dg <= STRONG_SD_THRESHOLD:
        return "Present. Strong binding."
    if dg <= MODERATE_SD_THRESHOLD:
        return "Present. Low to moderate binding."
    return "None (SD absent)"
