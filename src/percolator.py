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

from .database.proteome_matcher import tryptic_peptides
from .utils import REPO_ROOT, ensure_dir


# Non-isobaric residue swaps that never introduce/destroy tryptic K/R/P sites or touch B/X/Z
_COLLISION_SWAP: dict[str, str] = {
    "A": "G",
    "G": "A",
    "V": "T",
    "T": "V",
    "L": "F",
    "I": "F",
    "F": "L",
    "S": "N",
    "N": "S",
    "D": "Q",
    "E": "Q",
    "Q": "D",
    "C": "S",
    "M": "V",
    "Y": "W",
    "W": "Y",
    "H": "N",
}


def _norm_il(seq: str) -> str:
    return seq.upper().replace("I", "L")


def split_tryptic_segments(seq: str) -> list[str]:
    """Split a protein sequence into tryptic segments (cutting after K/R unless followed by P)."""
    if not seq:
        return []
    segments: list[str] = []
    start = 0
    for idx in range(len(seq) - 1):
        if seq[idx] in ("K", "R") and seq[idx + 1] != "P":
            segments.append(seq[start : idx + 1])
            start = idx + 1
    segments.append(seq[start:])
    return segments


def _pseudo_reverse_segment(seg: str, target_il_peptides: set[str], is_nterm: bool = False) -> str:
    """Reverse internal residues of a single tryptic segment while keeping C-terminal residue fixed,
    and deterministically break palindromes or I==L collisions against target peptides."""
    if len(seg) <= 1:
        return seg

    core = seg[:-1]
    cterm = seg[-1]
    rev_core = core[::-1]
    if len(core) >= 2 and rev_core == core:
        rev_core = rev_core[1:] + rev_core[:1]

    candidate = rev_core + cterm
    if len(candidate) < 6:
        return candidate

    def _collides(cand: str) -> bool:
        if _norm_il(cand) in target_il_peptides:
            return True
        if (
            is_nterm
            and len(cand) >= 7
            and cand[0] == "M"
            and cand[1] in "ACGPSTV"
            and _norm_il(cand[1:]) in target_il_peptides
        ):
            return True
        return False

    if not _collides(candidate):
        return candidate

    # Try deterministic cyclic shifts of the internal core
    for shift in range(1, len(rev_core)):
        rotated = rev_core[shift:] + rev_core[:shift] + cterm
        if not _collides(rotated):
            return rotated

    # Fallback for homopolymeric/symmetric cores (e.g. AAAAAAR): deterministic non-isobaric residue swap
    mutated = list(rev_core)
    for idx in range(0, len(mutated), 2):
        res = mutated[idx]
        if res in _COLLISION_SWAP:
            mutated[idx] = _COLLISION_SWAP[res]
            cand_mut = "".join(mutated) + cterm
            if not _collides(cand_mut):
                return cand_mut

    return "".join(mutated) + cterm


def pseudo_reverse_tryptic(seq: str, target_il_peptides: set[str]) -> str:
    """Generate a tryptic internal-sequence pseudo-reversed decoy free of >=6 aa (I==L) target collisions."""
    clean = "".join(str(seq).split()).rstrip("*").upper()
    if len(clean) <= 1:
        return clean

    segments = split_tryptic_segments(clean)
    decoy_segs = [
        _pseudo_reverse_segment(seg, target_il_peptides, is_nterm=(idx == 0))
        for idx, seg in enumerate(segments)
    ]

    # Also verify 1-missed-cleavage adjacent pairs do not collide with target peptides
    for idx in range(len(decoy_segs) - 1):
        pair = decoy_segs[idx] + decoy_segs[idx + 1]
        if len(pair) >= 6 and _norm_il(pair) in target_il_peptides:
            seg = decoy_segs[idx]
            if len(seg) >= 2:
                core = list(seg[:-1])
                for pos in range(len(core)):
                    if core[pos] in _COLLISION_SWAP:
                        core[pos] = _COLLISION_SWAP[core[pos]]
                        break
                decoy_segs[idx] = "".join(core) + seg[-1]

    return "".join(decoy_segs)


class Decoy(object):
    """Generates a 1:1 symmetric tryptic pseudo-reversed decoy FASTA from a target FASTA."""

    def __init__(self, db, db_type):
        self.df = db
        self.seqs, self.entries = self.__get_seqs()
        self.reversed: list[str] = []
        self.type = db_type
        self.__create_dir()
        self.path = str(REPO_ROOT)

    def __create_dir(self):
        ensure_dir(f"{self.type}/Percolator")

    def __get_seqs(self) -> tuple[list[str], list[str]]:
        seqs: list[str] = []
        entries: list[str] = []
        for record in SeqIO.parse(self.df, 'fasta'):
            rec_id = str(record.id)
            # Never double-decoy or ingest pre-baked Reverse_ entries if present
            if rec_id.startswith("decoy_") or rec_id.startswith("Reverse_"):
                continue
            clean_seq = "".join(str(record.seq).split()).rstrip("*").upper()
            if not clean_seq:
                continue
            seqs.append(clean_seq)
            entries.append(str(record.description))
        return seqs, entries

    def reverse_sequences(self):
        """Pseudo-reverses internal tryptic segments while keeping C-terminal K/R fixed
        and masking any >=6 aa (I==L) collisions with target tryptic peptides."""
        target_il_peptides: set[str] = set()
        for seq in self.seqs:
            for pep in tryptic_peptides(seq, min_len=6, max_missed_cleavages=1, include_met_excision=True):
                target_il_peptides.add(_norm_il(pep))

        self.reversed = [
            pseudo_reverse_tryptic(seq, target_il_peptides)
            for seq in self.seqs
        ]
        return self

    def to_fasta(self):
        out = [
            f">decoy_{self.entries[i]}\n{self.reversed[i]}\n"
            for i in range(len(self.reversed))
        ]
        with open(f"{self.type}/Percolator/{self.type}_decoy.fasta", 'w') as fa:
            fa.writelines(out)
        return self
