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


_IUPAC_SRC = "ACGTRYKMSWBDHVN-"
_IUPAC_DST = "TGCAYRMKSWVHDBN-"
_IUPAC_TRANS = str.maketrans(_IUPAC_SRC, _IUPAC_DST)
_NON_IUPAC_RE = re.compile(r"[^ACGTRYKMSWBDHVN\-]")


def normalize_dna(seq: str) -> str:
    """Normalize arbitrary DNA/RNA string (upper/lower, U->T, unknown->N)."""
    cleaned = str(seq).upper().replace("U", "T")
    return _NON_IUPAC_RE.sub("N", cleaned)


def complement_dna(seq: str) -> str:
    """Return the uppercase IUPAC DNA complement of *seq* without raising KeyError."""
    return normalize_dna(seq).translate(_IUPAC_TRANS)


def reverse_complement_dna(seq: str) -> str:
    """Return the uppercase IUPAC DNA reverse-complement of *seq*."""
    return complement_dna(seq)[::-1]


class StrandConverter(object):
    def __init__(self, sequence):
        self.sequence = sequence
        self.complementary = []

    def complement(self):
        self.complementary = []
        for chromosome in self.sequence:
            chrom_str = str(chromosome)
            if "*" in chrom_str:
                continue
            self.complementary.append(complement_dna(chrom_str))
        return self

    def reverse(self):
        return [seq[::-1] for seq in self.complementary]
