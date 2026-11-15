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


from functools import lru_cache
import typing as t

from Bio.Data import CodonTable


_REASSIGNED_STOP_FALLBACK = {
    "TGA": "W",  # NCBI Table 4 (Mycoplasma/Spiroplasma/Entomoplasmatales)
    "TAA": "Q",  # Natural ochre reassignment fallback
    "TAG": "Q",  # Natural amber reassignment fallback
}


@lru_cache(maxsize=32)
def build_codon_table(
    stops: t.Tuple[str, ...] = ("TAA", "TAG", "TGA"),
    genetic_code: int = 11,
) -> dict[str, str]:
    """Build a DNA codon -> amino acid lookup honoring custom stops & NCBI code."""
    ncbi_table = CodonTable.unambiguous_dna_by_id[genetic_code]
    table: dict[str, str] = dict(ncbi_table.forward_table)

    stop_set = {c.upper() for c in stops}
    for codon in ("TAA", "TAG", "TGA"):
        if codon not in table and codon not in stop_set:
            table[codon] = _REASSIGNED_STOP_FALLBACK[codon]

    for stop_codon in stop_set:
        table[stop_codon] = "*"

    return table


class Translator(object):
    def __init__(
        self,
        genome: str,
        stops: t.Optional[t.Iterable[str]] = None,
        genetic_code: int = 11,
    ):
        self.genome = str(genome)
        if stops is None:
            ncbi_stops = CodonTable.unambiguous_dna_by_id[genetic_code].stop_codons
            self.stops = tuple(c.upper() for c in ncbi_stops)
        else:
            self.stops = tuple(c.upper() for c in stops)
        self.genetic_code = int(genetic_code)

    def translate(self, initiator: bool = True) -> str:
        table = build_codon_table(self.stops, self.genetic_code)
        stop_set = set(self.stops)
        seq = self.genome.upper().replace("U", "T")
        chars: list[str] = []
        for i in range(0, len(seq), 3):
            codon = seq[i : i + 3]
            if len(codon) == 3:
                if i == 0 and initiator and codon not in stop_set:
                    chars.append("M")
                else:
                    chars.append(table[codon])
        return "".join(chars)

    def complement(self) -> str:
        nuc_table = {
            'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'
        }
        return "".join(nuc_table[i] for i in self.genome)

