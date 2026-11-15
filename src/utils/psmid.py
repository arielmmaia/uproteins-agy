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


_ROLE_BEFORE_SII_RE = re.compile(r"_(?:target|decoy)(?=_SII_)", re.IGNORECASE)
_ROLE_BEFORE_SCAN_RE = re.compile(r"_(?:target|decoy)(?=_\d+_\d+_\d+$)", re.IGNORECASE)


def normalize_pin_spec_id(spec_id: str) -> tuple[str, str]:
    """Strip `_target` / `_decoy` search-role suffixes from a PIN `SpecId` and
    return `(normalized_spec_id, run_stem)` so target and decoy hypotheses for the
    same MS2 scan share the exact same spectrum run stem."""
    norm = _ROLE_BEFORE_SII_RE.sub("", spec_id, count=1)
    if norm == spec_id:
        norm = _ROLE_BEFORE_SCAN_RE.sub("", spec_id, count=1)

    sii_idx = norm.rfind("_SII_")
    if sii_idx != -1:
        return norm, norm[:sii_idx]

    parts = norm.rsplit("_", 3)
    if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
        return norm, parts[0]

    return norm, norm


def parse_percolator_psmid(psm_id: str) -> tuple[str, int, int, int]:
    """Parse a Percolator / `msgf2pin` `PSMId` into `(spec_stem, scan_num, charge, rank)`.

    Handles all three `msgf2pin` formats safely (even when sample names contain `'SII'`
    or underscores):
      - `{spec_stem}_SII_{sii_idx}_{sii_rank}_{ScanNum}_{Charge}_{Rank}`
      - `{spec_stem}_{target|decoy}_SII_{sii_idx}_{sii_rank}_{ScanNum}_{Charge}_{Rank}`
      - `{spec_stem}_{ScanNum}_{Charge}_{Rank}` (modern 3-suffix `msgf2pin`)
    """
    norm_id, _ = normalize_pin_spec_id(str(psm_id).strip())
    parts = norm_id.rsplit("_", 3)
    if len(parts) < 4:
        raise ValueError(f"Malformed Percolator PSMId: {psm_id!r}")
    scan_num = int(parts[-3])
    charge = int(parts[-2])
    rank = int(parts[-1])

    sii_pos = norm_id.rfind("_SII_")
    if sii_pos != -1:
        spec_stem = norm_id[:sii_pos]
    else:
        spec_stem = parts[0]
    return spec_stem, scan_num, charge, rank
