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
import shutil

from ..utils import ensure_dir, resolve_executable, run_cmd


# MS-GF+ `-e` integer codes (0..9) -> msgf2pin `-e` and percolator `--protein-enzyme`
MSGF_ENZYME_TO_MSGF2PIN: dict[str, str] = {
    "0": "unspecific",
    "1": "trypsin",
    "2": "chymotrypsin",
    "3": "lys-c",
    "4": "lys-n",
    "5": "glu-c",
    "6": "arg-c",
    "7": "asp-n",
    "8": "alpha_lp",
    "9": "no_enzyme",
}

MSGF_ENZYME_TO_PERCOLATOR: dict[str, str] = {
    "0": "no_enzyme",
    "1": "trypsin",
    "2": "chymotrypsin",
    "3": "lys-c",
    "4": "lys-n",
    "5": "glu-c",
    "6": "arg-c",
    "7": "asp-n",
    "8": "elastase",
    "9": "no_enzyme",
}

# Reverse mapping when user passes --enzyme name directly without --e
ENZYME_NAME_TO_MSGF2PIN: dict[str, str] = {
    "trypsin": "trypsin",
    "trypsinp": "trypsinp",
    "chymotrypsin": "chymotrypsin",
    "lys-c": "lys-c",
    "lys-n": "lys-n",
    "glu-c": "glu-c",
    "arg-c": "arg-c",
    "asp-n": "asp-n",
    "elastase": "elastase",
    "alpha_lp": "alpha_lp",
    "no_enzyme": "no_enzyme",
    "unspecific": "unspecific",
}

_DBSEQ_ACCESSION_RE = re.compile(r'<DBSequence\b[^>]*\baccession="([^"]+)"', re.IGNORECASE)
_ENZYME_XML_HINTS: tuple[tuple[str, str], ...] = (
    ("unspecific", "0"),
    ("no cleavage", "9"),
    ("nocleavage", "9"),
    ("chymotrypsin", "2"),
    ("lys-c", "3"),
    ("lysc", "3"),
    ("lys-n", "4"),
    ("lysn", "4"),
    ("glu-c", "5"),
    ("glutamyl", "5"),
    ("v8-de", "5"),
    ("arg-c", "6"),
    ("argc", "6"),
    ("asp-n", "7"),
    ("aspn", "7"),
    ("alphalp", "8"),
    ("alpha_lp", "8"),
    ("trypsin", "1"),
)

_ROLE_BEFORE_SII_RE = re.compile(r"_(?:target|decoy)(?=_SII_)", re.IGNORECASE)
_ROLE_BEFORE_SCAN_RE = re.compile(r"_(?:target|decoy)(?=_\d+_\d+_\d+$)", re.IGNORECASE)

# Search-engine primary score columns in priority order: (column_name, default_higher_is_better)
_TDC_SCORE_CANDIDATES: tuple[tuple[str, int], ...] = (
    ("lnSpecEValue", 1),
    ("SpecEValue", -1),
    ("lnEValue", 1),
    ("EValue", -1),
    ("MS8_feature_20", 1),
    ("RawScore", 1),
    ("DeNovoScore", 1),
)


def classify_mzid_file(mzid_path: str | pathlib.Path) -> str:
    """Classify an `.mzid` file strictly as `'target'`, `'decoy'`, or `'concatenated'`.

    1. First checks exact filename suffixes (`_decoy.mzid`, `_target.mzid`,
       `_concatenated.mzid`, `_concat.mzid`) — never substring `'decoy' in filename`.
    2. For legacy `.mzid` files lacking an explicit role suffix, inspects up to 500
       `<DBSequence accession="...">` entries in the XML header.
    """
    path = pathlib.Path(mzid_path)
    lower_name = path.name.lower()
    if not lower_name.endswith(".mzid"):
        raise ValueError(f"Not an .mzid file: {path}")

    if lower_name.endswith("_decoy.mzid"):
        return "decoy"
    if lower_name.endswith("_target.mzid"):
        return "target"
    if lower_name.endswith(("_concat.mzid", "_concatenated.mzid")):
        return "concatenated"

    # Legacy fallback: inspect XML <DBSequence accession="...">
    saw_target = False
    saw_decoy = False
    seen = 0
    if path.is_file():
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for match in _DBSEQ_ACCESSION_RE.finditer(line):
                    acc = match.group(1)
                    if acc.lower().startswith(("decoy_", "reverse_")):
                        saw_decoy = True
                    else:
                        saw_target = True
                    seen += 1
                    if (saw_target and saw_decoy) or seen >= 500:
                        break
                if (saw_target and saw_decoy) or seen >= 500:
                    break

    if saw_target and saw_decoy:
        return "concatenated"
    if saw_decoy and not saw_target:
        return "decoy"
    return "target"


def detect_enzyme_code_from_mzid(mzid_path: str | pathlib.Path) -> str | None:
    """Inspect `<Enzymes>` metadata in an `.mzid` file to recover MS-GF+ `-e` code."""
    path = pathlib.Path(mzid_path)
    if not path.is_file():
        return None
    in_enzymes = False
    buf: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh):
            if "<Enzymes" in line or "<Enzyme " in line or "<Enzyme>" in line:
                in_enzymes = True
            if in_enzymes:
                buf.append(line.lower())
                if "</Enzymes>" in line or "</Enzyme>" in line or len(buf) >= 40:
                    break
            if idx > 2000 and not in_enzymes:
                break
    if not buf:
        return None
    joined = " ".join(buf)
    for hint, code in _ENZYME_XML_HINTS:
        if hint in joined:
            return code
    return None


def resolve_enzyme_pair(
    args=None, sample_mzid: str | pathlib.Path | None = None
) -> tuple[str, str]:
    """Return `(msgf2pin_enzyme, percolator_enzyme)` aligned with Stage 3 MS-GF+."""
    e_code = getattr(args, "e", None) if args is not None else None
    explicit_perc_enz = getattr(args, "enzyme", None) if args is not None else None

    if e_code is not None:
        code_str = str(e_code).strip()
        if code_str not in MSGF_ENZYME_TO_MSGF2PIN:
            raise ValueError(
                f"Unsupported MS-GF+ --e enzyme code {code_str!r}; expected 0..9."
            )
        pin_enz = MSGF_ENZYME_TO_MSGF2PIN[code_str]
        perc_enz = explicit_perc_enz or MSGF_ENZYME_TO_PERCOLATOR[code_str]
        return pin_enz, perc_enz

    if explicit_perc_enz:
        norm = str(explicit_perc_enz).strip().lower()
        pin_enz = ENZYME_NAME_TO_MSGF2PIN.get(norm, "trypsin")
        return pin_enz, str(explicit_perc_enz).strip()

    if sample_mzid is not None:
        detected = detect_enzyme_code_from_mzid(sample_mzid)
        if detected is not None:
            return (
                MSGF_ENZYME_TO_MSGF2PIN[detected],
                MSGF_ENZYME_TO_PERCOLATOR[detected],
            )

    return "trypsin", "trypsin"


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


def _tiebreak_digest(seed: int, run_stem: str, scan_nr: str, label: int) -> bytes:
    payload = f"{seed}:{run_stem}:{scan_nr}:{label}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).digest()


def compete_target_decoy_pin(
    pin_path: str | pathlib.Path,
    *,
    seed: int = 42,
    save_uncompeted_backup: bool = True,
) -> pathlib.Path:
    """Apply exact per-scan top-1 Target-Decoy Competition (TDC) to a PIN file.

    - Normalizes `{stem}_{target|decoy}_SII_...` `SpecId`s to `{stem}_SII_...`.
    - Archives the raw pre-TDC PIN at `{stem}_uncompeted.txt` when requested.
    - Groups all PSM rows by `(run_stem, ScanNr)` and retains strictly the single
      best-scoring hypothesis across target (`Label == 1`) and decoy (`Label == -1`).
    - Breaks exact score ties deterministically and unbiasedly (50/50) via BLAKE2b.
    """
    pin = pathlib.Path(pin_path)
    if not pin.is_file():
        return pin

    with pin.open("r", encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    if not raw_lines:
        return pin

    if save_uncompeted_backup:
        backup = pin.with_name(f"{pin.stem}_uncompeted{pin.suffix}")
        shutil.copy2(pin, backup)

    header_line = raw_lines[0]
    headers = header_line.rstrip("\r\n").split("\t")
    col_idx = {name: idx for idx, name in enumerate(headers)}

    spec_id_col = col_idx.get("SpecId", 0)
    label_col = col_idx.get("Label", 1)
    scan_col = col_idx.get("ScanNr", 2)

    data_start = 1
    default_dir_line: str | None = None
    explicit_directions: dict[str, int] = {}
    if len(raw_lines) > 1:
        second_cols = raw_lines[1].rstrip("\r\n").split("\t")
        if second_cols and second_cols[0] == "DefaultDirection":
            default_dir_line = raw_lines[1]
            data_start = 2
            for name, val in zip(headers, second_cols):
                try:
                    fval = float(val)
                    if fval > 0:
                        explicit_directions[name] = 1
                    elif fval < 0:
                        explicit_directions[name] = -1
                except ValueError:
                    continue

    score_col: int | None = None
    score_sign: int = 1
    for cand_name, default_sign in _TDC_SCORE_CANDIDATES:
        if cand_name in col_idx:
            score_col = col_idx[cand_name]
            score_sign = explicit_directions.get(cand_name, default_sign)
            break

    # Best entry per (run_stem, scan_nr):
    # key -> (directed_score, tiebreak_bytes, first_row_order, normalized_line)
    winners: dict[tuple[str, str], tuple[float, bytes, int, str]] = {}
    line_ending = "\n"

    for row_order, line in enumerate(raw_lines[data_start:]):
        stripped = line.rstrip("\r\n")
        if not stripped:
            continue
        cols = stripped.split("\t")
        if cols[0] == "SpecId":
            continue

        norm_spec_id, run_stem = normalize_pin_spec_id(cols[spec_id_col])
        cols[spec_id_col] = norm_spec_id
        scan_nr = cols[scan_col]
        try:
            label_int = int(cols[label_col])
        except ValueError:
            label_int = 1

        if score_col is not None and score_col < len(cols):
            try:
                directed_score = float(cols[score_col]) * score_sign
            except ValueError:
                directed_score = float("-inf")
        else:
            directed_score = 0.0

        tb = _tiebreak_digest(seed, run_stem, scan_nr, label_int)
        norm_line = "\t".join(cols) + line_ending
        scan_key = (run_stem, scan_nr)

        prev = winners.get(scan_key)
        if prev is None or (directed_score, tb) > (prev[0], prev[1]):
            winners[scan_key] = (directed_score, tb, row_order, norm_line)

    ordered_winners = sorted(winners.values(), key=lambda item: item[2])
    with pin.open("w", encoding="utf-8") as out_fh:
        out_fh.write(header_line if header_line.endswith("\n") else header_line + "\n")
        if default_dir_line is not None:
            out_fh.write(
                default_dir_line if default_dir_line.endswith("\n") else default_dir_line + "\n"
            )
        for _, _, _, norm_line in ordered_winners:
            out_fh.write(norm_line)

    return pin


class PercolatorProcessing(object):
    def __init__(self, folder, filetype, args=None):
        self.folder = folder
        self.args = args
        self.target = None
        self.decoy = None
        self.concatenated_mzids: list[str] = []
        self.sample_mzid: pathlib.Path | None = None
        self.filetype = filetype

    def create_metafiles(self):
        """Create target and decoy `.mzid` metafiles strictly by `.mzid` extension and
        `_target.mzid` / `_decoy.mzid` suffix (with XML `<DBSequence>` fallback)."""
        folder_path = pathlib.Path(self.folder)
        real: list[str] = []
        decoy: list[str] = []
        concat: list[str] = []

        for entry in sorted(folder_path.iterdir(), key=lambda p: p.name):
            if not entry.is_file() or not entry.name.lower().endswith(".mzid"):
                continue
            role = classify_mzid_file(entry)
            if role != "decoy" and (
                self.sample_mzid is None or classify_mzid_file(self.sample_mzid) == "decoy"
            ):
                self.sample_mzid = entry
            elif self.sample_mzid is None:
                self.sample_mzid = entry
            abs_line = str(entry.resolve()) + "\n"
            if role == "decoy":
                decoy.append(abs_line)
            elif role == "concatenated":
                concat.append(abs_line)
            else:
                real.append(abs_line)

        self.concatenated_mzids = concat
        target_meta = folder_path / f"{self.folder}_target_metafile.txt"
        decoy_meta = folder_path / f"{self.folder}_decoy_metafile.txt"
        with target_meta.open("w", encoding="utf-8") as out:
            out.writelines(real if real else concat)
        with decoy_meta.open("w", encoding="utf-8") as decoy_out:
            decoy_out.writelines(decoy)

        self.target = str(target_meta)
        self.decoy = str(decoy_meta)
        return self

    def convert_to_pin(self):
        """Convert `.mzid` files into a Percolator PIN table with aligned enzyme rules
        and per-scan top-1 Target-Decoy Competition."""
        msgf2pin_bin = resolve_executable(
            getattr(self.args, "msgf2pin_path", None),
            "msgf2pin",
            env_var="UPROTEINS_MSGF2PIN",
            cli_flag="--msgf2pin_path",
        )
        pin_enz, _ = resolve_enzyme_pair(self.args, self.sample_mzid)
        perc_dir = ensure_dir(f"{self.folder}/Percolator")
        pin_out = perc_dir / f"{self.folder}_pin.txt"

        cmd_pin = [str(msgf2pin_bin), str(self.target)]
        if self.decoy and pathlib.Path(self.decoy).is_file() and pathlib.Path(self.decoy).stat().st_size > 0:
            cmd_pin.append(str(self.decoy))

        cmd_pin.extend([
            "-o", str(pin_out),
            "-e", pin_enz,
            "-F", f"{self.filetype}_database.fasta,{self.folder}/Percolator/{self.folder}_decoy.fasta",
            "-c", "2",
            "-P", "decoy_",
        ])
        run_cmd(cmd_pin)
        compete_target_decoy_pin(pin_out)
        return self

    def percolate(self):
        """Run Percolator on the competed PIN table with aligned enzyme and TDC."""
        perc_bin = resolve_executable(
            getattr(self.args, "percolator_path", None),
            "percolator",
            env_var="UPROTEINS_PERCOLATOR",
            cli_flag="--percolator_path",
        )
        _, perc_enz = resolve_enzyme_pair(self.args, self.sample_mzid)
        folder = f"{self.folder}/Percolator"
        cmd_perc = [
            str(perc_bin),
            "-X", f"{folder}/{self.filetype}_percolator_out.xml",
            "--tab-out", f"{folder}/comp_features.txt",
            "-w", f"{folder}/feature_weights.txt",
            "-v", "3",
            "-r", f"{folder}/{self.filetype}_pep_results.txt",
            "-m", f"{folder}/{self.filetype}_results_psm.txt",
            "--search-input", "concatenated",
            "--results-proteins", f"{folder}/{self.filetype}_protein_results.txt",
            "--protein-enzyme", str(perc_enz),
            "--protein-report-duplicates",
            "--protein-report-fragments",
            "--spectral-counting-fdr", "0.01",
            "-Y",
            "-P", "decoy_",
            "--picked-protein", "auto",
            f"{folder}/{self.folder}_pin.txt",
        ]
        run_cmd(cmd_perc)
        return self
