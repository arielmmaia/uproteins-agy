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


import pathlib
import typing as t

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..utils import REPO_ROOT, concat_files


_PKG_CONTAMINANTS = pathlib.Path(__file__).resolve().parents[1] / "seqlib" / "contaminants.txt"
DEFAULT_CONTAMINANTS_PATH = (
    _PKG_CONTAMINANTS
    if _PKG_CONTAMINANTS.is_file()
    else REPO_ROOT / "seqlib" / "contaminants.txt"
)


def load_forward_contaminants(
    contaminants_path: t.Optional[t.Union[str, pathlib.Path]] = None,
) -> list[SeqRecord]:
    """Load only the 121 forward real contaminant entries (`>contaminant_...`) from seqlib/contaminants.txt,
    discarding pre-baked `>Reverse_contaminant_...` entries while keeping headers and B/X/Z residues intact."""
    path = pathlib.Path(contaminants_path) if contaminants_path is not None else DEFAULT_CONTAMINANTS_PATH
    if not path.is_file():
        return []

    forward_records: list[SeqRecord] = []
    for record in SeqIO.parse(str(path), "fasta"):
        rec_id = str(record.id)
        if rec_id.startswith("Reverse_") or rec_id.lower().startswith("reverse_"):
            continue
        if not rec_id.startswith("contaminant_"):
            rec_id = f"contaminant_{rec_id}"

        desc_tokens = record.description.split()
        while desc_tokens and desc_tokens[0] in (record.id, rec_id):
            desc_tokens.pop(0)
        extra_desc = " ".join(desc_tokens)

        clean_seq = "".join(str(record.seq).split()).rstrip("*").upper()
        if not clean_seq:
            continue

        forward_records.append(
            SeqRecord(
                Seq(clean_seq),
                id=rec_id,
                name=rec_id,
                description=extra_desc,
            )
        )
    return forward_records


def write_forward_contaminants_fasta(
    output_path: t.Union[str, pathlib.Path] = "contaminants_forward.fasta",
    contaminants_path: t.Optional[t.Union[str, pathlib.Path]] = None,
) -> pathlib.Path:
    out = pathlib.Path(output_path)
    records = load_forward_contaminants(contaminants_path=contaminants_path)
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            SeqIO.write(rec, handle, "fasta")
    return out


class Database(object):
    def __init__(
        self,
        ob,
        p,
        filetype,
        include_contaminants: bool = True,
        contaminants_path: t.Optional[t.Union[str, pathlib.Path]] = None,
    ):
        self.orf_to_blast = ob
        self.proteome = p
        self.filetype = filetype
        self.include_contaminants = bool(include_contaminants)
        self.contaminants_path = contaminants_path
        self.annotated = 'annotated.fasta'

    def mark_annotated(self):
        with open('annotated.fasta', 'w') as handle:
            for record in SeqIO.parse(self.proteome, 'fasta'):
                orig_id = record.id
                new_id = orig_id if orig_id.endswith('_ANNO') else f"{orig_id}_ANNO"
                desc_tokens = record.description.split()
                while desc_tokens and desc_tokens[0] in (orig_id, new_id):
                    desc_tokens.pop(0)
                extra_desc = " ".join(desc_tokens)
                clean_seq = "".join(str(record.seq).split()).rstrip("*").upper()
                if not clean_seq:
                    continue

                out_record = SeqRecord(
                    Seq(clean_seq),
                    id=new_id,
                    name=new_id,
                    description=extra_desc,
                )
                SeqIO.write(out_record, handle, 'fasta')
        self.annotated = 'annotated.fasta'

    def unify(self):
        sources: list[t.Union[str, pathlib.Path]] = [self.orf_to_blast, self.annotated]
        if self.include_contaminants:
            contam_fa = write_forward_contaminants_fasta(
                output_path="contaminants_forward.fasta",
                contaminants_path=self.contaminants_path,
            )
            if contam_fa.stat().st_size > 0:
                sources.append(contam_fa)

        concat_files(
            sources,
            f'{self.filetype}_database.fasta',
        )
