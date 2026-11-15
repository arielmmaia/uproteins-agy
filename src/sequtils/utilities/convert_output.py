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


from Bio import SeqIO
import pandas as pd


def _load_optional_gff(gff_path) -> pd.DataFrame:
    from ..locus import _IGNORED_GFF_FEATURES, read_gff_dataframe

    df = read_gff_dataframe(gff_path)
    if df.empty:
        return df
    feat_lower = df["feature"].astype(str).str.lower()
    return df[~feat_lower.isin(_IGNORED_GFF_FEATURES)].copy()


def _extract_gene_label(raw_attrs: str) -> str | None:
    from ..locus import parse_gff_attributes

    attrs = parse_gff_attributes(raw_attrs)
    for key in ("locus_tag", "Name", "gene", "Parent", "ID", "gene_id"):
        val = attrs.get(key)
        if val:
            for prefix in ("gene-", "cds-", "rna-"):
                if val.startswith(prefix):
                    return val[len(prefix) :]
            return val
    parts = [p.strip() for p in str(raw_attrs).split(";") if p.strip()]
    if len(parts) > 1 and len(parts[1]) > 12:
        return parts[1][12:]
    return None


class FastaConverter(object):
    def __init__(self, fasta=None, conversion_file=None, gff=None):
        self.fasta = fasta
        self.cFile = conversion_file
        self.refSeq = gff
        self.gff = self.__gff_info()

    def __gff_info(self) -> pd.DataFrame:
        return _load_optional_gff(self.refSeq)

    def convert_entries(self, sep="\t", **kwargs):
        """Convert protein entries in a FASTA file using an optional ID mapping table
        and reference GFF3/GTF annotation."""
        pattern = kwargs.get("pattern") or ""
        id_dict: dict[str, str] = {}
        if self.cFile is not None and not self.gff.empty:
            df = pd.read_csv(self.cFile, sep=sep)
            df = df[df["To"].astype(str).str.contains(pattern, regex=False, na=False)]
            ids = df["From"].tolist()
            target = df["To"].tolist()
            for src_id, tgt_id in zip(ids, target):
                gff_hits = self.gff[
                    self.gff["attributes"].astype(str).str.contains(
                        str(tgt_id), regex=False, na=False
                    )
                ]
                if not gff_hits.empty:
                    gene = _extract_gene_label(gff_hits["attributes"].iloc[0])
                    if gene:
                        id_dict[str(src_id)] = gene

        fasta_file: list[str] = []
        records = SeqIO.parse(self.fasta, "fasta")
        for record in records:
            if "|" in record.id:
                parts = record.id.split("|")
                name = parts[2] if len(parts) > 2 else parts[-1]
                converted = id_dict.get(name, name)
            else:
                converted = id_dict.get(record.id, record.id)
            fasta_file.append(f">{converted}\n{record.seq}\n")
        with open("converted_genome_database.fasta", "w") as fa:
            fa.writelines(fasta_file)


class PercolatorConverter(object):
    def __init__(self, pout=None, handle="psm", conversion_file=None, gff=None):
        """Convert a Percolator PSM or Protein table into standard 6-column TSV format.
        Gracefully accepts `gff=None` when `--gff` is omitted."""
        self.handle = handle
        self.pout = pout
        self.__check_handle()
        self.cFile = conversion_file
        self.refSeq = gff
        self.gff = self.__gff_info()

    def __check_handle(self):
        if self.handle == "psm":
            self.dataFrame = pd.read_csv(self.pout, sep="\t", usecols=[0, 1, 2, 3, 4, 5])
            if "proteinIds" in self.dataFrame.columns:
                self.dataFrame = self.dataFrame.drop("proteinIds", axis=1)
            self.__fix_columns()
        elif self.handle == "protein":
            self.dataFrame = pd.read_csv(self.pout, sep="\t")

    def __fix_columns(self):
        """Collapse all trailing tab-separated protein IDs of Percolator PSM output
        into a single comma-delimited `proteinIds` column."""
        ids: list[str] = []
        with open(self.pout, "r", encoding="utf-8", errors="replace") as psm:
            for idx, line in enumerate(psm):
                if idx == 0:
                    continue
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                proteins = [p.strip() for p in stripped.split("\t")[5:] if p.strip()]
                ids.append(",".join(proteins))
        self.dataFrame.insert(5, "proteinIds", ids)

    def convert_entries(self, sep="\t", **kwargs):
        """Write standardized Percolator PSM/Protein table to `{output}.txt`."""
        self.dataFrame.to_csv(f'{kwargs.get("output")}.txt', sep="\t", index=False)

    def __gff_info(self) -> pd.DataFrame:
        return _load_optional_gff(self.refSeq)