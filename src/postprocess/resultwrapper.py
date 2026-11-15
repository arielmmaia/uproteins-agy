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


import pathlib

import pandas as pd

from ..sequtils.locus import (
    StringTieGFF,
    _format_genomic_coord,
    format_orf_display_name,
    parse_coordinate_token,
    parse_torf_header,
    project_transcript_interval_to_genome,
)
from ..sequtils.utilities import check_dir


def _normalize_coord_value(raw_val: object) -> str:
    if pd.isna(raw_val):
        return ""
    parsed = parse_coordinate_token(str(raw_val))
    if parsed is None:
        return str(raw_val).strip()
    contig, lo, hi, strand = parsed
    if contig:
        sign = strand if strand in ("+", "-") else "+"
        return f"{contig}:{lo}-{hi}({sign})"
    return f"{lo}-{hi}"


class ResultsWrapper(object):
    def __init__(self, df, folder, filetype, transcriptome_gff: str | pathlib.Path | None = None):
        self.df = pd.read_csv(df, sep="\t") if not isinstance(df, pd.DataFrame) else df.copy()
        self.folder = folder
        self.filetype = filetype
        self.transcriptome_gff = transcriptome_gff
        self.resultsFolder = f"{folder}/Results"
        check_dir(self.resultsFolder)

    def __get_orf_names(self) -> list[str]:
        if "Protein" not in self.df.columns or self.df.empty:
            return []
        return [
            format_orf_display_name(str(orf))
            for orf in self.df["Protein"].tolist()
        ]

    def _resolve_stringtie_dict(self) -> dict:
        candidates: list[pathlib.Path] = []
        if self.transcriptome_gff is not None:
            candidates.append(pathlib.Path(self.transcriptome_gff))
        candidates.append(pathlib.Path(self.folder).parent / "assembled.gtf")
        candidates.append(pathlib.Path("assembled.gtf"))
        for cand in candidates:
            if cand.is_file():
                return StringTieGFF(cand).get_dict()
        return {}

    def _project_transcriptome_extended_coords(self) -> tuple[list[str], list[str]]:
        """Return `(genomic_coords, transcript_coords)` for Transcriptome results without
        ever overwriting genomic coordinates with local RNA offsets."""
        n = len(self.df)
        if n == 0:
            return [], []

        local_tx_coords = [
            _normalize_coord_value(v)
            for v in (
                self.df["Extended Coordinates"].tolist()
                if "Extended Coordinates" in self.df.columns
                else [""] * n
            )
        ]

        if "Extended Genomic Coordinates" in self.df.columns:
            ext_g = [
                _normalize_coord_value(v)
                for v in self.df["Extended Genomic Coordinates"].tolist()
            ]
            fallback_g = [
                _normalize_coord_value(v)
                for v in (
                    self.df["Genome Coordinates"].tolist()
                    if "Genome Coordinates" in self.df.columns
                    else [""] * n
                )
            ]
            resolved_g = [
                eg if eg else fg for eg, fg in zip(ext_g, fallback_g)
            ]
            return resolved_g, local_tx_coords

        st_dict = self._resolve_stringtie_dict()
        contigs_seen = {
            str(orf.chromosome)
            for orf in st_dict.values()
            if getattr(orf, "chromosome", None)
        }
        include_contig = len(contigs_seen) > 1
        proteins = (
            self.df["Protein"].tolist() if "Protein" in self.df.columns else [""] * n
        )
        orig_genomic = (
            [_normalize_coord_value(v) for v in self.df["Genome Coordinates"].tolist()]
            if "Genome Coordinates" in self.df.columns
            else [""] * n
        )

        projected_genomic: list[str] = []
        for prot, tx_coord_str, orig_g in zip(proteins, local_tx_coords, orig_genomic):
            t_parsed = parse_torf_header(str(prot))
            c_parsed = parse_coordinate_token(tx_coord_str)
            if t_parsed is not None and c_parsed is not None and st_dict:
                tx_id, _orig_s, _orig_e, orf_strand = t_parsed
                _c_contig, tx_lo, tx_hi, _c_strand = c_parsed
                tx_entry = st_dict.get(tx_id)
                if tx_entry is None and tx_id.startswith("uproteins."):
                    short_id = ".".join(tx_id.split(".")[:2])
                    tx_entry = st_dict.get(short_id)
                if tx_entry is not None:
                    exons = tx_entry.exons or [(int(tx_entry.start), int(tx_entry.end))]
                    g_lo, g_hi, g_strand = project_transcript_interval_to_genome(
                        exons=exons,
                        tx_strand=str(tx_entry.strand or "+"),
                        tx_orf_start=tx_lo,
                        tx_orf_end=tx_hi,
                        orf_strand=orf_strand,
                    )
                    orig_has_contig = ":" in orig_g
                    projected_genomic.append(
                        _format_genomic_coord(
                            getattr(tx_entry, "chromosome", None),
                            g_lo,
                            g_hi,
                            g_strand,
                            include_contig or orig_has_contig,
                        )
                    )
                    continue
            projected_genomic.append(orig_g if orig_g else tx_coord_str)

        return projected_genomic, local_tx_coords

    def reformat(self, pre_validation=True):
        is_transcriptome = str(self.filetype).lower() == "transcriptome"
        new_names = self.__get_orf_names()

        if is_transcriptome:
            genomic_coords, tx_coords = self._project_transcriptome_extended_coords()
            drop_cols = [
                c
                for c in (
                    "ORF Sequence",
                    "Protein",
                    "Fixed Peptides",
                    "Extended ORF",
                    "Extended Coordinates",
                    "Extended Genomic Coordinates",
                    "Genome Coordinates",
                )
                if c in self.df.columns
            ]
            out_df = self.df.drop(columns=drop_cols).copy()
            out_df = out_df.rename(columns={"Extended Sequence": "Protein sequence"})
            insert_pos = min(4, len(out_df.columns))
            out_df.insert(insert_pos, "Genome Coordinates", genomic_coords)
            out_df.insert(insert_pos + 1, "Transcript Coordinates", tx_coords)
        else:
            norm_ext_coords = (
                [_normalize_coord_value(v) for v in self.df["Extended Coordinates"].tolist()]
                if "Extended Coordinates" in self.df.columns
                else (
                    [_normalize_coord_value(v) for v in self.df["Genome Coordinates"].tolist()]
                    if "Genome Coordinates" in self.df.columns
                    else []
                )
            )
            drop_cols = [
                c
                for c in (
                    "ORF Sequence",
                    "Protein",
                    "Fixed Peptides",
                    "Extended ORF",
                    "Genome Coordinates",
                    "Extended Genomic Coordinates",
                )
                if c in self.df.columns
            ]
            out_df = self.df.drop(columns=drop_cols).copy()
            if "Extended Coordinates" in out_df.columns:
                out_df["Extended Coordinates"] = norm_ext_coords
                out_df = out_df.rename(
                    columns={
                        "Extended Sequence": "Protein sequence",
                        "Extended Coordinates": "Genome Coordinates",
                    }
                )
            else:
                out_df = out_df.rename(columns={"Extended Sequence": "Protein sequence"})
                insert_pos = min(4, len(out_df.columns))
                out_df.insert(insert_pos, "Genome Coordinates", norm_ext_coords)

        name_pos = min(1, len(out_df.columns))
        out_df.insert(name_pos, "ORF name", new_names)
        self.df = out_df

        pattern = "pre_validation" if pre_validation else "post_validation"
        self.df.to_csv(
            f"{self.resultsFolder}/{self.filetype}_{pattern}_results.txt",
            sep="\t",
            index=False,
        )
        return self
