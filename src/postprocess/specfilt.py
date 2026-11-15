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
import re
import shutil

from Bio import SeqIO
import numpy as np
import pandas as pd

from .percolator import parse_percolator_psmid
from ..sequtils import GenomeCoordinates, GenomeCoordinatesRNA, PercolatorUTP, RefSeqGFF, StringTieGFF
from ..sequtils.postsearch import LinkData, SequenceFinder, TSVChunks
from ..sequtils.utilities import PercolatorConverter
from ..utils import concat_files, ensure_dir


_FLANKING_RE = re.compile(r"^[A-Za-z\-]\.(.*)\.[A-Za-z\-]$")
_MOD_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_AA_RE = re.compile(r"[^A-Za-z]")
_PROTEIN_SPLIT_RE = re.compile(r"[,;\t]+")


def strip_psm_peptide(raw_peptide: str) -> str:
    """Strip Percolator/MS-GF+ flanking cleavage (`R.PEPTIDE.K`) and PTMs."""
    pep = str(raw_peptide).strip()
    m = _FLANKING_RE.match(pep)
    if m:
        pep = m.group(1)
    pep = _MOD_BRACKET_RE.sub("", pep)
    return _NON_AA_RE.sub("", pep).upper()


def split_protein_tokens(protein_field: str) -> list[str]:
    """Split a `proteinIds` / MS-GF+ `Protein` field into clean accession tokens,
    stripping MS-GF+ `(pre=...,post=...)` cleavage suffixes."""
    if pd.isna(protein_field):
        return []
    tokens: list[str] = []
    for raw_tok in _PROTEIN_SPLIT_RE.split(str(protein_field)):
        clean = raw_tok.strip().split("(")[0].strip()
        if clean:
            tokens.append(clean)
    return tokens


def _is_decoy_token(tok: str) -> bool:
    low = tok.lower()
    return low.startswith(("decoy_", "reverse_"))


def _strip_decoy_prefix(tok: str) -> str:
    low = tok.lower()
    if low.startswith("decoy_"):
        return tok[6:]
    if low.startswith("reverse_"):
        return tok[8:]
    return tok


def _is_canonical_or_contaminant_base(base_id: str) -> bool:
    low = base_id.lower()
    return (
        base_id.endswith("_ANNO")
        or "_ANNO_" in base_id
        or low.startswith("contaminant_")
    )


def classify_protein_group(protein_ids_str: str) -> str:
    """Classify a PSM's `proteinIds` into one of four mutually exclusive classes:
    - `'canonical_target'`: matches >=1 non-decoy `_ANNO` or `contaminant_` entry.
    - `'novel_target'`: matches exclusively non-decoy unannotated ORFs (`gORF`/`tORF`).
    - `'canonical_decoy'`: all entries are decoys and >=1 base ID is `_ANNO`/`contaminant_`.
    - `'novel_decoy'`: all entries are decoys of unannotated ORFs (`decoy_gORF_`/`decoy_tORF_`).
    """
    tokens = split_protein_tokens(protein_ids_str)
    if not tokens:
        return "novel_target"

    target_tokens = [t for t in tokens if not _is_decoy_token(t)]
    decoy_tokens = [t for t in tokens if _is_decoy_token(t)]

    if target_tokens:
        if any(_is_canonical_or_contaminant_base(t) for t in target_tokens):
            return "canonical_target"
        return "novel_target"

    # Decoy-only row
    if any(_is_canonical_or_contaminant_base(_strip_decoy_prefix(d)) for d in decoy_tokens):
        return "canonical_decoy"
    return "novel_decoy"


def compute_tdc_qvalues(
    scores: np.ndarray,
    is_decoy: np.ndarray,
    *,
    pi0: float = 1.0,
) -> np.ndarray:
    """Compute conservative Storey/Käll Target-Decoy Competition q-values:

        q(s_i) = min_{s <= s_i} min(1.0, pi0 * (D(>= s) + 1) / max(1, T(>= s)))

    Exact score ties receive identical cumulative target/decoy counts at threshold `s`.
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=float)

    scores_arr = np.asarray(scores, dtype=float)
    decoy_arr = np.asarray(is_decoy, dtype=bool)

    # Sort descending by score
    order = np.argsort(-scores_arr, kind="mergesort")
    sorted_scores = scores_arr[order]
    sorted_decoy = decoy_arr[order]

    cum_d = np.cumsum(sorted_decoy.astype(np.int64))
    cum_t = np.cumsum((~sorted_decoy).astype(np.int64))

    # Ensure tied scores all see the total cumulative T and D across the tie group
    if n > 1:
        group_end = np.empty(n, dtype=bool)
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
        group_end[-1] = True
        end_indices = np.flatnonzero(group_end)
        # Broadcast end-of-tie cumulative counts backward across each tie block
        block_ids = np.searchsorted(end_indices, np.arange(n), side="left")
        cum_d = cum_d[end_indices[block_ids]]
        cum_t = cum_t[end_indices[block_ids]]

    raw_fdr = np.minimum(1.0, float(pi0) * (cum_d + 1.0) / np.maximum(1, cum_t))
    # Monotonic step-up (cumulative minimum from lowest score to highest score)
    sorted_q = np.minimum.accumulate(raw_fdr[::-1])[::-1]

    qvals = np.empty(n, dtype=float)
    qvals[order] = sorted_q
    return qvals


class SubgroupFDR(object):
    """Class-specific (subgroup) Target-Decoy FDR estimator for novel unannotated ORFs
    (Nesvizhskii 2010; Blakeley et al. 2012; Sticker et al. 2017)."""

    def __init__(self, pi0: float = 1.0):
        self.pi0 = float(pi0)

    def score_novel_subgroup(
        self,
        novel_targets_df: pd.DataFrame,
        decoy_psm_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = novel_targets_df.copy()
        if df.empty:
            for col in ("global_q_value", "novel_psm_q_value", "novel_peptide_q_value"):
                if col not in df.columns:
                    df[col] = pd.Series(dtype=float)
            return df

        df = df[df["PSMId"] != "PSMId"].copy()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["q-value"] = pd.to_numeric(df["q-value"], errors="coerce")
        if "posterior_error_prob" in df.columns:
            df["posterior_error_prob"] = pd.to_numeric(
                df["posterior_error_prob"], errors="coerce"
            )
        df["global_q_value"] = df["q-value"]

        if decoy_psm_df is None or decoy_psm_df.empty:
            # Legacy fallback when Percolator decoy PSM output is unavailable
            df["novel_psm_q_value"] = df["q-value"]
            df["novel_peptide_q_value"] = df["q-value"]
            return df

        dec = decoy_psm_df[decoy_psm_df["PSMId"] != "PSMId"].copy()
        dec["score"] = pd.to_numeric(dec["score"], errors="coerce")
        dec_classes = dec["proteinIds"].map(classify_protein_group)
        novel_decoys = dec[dec_classes == "novel_decoy"].copy()

        # 1. Subgroup PSM-level q-values across T_novel U D_novel
        t_scores = df["score"].to_numpy(dtype=float)
        d_scores = novel_decoys["score"].to_numpy(dtype=float)
        combined_scores = np.concatenate([t_scores, d_scores])
        combined_is_decoy = np.concatenate([
            np.zeros(len(t_scores), dtype=bool),
            np.ones(len(d_scores), dtype=bool),
        ])
        combined_psm_q = compute_tdc_qvalues(
            combined_scores, combined_is_decoy, pi0=self.pi0
        )
        df["novel_psm_q_value"] = combined_psm_q[: len(t_scores)]

        # 2. Subgroup Peptide-level q-values across distinct stripped peptides
        t_peps = df["peptide"].map(strip_psm_peptide)
        d_peps = novel_decoys["peptide"].map(strip_psm_peptide)

        pep_best: dict[str, tuple[float, bool]] = {}
        for pep_seq, sc in zip(t_peps, t_scores):
            if not pep_seq:
                continue
            prev = pep_best.get(pep_seq)
            if prev is None or sc > prev[0]:
                pep_best[pep_seq] = (float(sc), False)

        for pep_seq, sc in zip(d_peps, d_scores):
            if not pep_seq:
                continue
            prev = pep_best.get(pep_seq)
            # Conservative tie-break: if equal score, decoy wins
            if prev is None or sc > prev[0] or (sc == prev[0] and not prev[1]):
                pep_best[pep_seq] = (float(sc), True)

        pep_keys = list(pep_best.keys())
        if pep_keys:
            pep_scores = np.array([pep_best[k][0] for k in pep_keys], dtype=float)
            pep_is_dec = np.array([pep_best[k][1] for k in pep_keys], dtype=bool)
            pep_qvals = compute_tdc_qvalues(pep_scores, pep_is_dec, pi0=self.pi0)
            pep_q_map = dict(zip(pep_keys, pep_qvals))
            df["novel_peptide_q_value"] = [
                pep_q_map.get(p, 1.0) for p in t_peps
            ]
        else:
            df["novel_peptide_q_value"] = 1.0

        return df


def _load_percolator_psm_table(path: str | pathlib.Path) -> pd.DataFrame:
    """Load a Percolator PSM output table where trailing tab-separated columns
    belong to `proteinIds`."""
    p = pathlib.Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(
            columns=["PSMId", "score", "q-value", "posterior_error_prob", "peptide", "proteinIds"]
        )
    rows: list[list[str]] = []
    with p.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if idx == 0 and parts[0] == "PSMId":
                continue
            if parts[0] == "PSMId":
                continue
            if len(parts) >= 6:
                rows.append([
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    ",".join(p_tok for p_tok in parts[5:] if p_tok),
                ])
    return pd.DataFrame(
        rows,
        columns=["PSMId", "score", "q-value", "posterior_error_prob", "peptide", "proteinIds"],
    )


def _specfile_stem(spec_file: str) -> str:
    name = pathlib.Path(str(spec_file).strip()).name
    stem = pathlib.Path(name).stem
    low = stem.lower()
    if low.endswith("_target"):
        return stem[:-7]
    if low.endswith("_decoy"):
        return stem[:-6]
    return stem


class PostPercolator(object):
    def __init__(self, args, folder, filetype):
        self.args = args
        self.folder = folder
        self.percDir = f"{self.folder}/post_perc"
        self.filetype = filetype
        self.__check_dir()

    def __check_dir(self):
        ensure_dir(f"{self.folder}/post_perc")

    def convert_output(self):
        """Convert Percolator PSM table to standard 6-column tab-separated format."""
        print("Converting output\n")
        pout = PercolatorConverter(
            pout=f"{self.folder}/Percolator/{self.filetype}_results_psm.txt",
            conversion_file=".",
            handle="psm",
            gff=getattr(self.args, "gff", None),
        )
        pout.convert_entries(
            pattern="WP", output=f"{self.percDir}/{self.filetype}_converted_psm"
        )

    def get_coordinates_rna(self):
        """Find genome coordinates of each transcript ORF."""
        print("Getting coordinates\n")
        str_gff = StringTieGFF(gff="assembled.gtf")
        orf_dict = str_gff.get_dict()
        ref_gff = RefSeqGFF(gff=getattr(self.args, "gff", None))
        ref_dict = ref_gff.get_dict()
        coordinates = GenomeCoordinatesRNA(
            f"{self.percDir}/{self.filetype}_converted_psm.txt", orf_dict, ref_dict
        )
        coordinates.get_proteins().save_table(
            output=f"{self.percDir}/{self.filetype}_psm_coords"
        )
        return self

    def get_coordinates_genome(self):
        """Find genome coordinates of each genome ORF."""
        print("Getting coordinates\n")
        ref_gff = RefSeqGFF(gff=getattr(self.args, "gff", None))
        ref_dict = ref_gff.get_dict()
        coordinates = GenomeCoordinates(
            f"{self.percDir}/{self.filetype}_converted_psm.txt", ref_dict
        )
        coordinates.get_coords().save_table(
            output=f"{self.percDir}/{self.filetype}_psm_coords"
        )
        return self

    def filter_novel(self):
        print("Filtering novel peptides\n")
        anno = AnnoFilter(f"{self.percDir}/{self.filetype}_psm_coords.txt")
        anno.remove_annotated(f"{self.percDir}/{self.filetype}_no_anno.txt")
        return self

    def apply_subgroup_fdr(self, qvalue: float | None = None, pep: float | None = None):
        """Compute class-specific (subgroup) novel ORF PSM & peptide q-values and
        filter `{filetype}_no_anno.txt` prior to downstream uniqueness/MS-GF+ linking."""
        q_thresh = (
            float(qvalue)
            if qvalue is not None
            else float(getattr(self.args, "qvalue", 0.01))
        )
        pep_thresh = (
            float(pep)
            if pep is not None
            else float(getattr(self.args, "pep", 1.0))
        )

        no_anno_path = pathlib.Path(f"{self.percDir}/{self.filetype}_no_anno.txt")
        novel_df = pd.read_csv(no_anno_path, sep="\t")

        decoy_psm_path = pathlib.Path(
            f"{self.folder}/Percolator/{self.filetype}_decoy_results_psm.txt"
        )
        decoy_df = (
            _load_percolator_psm_table(decoy_psm_path)
            if decoy_psm_path.is_file()
            else None
        )
        # Also pick up any decoy rows embedded directly inside converted_psm.txt
        conv_path = pathlib.Path(f"{self.percDir}/{self.filetype}_converted_psm.txt")
        if (decoy_df is None or decoy_df.empty) and conv_path.is_file():
            conv_df = pd.read_csv(conv_path, sep="\t")
            if not conv_df.empty and "proteinIds" in conv_df.columns:
                classes = conv_df["proteinIds"].map(classify_protein_group)
                embedded_decoys = conv_df[classes.isin(["novel_decoy", "canonical_decoy"])]
                if not embedded_decoys.empty:
                    decoy_df = embedded_decoys

        estimator = SubgroupFDR(pi0=1.0)
        scored_df = estimator.score_novel_subgroup(novel_df, decoy_df)
        scored_df.to_csv(
            f"{self.percDir}/{self.filetype}_subgroup_fdr.tsv",
            sep="\t",
            index=False,
        )

        mask = (scored_df["novel_psm_q_value"] <= q_thresh) & (
            scored_df["novel_peptide_q_value"] <= q_thresh
        )
        if "posterior_error_prob" in scored_df.columns:
            mask = mask & (scored_df["posterior_error_prob"] <= pep_thresh)

        filtered_df = scored_df[mask].copy()
        filtered_df.to_csv(no_anno_path, sep="\t", index=False)
        return self

    def unique_peptides(self):
        """Copy / filter unique novel peptides into `{filetype}_utps.txt`."""
        print("Removing non-unique peptides\n")
        shutil.copy2(
            f"{self.percDir}/{self.filetype}_no_anno.txt",
            f"{self.percDir}/{self.filetype}_utps.txt",
        )
        return self

    def msgf_info(self):
        """Add MS-GF+ TSV columns to FDR-filtered Percolator novel PSMs."""
        print("Adding MSGF info\n")
        tsv_files = sorted(pathlib.Path(self.folder, "tsv_msgf").glob("*.tsv"))
        concat_files(tsv_files, f"{self.percDir}/{self.filetype}_search.tsv")
        if self.args.transcriptome and self.filetype == "transcriptome":
            chunks = TSVChunks(folder="Transcriptome", filetype="transcriptome")
            chunks.filter_search()
        if self.filetype == "genome":
            chunks = TSVChunks(folder=self.folder, filetype=self.filetype)
            chunks.filter_search()
        link = LinkData(
            f"{self.percDir}/{self.filetype}_chunk_search.tsv",
            f"{self.percDir}/{self.filetype}_utps.txt",
        )
        link.filter_msgf(f"{self.percDir}/{self.filetype}_linked")

    def protein_seqs(self):
        """Attach full database protein sequences to linked PSMs."""
        print("Adding protein sequences\n")
        seq = SequenceFinder(
            f"{self.percDir}/{self.filetype}_linked.tsv",
            f"{self.filetype}_database.fasta",
        )
        seq.df_proteins().save(f"{self.percDir}/{self.filetype}_proteined")

    def protein_threshold(self):
        """Apply protein-level FDR cutoff when requested."""
        print("Protein threshold\n")
        protein = ProteinFDR(folder=self.folder, filetype=self.filetype)
        protein.protein_cutoff()
        protein.apply_to_psm()
        protein.filter_from_utp()
        protein.add_proteins(f"{self.percDir}/{self.filetype}_results_02.txt")

    def add_coordinates(self, qvalue: float | None = None, pep: float | None = None):
        q_thresh = (
            float(qvalue)
            if qvalue is not None
            else float(getattr(self.args, "qvalue", 0.01))
        )
        pep_thresh = (
            float(pep)
            if pep is not None
            else float(getattr(self.args, "pep", 1.0))
        )
        coords = Coordinator(
            proteined=f"{self.percDir}/{self.filetype}_proteined.tsv",
            utps=f"{self.percDir}/{self.filetype}_utps.txt",
            qvalue=q_thresh,
            pep=pep_thresh,
        )
        coords.add_information(f"{self.percDir}/{self.filetype}_results_02.txt")


class AnnoFilter(object):
    def __init__(self, df):
        self.df = pd.read_csv(df, sep="\t")

    def remove_annotated(self, output):
        if self.df.empty:
            self.df.to_csv(output, sep="\t", index=False)
            return self
        classes = self.df["proteinIds"].map(classify_protein_group)
        df = self.df[classes == "novel_target"].copy()
        df.to_csv(output, sep="\t", index=False)
        return self


class Coordinator(object):
    def __init__(self, utps, proteined, qvalue=0.01, pep=1.0):
        self.qvalue = float(qvalue)
        self.pep = float(pep)
        self.UTPs = pd.read_csv(utps, sep="\t")
        if not self.UTPs.empty and "q-value" in self.UTPs.columns:
            self.UTPs = self.UTPs[self.UTPs["q-value"] != "q-value"].copy()
            self.UTPs["q-value"] = pd.to_numeric(self.UTPs["q-value"], errors="coerce")

            if (
                "novel_psm_q_value" in self.UTPs.columns
                and "novel_peptide_q_value" in self.UTPs.columns
            ):
                self.UTPs["novel_psm_q_value"] = pd.to_numeric(
                    self.UTPs["novel_psm_q_value"], errors="coerce"
                )
                self.UTPs["novel_peptide_q_value"] = pd.to_numeric(
                    self.UTPs["novel_peptide_q_value"], errors="coerce"
                )
                q_mask = (self.UTPs["novel_psm_q_value"] <= self.qvalue) & (
                    self.UTPs["novel_peptide_q_value"] <= self.qvalue
                )
            else:
                q_mask = self.UTPs["q-value"] <= self.qvalue

            if "posterior_error_prob" in self.UTPs.columns:
                self.UTPs["posterior_error_prob"] = pd.to_numeric(
                    self.UTPs["posterior_error_prob"], errors="coerce"
                )
                q_mask = q_mask & (self.UTPs["posterior_error_prob"] <= self.pep)

            self.UTPs = self.UTPs[q_mask].copy()

        self.proteined = pd.read_csv(proteined, sep="\t")
        self.passing_psms: set[tuple[str, int]] = self._get_passing_psms()
        self.psm_protein_coords: dict[tuple[str, int, str], str] = {}
        self.coordinates = self._get_coordinates()

    def _get_passing_psms(self) -> set[tuple[str, int]]:
        passing: set[tuple[str, int]] = set()
        if self.UTPs.empty or "PSMId" not in self.UTPs.columns:
            return passing
        for psm_id in self.UTPs["PSMId"].tolist():
            try:
                spec_stem, scan_num, _, _ = parse_percolator_psmid(str(psm_id))
                passing.add((spec_stem, int(scan_num)))
            except ValueError:
                continue
        return passing

    def _get_coordinates(self) -> dict[str, str]:
        if self.UTPs.empty:
            return {}
        coordinates = self.UTPs["Genome Coordinates"].tolist()
        orfs = self.UTPs["proteinIds"].tolist()
        psm_ids = self.UTPs["PSMId"].tolist()
        coordict: dict[str, str] = {}
        for psm_id, coords, orf in zip(psm_ids, coordinates, orfs):
            coord_list = [c.strip() for c in str(coords).split(",") if c.strip()]
            orf_list = split_protein_tokens(orf)
            parsed_key: tuple[str, int] | None = None
            try:
                stem, scan_num, _, _ = parse_percolator_psmid(str(psm_id))
                parsed_key = (stem, int(scan_num))
            except ValueError:
                parsed_key = None

            for idx, prot_id in enumerate(orf_list):
                coord_val = coord_list[idx] if idx < len(coord_list) else (coord_list[0] if coord_list else "")
                if prot_id not in coordict:
                    coordict[prot_id] = coord_val
                if parsed_key is not None:
                    self.psm_protein_coords[(parsed_key[0], parsed_key[1], prot_id)] = coord_val
        return coordict

    def add_information(self, output):
        spec_col = "SpecFile" if "SpecFile" in self.proteined.columns else "#SpecFile"

        if self.proteined.empty or not self.coordinates or not self.passing_psms:
            empty_cols = [c for c in self.proteined.columns if c != "Protein"]
            # Preserve exact output column positions: Genome Coordinates at 5, Protein at 8+1=9
            base_cols = list(self.proteined.columns)
            base_cols.insert(5, "Genome Coordinates")
            base_cols.insert(9, "entry")
            ndf = pd.DataFrame(columns=base_cols)
            if "Protein" in ndf.columns:
                ndf = ndf.drop(columns="Protein")
            ndf = ndf.rename(columns={"entry": "Protein"})
            ndf.to_csv(output, sep="\t", index=False)
            return self

        # Strictly restrict proteined rows to FDR-passing (spec_stem, ScanNum) PSMs!
        prot_df = self.proteined.copy()
        prot_stems = prot_df[spec_col].map(_specfile_stem)
        prot_scans = pd.to_numeric(prot_df["ScanNum"], errors="coerce")

        keep_mask = [
            ( pd.notna(sc) and (st, int(sc)) in self.passing_psms )
            for st, sc in zip(prot_stems, prot_scans)
        ]
        prot_df = prot_df[keep_mask].copy()

        expanded_rows: list[pd.DataFrame] = []
        for row_idx, row in prot_df.iterrows():
            row_tokens = split_protein_tokens(row["Protein"])
            row_stem = _specfile_stem(row[spec_col])
            row_scan = int(row["ScanNum"])
            for tok in row_tokens:
                if tok not in self.coordinates:
                    continue
                coord_val = self.psm_protein_coords.get(
                    (row_stem, row_scan, tok), self.coordinates[tok]
                )
                single = prot_df.loc[[row_idx]].copy()
                single.insert(8, "entry", [tok])
                single.insert(5, "Genome Coordinates", [coord_val])
                expanded_rows.append(single)

        if expanded_rows:
            ndf = pd.concat(expanded_rows, ignore_index=True)
        else:
            base_cols = list(self.proteined.columns)
            base_cols.insert(5, "Genome Coordinates")
            base_cols.insert(9, "entry")
            ndf = pd.DataFrame(columns=base_cols)

        ndf = ndf.drop_duplicates()
        if "Protein" in ndf.columns:
            ndf = ndf.drop(columns="Protein")
        ndf = ndf.rename(columns={"entry": "Protein"})
        ndf.to_csv(output, sep="\t", index=False)
        return self


class ProteinFDR(object):
    def __init__(self, folder, filetype, fdr=0.5):
        self.folder = folder
        self.filetype = filetype
        self.fdr = fdr
        self.percDir = f"{self.folder}/post_perc"
        self.filteredProtein = None
        self.__cat_protein_results()

    def __cat_protein_results(self):
        self.catProteinFile = f"{self.percDir}/{self.filetype}_cat_protein_results.txt"
        protein_files = sorted(
            pathlib.Path(self.folder, "Percolator").glob("*protein_results*")
        )
        concat_files(protein_files, self.catProteinFile)
        return self

    def protein_cutoff(self):
        df = pd.read_csv(self.catProteinFile, sep="\t")
        df = df[df["q-value"] != "q-value"].copy()
        df["q-value"] = pd.to_numeric(df["q-value"], errors="coerce")
        df = df[df["q-value"] <= self.fdr].copy()
        self.filteredProtein = df

    def apply_to_psm(self):
        protein_df = self.filteredProtein[
            self.filteredProtein["ProteinId"].map(classify_protein_group) == "novel_target"
        ]
        allowed_names = set(protein_df["ProteinId"].tolist())
        results = pd.read_csv(f"{self.percDir}/{self.filetype}_proteined.tsv", sep="\t")
        mask = results["Protein"].map(
            lambda p: any(tok in allowed_names for tok in split_protein_tokens(p))
        )
        filtered_results = results[mask].drop_duplicates()
        self.ProteinPSMFiltered = f"{self.percDir}/{self.filetype}_psm_protein_filtered.txt"
        filtered_results.to_csv(self.ProteinPSMFiltered, sep="\t", index=False)

    def __get_utp_prots_and_coords(self):
        utps = f"{self.percDir}/{self.filetype}_utps.txt"
        utp_df = pd.read_csv(utps, sep="\t")
        prots = []
        coords = []
        for prot_list, coord_list in zip(
            utp_df["proteinIds"].tolist(), utp_df["Genome Coordinates"].tolist()
        ):
            splat_prot = split_protein_tokens(prot_list)
            splat_coords = [c.strip() for c in str(coord_list).split(",")]
            for prot, coord in zip(splat_prot, splat_coords):
                prots.append(prot)
                coords.append(coord)
        return prots, coords

    def __rename_orfs(self):
        df = pd.read_csv(
            f"{self.percDir}/{self.filetype}_psm_protein_filtered.txt", sep="\t"
        )
        renamed = [name.split("(")[0] for name in df["Protein"].tolist()]
        df = df.drop(columns="Protein")
        df.insert(8, "Protein", renamed)
        return df

    def filter_from_utp(self):
        prots, coords = self.__get_utp_prots_and_coords()
        df = self.__rename_orfs()
        frames = []
        for prot, coord in zip(prots, coords):
            ndf = df[df["Protein"] == prot].copy()
            ncoords = [coord for _ in range(len(ndf))]
            ndf.insert(5, "Genome Coordinates", ncoords)
            frames.append(ndf)
        if frames:
            filtered_df = pd.concat(frames, ignore_index=True)
        else:
            filtered_df = pd.DataFrame(columns=df.columns)
            filtered_df.insert(5, "Genome Coordinates", [])
        filtered_df = filtered_df.drop_duplicates()
        filtered_df.to_csv(
            f"{self.percDir}/{self.filetype}_results_01.txt", sep="\t", index=False
        )
        return self

    def __read_db(self):
        protein_dict = {}
        db = f"{self.filetype}_database.fasta"
        for record in SeqIO.parse(db, "fasta"):
            if str(record.description) not in protein_dict:
                protein_dict[str(record.description)] = str(record.seq)
        return protein_dict

    def add_proteins(self, output):
        db_proteins = self.__read_db()
        results = pd.read_csv(
            f"{self.percDir}/{self.filetype}_results_01.txt", sep="\t"
        )
        results = results.drop(columns="ORF Sequence")
        fixed_seqs = [db_proteins[entry] for entry in results["Protein"].tolist()]
        results.insert(9, "db entry", fixed_seqs)
        results.to_csv(output, sep="\t", index=False)
        return self
