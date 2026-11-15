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

"""Stage 6 summary statistics, composition/RBS plots, non-redundant FASTA export, and overlap Venns."""

from __future__ import annotations

import os
import pathlib
import typing as t

from Bio import SeqIO
import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
import seaborn as sns


def _normalize_rbs_label(raw_value: t.Any) -> str:
    if pd.isna(raw_value):
        return "Unannotated"
    text = str(raw_value).strip()
    return text if text else "Unannotated"


def _format_rbs_tick(label: str) -> str:
    return (
        label.replace("Present. Low to moderate binding", "Present. Low to\nmoderate binding")
        .replace("Present. Strong binding", "Present.\nStrong binding")
        .replace("None (SD absent)", "None\n(SD absent)")
    )


class Metrics:
    def __init__(self, args: t.Any) -> None:
        self.workingDir = os.getcwd()
        self.outdir = str(self.workingDir)
        self.args = args
        self.transcriptomeDir = f"{self.outdir}/Transcriptome"
        self.genomeDir = f"{self.outdir}/Genome"
        self.maxsize = int(getattr(args, "maxsize", 100))
        self.non_redundant = bool(getattr(args, "non_redundant", False))
        self.metricsDir = f"{self.outdir}/metrics"
        self.__check_dirs([self.metricsDir])

        self.RBS: dict[str, t.Any] = {}
        self.energies: dict[str, t.Any] = {}
        self.data: dict[str, list[t.Any]] = {
            "microproteins": [],
            "RBS": [],
            "Energies": [],
            "Database": [],
            "Subset": [],
        }

        self.genomeResultsDir = f"{self.genomeDir}/Results"
        self.transcriptomeResultsDir = f"{self.transcriptomeDir}/Results"
        self.genomePreValidation = (
            f"{self.genomeDir}/Results/genome_pre_validation_results.txt"
        )
        self.genomePostValidation = self.genomePreValidation.replace(
            "pre_validation", "post_validation"
        )
        self.transcriptomePreValidation = (
            f"{self.transcriptomeDir}/Results/transcriptome_pre_validation_results.txt"
        )
        self.transcriptomePostValidation = self.transcriptomePreValidation.replace(
            "pre_validation", "post_validation"
        )

        self.has_transcriptome = self._detect_transcriptome()
        self.colors = ["#6A93E6", "lightgreen"]
        self.palette = {"Genome": "#6A93E6", "Transcriptome": "lightgreen"}

        self.fastaFiles: list[str] = []
        self.df = pd.DataFrame(data=self.data)
        self.aa_distribution_df = pd.DataFrame()
        self.rbs_summary_df = pd.DataFrame()

    def _detect_transcriptome(self) -> bool:
        explicit = getattr(self.args, "transcriptome", None)
        files_exist = os.path.isfile(self.transcriptomePreValidation) or os.path.isfile(
            self.transcriptomePostValidation
        )
        if explicit is False:
            return False
        if isinstance(explicit, str):
            norm = explicit.strip().upper()
            if norm in {"NO", "N", "FALSE", "0"}:
                return False
            if norm in {"YES", "Y", "TRUE", "1"}:
                return files_exist
        return files_exist

    @property
    def active_databases(self) -> list[str]:
        return ["Genome", "Transcriptome"] if self.has_transcriptome else ["Genome"]

    @staticmethod
    def __check_dirs(folders: t.Iterable[str]) -> None:
        for folder in folders:
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

    def _apply_non_redundant(self) -> None:
        if self.non_redundant and not self.df.empty:
            self.df = (
                self.df.drop_duplicates(subset=["Subset", "microproteins"], keep="first")
                .reset_index(drop=True)
            )
            self.data = {
                col: self.df[col].tolist()
                for col in ("microproteins", "RBS", "Energies", "Database", "Subset")
            }

    def count_sequences(self) -> pd.DataFrame:
        self.data = {
            "microproteins": [],
            "RBS": [],
            "Energies": [],
            "Database": [],
            "Subset": [],
        }

        def count(filepath: str, subset: str, db: str) -> None:
            if not os.path.isfile(filepath):
                return
            df = pd.read_csv(filepath, sep="\t", low_memory=False)
            if df.empty or "Protein sequence" not in df.columns:
                return
            df = df.dropna(subset=["Protein sequence"]).copy()
            df["Protein sequence"] = df["Protein sequence"].astype(str)
            df = df[df["Protein sequence"].str.len() <= self.maxsize]
            df = df.drop_duplicates(subset=["Protein sequence"])

            total_mips = df["Protein sequence"].tolist()
            rbs_list = (
                df["Shine Dalgarno"].tolist()
                if "Shine Dalgarno" in df.columns
                else [np.nan] * len(total_mips)
            )
            energies = (
                pd.to_numeric(df["Free Energy"], errors="coerce").tolist()
                if "Free Energy" in df.columns
                else [np.nan] * len(total_mips)
            )

            for i, mp in enumerate(total_mips):
                self.data["microproteins"].append(mp)
                self.data["RBS"].append(rbs_list[i])
                self.data["Energies"].append(energies[i])
                self.data["Subset"].append(subset)
                self.data["Database"].append(db)

        count(filepath=self.genomePreValidation, db="Genome", subset="Pre-validation")
        count(filepath=self.genomePostValidation, db="Genome", subset="Validated")
        if self.has_transcriptome:
            count(
                filepath=self.transcriptomePreValidation,
                db="Transcriptome",
                subset="Pre-validation",
            )
            count(
                filepath=self.transcriptomePostValidation,
                db="Transcriptome",
                subset="Validated",
            )

        self.df = pd.DataFrame(data=self.data)
        self._apply_non_redundant()
        return self.df

    def plot_aa_dist(self) -> pd.DataFrame:
        """Compute per-(Database, Subset) amino-acid frequencies summing to 100.0% per cohort."""
        aa_counts: dict[tuple[str, str], dict[str, int]] = {}
        subset_totals: dict[tuple[str, str], int] = {}

        for mp, db, subset in zip(
            self.df["microproteins"].astype(str) if not self.df.empty else [],
            self.df["Database"].astype(str) if not self.df.empty else [],
            self.df["Subset"].astype(str) if not self.df.empty else [],
        ):
            key = (db, subset)
            bucket = aa_counts.setdefault(key, {})
            subset_totals.setdefault(key, 0)
            for aa in mp:
                bucket[aa] = bucket.get(aa, 0) + 1
                subset_totals[key] += 1

        rows: list[dict[str, t.Any]] = []
        for (db, subset), bucket in aa_counts.items():
            denom = subset_totals[(db, subset)]
            if denom <= 0:
                continue
            for aa in sorted(bucket.keys()):
                cnt = bucket[aa]
                rows.append(
                    {
                        "Amino acid": aa,
                        "Database": db,
                        "Subset": subset,
                        "Residue_Count": cnt,
                        "Frequency_Pct": (cnt / denom) * 100.0,
                    }
                )

        self.aa_distribution_df = pd.DataFrame(
            rows,
            columns=["Amino acid", "Database", "Subset", "Residue_Count", "Frequency_Pct"],
        )
        self.aa_distribution_df.to_csv(
            f"{self.metricsDir}/aa_distribution.tsv", sep="\t", index=False
        )

        present_subsets = [
            s
            for s in ("Pre-validation", "Validated")
            if not self.aa_distribution_df.empty
            and (self.aa_distribution_df["Subset"] == s).any()
        ]
        n_panels = max(len(present_subsets), 1)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)

        if not present_subsets:
            axes[0, 0].text(
                0.5, 0.5, "No microprotein sequences available", ha="center", va="center"
            )
            axes[0, 0].set_axis_off()
        else:
            active_palette = {db: self.palette[db] for db in self.active_databases}
            for idx, sub in enumerate(present_subsets):
                ax = axes[0, idx]
                sub_df = self.aa_distribution_df[
                    self.aa_distribution_df["Subset"] == sub
                ]
                sns.barplot(
                    data=sub_df,
                    x="Amino acid",
                    y="Frequency_Pct",
                    hue="Database",
                    errorbar=None,
                    palette=active_palette,
                    edgecolor="black",
                    ax=ax,
                )
                ax.set_title(f"Amino Acid Composition ({sub})")
                ax.set_ylabel("Frequency (%)" if idx == 0 else "")
                sns.despine(ax=ax, top=True, right=True)

        fig.tight_layout()
        fig.savefig(f"{self.metricsDir}/aa_distribution.pdf", dpi=300)
        plt.close(fig)
        return self.aa_distribution_df

    def plot_orf_numbers(self) -> pd.DataFrame:
        dbs = self.active_databases
        subsets = ["Pre-validation", "Validated"]
        rows: list[dict[str, t.Any]] = []
        for db in dbs:
            for subset in subsets:
                if self.df.empty:
                    cnt = 0
                else:
                    sub_df = self.df[
                        (self.df["Subset"] == subset) & (self.df["Database"] == db)
                    ]
                    cnt = len(sub_df)
                rows.append({"Microproteins": cnt, "Subset": subset, "Database": db})

        summary_df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(5.5, 6))
        active_palette = {db: self.palette[db] for db in dbs}
        sns.barplot(
            data=summary_df,
            x="Subset",
            y="Microproteins",
            hue="Database",
            errorbar=None,
            palette=active_palette,
            edgecolor="black",
            ax=ax,
        )
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0:
                ax.annotate(
                    str(int(round(height))),
                    (patch.get_x() + patch.get_width() / 2.0, height),
                    ha="center",
                    va="center",
                    xytext=(0, 8),
                    textcoords="offset points",
                )
        ax.set_xlabel("Subset")
        ax.set_ylabel("Counts")
        ax.set_title("Identified Microproteins by Subset and Database")
        sns.despine(ax=ax, top=True, right=True)
        fig.tight_layout()
        fig.savefig(f"{self.metricsDir}/identified_microproteins.pdf", dpi=300)
        plt.close(fig)
        return summary_df

    def plot_energies(self) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 6))
        energy_df = self.df.copy() if not self.df.empty else pd.DataFrame()
        if not energy_df.empty:
            energy_df["Energies"] = pd.to_numeric(energy_df["Energies"], errors="coerce")
            energy_df = energy_df.dropna(subset=["Energies"])

        if energy_df.empty:
            ax.text(
                0.5,
                0.5,
                "No thermodynamic RBS energies available",
                ha="center",
                va="center",
            )
            ax.set_axis_off()
        else:
            active_palette = {db: self.palette[db] for db in self.active_databases}
            sns.boxplot(
                data=energy_df,
                x="Subset",
                hue="Database",
                y="Energies",
                palette=active_palette,
                ax=ax,
            )
            ax.set_ylabel("Free Energy (kcal/mol)")
            sns.despine(ax=ax, top=True, right=True)

        fig.tight_layout()
        fig.savefig(f"{self.metricsDir}/energies_distribution.pdf", dpi=300)
        plt.close(fig)

    def plot_rbs(self) -> pd.DataFrame:
        """Tabulate and plot RBS classes per Subset without double-counting or erasing Leaderless."""
        if self.df.empty:
            self.rbs_summary_df = pd.DataFrame(
                columns=["Subset", "Database", "RBS", "Count"]
            )
        else:
            work = self.df.copy()
            work["RBS_Clean"] = [_normalize_rbs_label(v) for v in work["RBS"]]
            grouped = (
                work.groupby(["Subset", "Database", "RBS_Clean"], dropna=False)
                .size()
                .reset_index(name="Count")
                .rename(columns={"RBS_Clean": "RBS"})
            )
            self.rbs_summary_df = grouped

        self.rbs_summary_df.to_csv(
            f"{self.metricsDir}/shine_dalgarno_counts.tsv", sep="\t", index=False
        )

        present_subsets = [
            s
            for s in ("Pre-validation", "Validated")
            if not self.rbs_summary_df.empty
            and (self.rbs_summary_df["Subset"] == s).any()
        ]
        n_panels = max(len(present_subsets), 1)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), squeeze=False)

        if not present_subsets:
            axes[0, 0].text(
                0.5, 0.5, "No Shine-Dalgarno annotations available", ha="center", va="center"
            )
            axes[0, 0].set_axis_off()
        else:
            active_palette = {db: self.palette[db] for db in self.active_databases}
            for idx, sub in enumerate(present_subsets):
                ax = axes[0, idx]
                sub_df = self.rbs_summary_df[self.rbs_summary_df["Subset"] == sub].copy()
                sub_df["RBS_Display"] = [_format_rbs_tick(v) for v in sub_df["RBS"]]
                sns.barplot(
                    data=sub_df,
                    x="RBS_Display",
                    y="Count",
                    hue="Database",
                    errorbar=None,
                    palette=active_palette,
                    edgecolor="black",
                    ax=ax,
                )
                ax.set_title(f"Ribosome Binding Sites ({sub})")
                ax.set_xlabel("Shine-Dalgarno Category")
                ax.set_ylabel("Count" if idx == 0 else "")
                for patch in ax.patches:
                    height = patch.get_height()
                    if height > 0:
                        ax.annotate(
                            str(int(round(height))),
                            (patch.get_x() + patch.get_width() / 2.0, height),
                            ha="center",
                            va="center",
                            xytext=(0, 8),
                            textcoords="offset points",
                        )
                sns.despine(ax=ax, top=True, right=True)

        fig.tight_layout()
        fig.savefig(f"{self.metricsDir}/shine-dalgarno.pdf", dpi=300)
        plt.close(fig)
        return self.rbs_summary_df

    def generate_fasta(self) -> list[str]:
        self.fastaFiles = []
        specs: list[tuple[str, str, str]] = [
            (self.genomePreValidation, self.genomeResultsDir, "Pre-validation"),
            (self.genomePostValidation, self.genomeResultsDir, "Validated"),
        ]
        if self.has_transcriptome:
            specs = [
                (
                    self.transcriptomePreValidation,
                    self.transcriptomeResultsDir,
                    "Pre-validation",
                ),
                (
                    self.transcriptomePostValidation,
                    self.transcriptomeResultsDir,
                    "Validated",
                ),
            ] + specs

        for filepath, folder, subset_name in specs:
            if not os.path.isfile(filepath):
                continue
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(filepath, sep="\t", low_memory=False)
            fasta_lines: list[str] = []
            if not df.empty and "Protein sequence" in df.columns:
                df = df.dropna(subset=["Protein sequence"]).copy()
                df["Protein sequence"] = df["Protein sequence"].astype(str)
                df = df[df["Protein sequence"].str.len() <= self.maxsize]
                df = df.drop_duplicates(subset=["Protein sequence"])

                entries = (
                    df["ORF name"].astype(str).tolist()
                    if "ORF name" in df.columns
                    else [f"smORF_{idx+1}" for idx in range(len(df))]
                )
                coords = (
                    df["Genome Coordinates"].fillna("").astype(str).tolist()
                    if "Genome Coordinates" in df.columns
                    else [""] * len(df)
                )
                seqs = df["Protein sequence"].tolist()

                seen_headers: dict[str, int] = {}
                for entry, coord, seq in zip(entries, coords, seqs):
                    clean_coord = coord.strip()
                    base_header = f"{entry}|{clean_coord}" if clean_coord else entry
                    if base_header in seen_headers:
                        seen_headers[base_header] += 1
                        header = f"{base_header}_{seen_headers[base_header]}"
                    else:
                        seen_headers[base_header] = 1
                        header = base_header
                    fasta_lines.append(f">{header}\n{seq}\n")

            fasta_file = f"{folder}/{subset_name}.fasta"
            with open(fasta_file, "w", encoding="utf-8") as handler:
                handler.writelines(fasta_lines)
            self.fastaFiles.append(fasta_file)

        return self.fastaFiles

    def plot_venn(self) -> list[str]:
        if not self.has_transcriptome:
            return []

        validated_files = [f for f in self.fastaFiles if f.endswith("/Validated.fasta")]
        pre_files = [f for f in self.fastaFiles if f.endswith("/Pre-validation.fasta")]
        written: list[str] = []

        if len(validated_files) == 2:
            val_out = f"{self.metricsDir}/venn_validated.png"
            val_venn = ResultsComparison(
                fasta_files=",".join(validated_files), groups="Transcriptome,Genome"
            )
            val_venn.gather_microproteins()
            val_venn.plot_venn(palette=self.colors[::-1], output=val_out)
            written.append(val_out)

        if len(pre_files) == 2:
            pre_out = f"{self.metricsDir}/venn_pre_validation.png"
            pre_venn = ResultsComparison(
                fasta_files=",".join(pre_files), groups="Transcriptome,Genome"
            )
            pre_venn.gather_microproteins()
            pre_venn.plot_venn(palette=self.colors[::-1], output=pre_out)
            written.append(pre_out)

        return written

    def plot_venn_2(self, subset: str = "Pre-validation") -> t.Optional[str]:
        if not self.has_transcriptome or self.df.empty:
            return None

        tx_seqs = set(
            self.df[
                (self.df["Subset"] == subset) & (self.df["Database"] == "Transcriptome")
            ]["microproteins"]
            .astype(str)
            .tolist()
        )
        g_seqs = set(
            self.df[(self.df["Subset"] == subset) & (self.df["Database"] == "Genome")][
                "microproteins"
            ]
            .astype(str)
            .tolist()
        )
        out_path = (
            f"{self.metricsDir}/venn_{subset.lower().replace('-', '_')}.pdf"
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        venn2(
            [tx_seqs, g_seqs],
            set_labels=("Transcriptome", "Genome"),
            set_colors=tuple(self.colors[::-1]),
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path


class ResultsComparison:
    def __init__(self, fasta_files: str, groups: str) -> None:
        self.folders = [f.strip() for f in fasta_files.split(",") if f.strip()]
        self.groups = [g.strip() for g in groups.split(",") if g.strip()]
        self.microproteins: dict[str, set[str]] = {}

    def gather_microproteins(self) -> dict[str, set[str]]:
        for fasta, group in zip(self.folders, self.groups):
            bucket = self.microproteins.setdefault(group, set())
            if not os.path.isfile(fasta):
                continue
            for record in SeqIO.parse(fasta, "fasta"):
                bucket.add(str(record.seq))
        return self.microproteins

    def plot_venn(
        self, palette: t.Sequence[str], output: str | os.PathLike[str]
    ) -> pathlib.Path:
        out_path = pathlib.Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        keys = list(self.groups[:2])
        if len(keys) == 2:
            set_a = self.microproteins.get(keys[0], set())
            set_b = self.microproteins.get(keys[1], set())
            colors = (
                (palette[0], palette[1])
                if len(palette) >= 2
                else ("lightgreen", "#6A93E6")
            )
            venn2([set_a, set_b], set_labels=(keys[0], keys[1]), set_colors=colors, ax=ax)
        else:
            ax.text(0.5, 0.5, "Single database run", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
