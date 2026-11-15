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


from ..sequtils.conversion import complement_dna
from ..sequtils.locus import (
    StringTieGFF,
    load_genome_contigs,
    resolve_contig_sequence,
)
from ..sequtils.orflib import ORFCollection
from ..sequtils.transcriptomics import TranscriptExtractor
from ..utils import REPO_ROOT, resolve_executable, resolve_free2bind_script, run_cmd
from .duplex_dg import (
    MIN_LEADER_UTR_NT,
    classify_rbs_category,
    compute_rbs_free_energy,
    extract_16s_anti_sd_tail,
    extract_genomic_upstream_21nt,
    extract_transcript_upstream,
)


class SDInspection(object):
    def __init__(
        self,
        args,
        filetype,
        folder,
        alternatives,
        subset="Genome",
        transcriptome_gff=None,
        transcripts=None,
        testing=False,
    ):
        """Inspect 21-nt upstream regions for Shine-Dalgarno 16S rRNA hybridization
        free energy (dG_37, kcal/mol)."""
        self.args = args
        self.subset = subset
        self.filetype = filetype
        self.folder = folder
        self.testing = testing
        self.pyPath = str(REPO_ROOT)
        self.freeAlignPath = None
        self.backend = str(getattr(args, "rbs_backend", "python") or "python").lower()
        self.circular = bool(getattr(args, "circular", True))

        self.tORFs = {}
        self.__get_transcripts(transcriptome_gff, assembly=transcripts)

        rrna_path = getattr(args, "rrna", None)
        self.rRNA = (
            extract_16s_anti_sd_tail(rrna_path, tail_length=13)
            if rrna_path is not None
            else ""
        )
        self.genome_seqs, self.default_contig = load_genome_contigs(args.genome)
        self.genome = (
            self.genome_seqs.get(self.default_contig, "")
            if self.default_contig
            else ""
        )

        self.alternatives = alternatives
        self.alternatives = self.__extract_upstream()

    def _score_upstream(self, upstream_seq: str) -> float:
        if not upstream_seq or not self.rRNA:
            return 0.0
        if self.backend == "free2bind":
            return float(self._run_free_align(upstream_seq))
        if self.backend == "auto":
            try:
                return float(self._run_free_align(upstream_seq))
            except Exception:
                return compute_rbs_free_energy(upstream_seq, self.rRNA)
        return compute_rbs_free_energy(upstream_seq, self.rRNA)

    def _run_free_align(self, upstream_seq: str) -> str:
        if self.freeAlignPath is None:
            self.freeAlignPath = resolve_free2bind_script(
                getattr(self.args, "free2bind_dir", None)
            )
        perl_bin = resolve_executable(None, "perl", env_var="UPROTEINS_PERL")
        res = run_cmd(
            [
                str(perl_bin),
                str(self.freeAlignPath),
                "-e",
                str(upstream_seq),
                str(self.rRNA),
            ],
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    def __get_transcripts(self, transcriptome_gff, assembly):
        if transcriptome_gff is not None and assembly is not None:
            self.subset = "Transcriptome"
            gff = StringTieGFF(gff=transcriptome_gff)
            torfs = gff.get_dict()
            sequences = TranscriptExtractor(assembly=assembly)
            transcripts = sequences.get_transcripts()
            for key, orf in torfs.items():
                if orf.name in transcripts:
                    rna_seq = transcripts[orf.name]
                elif orf.name.startswith("gene-"):
                    rna_seq = transcripts.get(orf.name[5:], "")
                elif orf.name.startswith("rna-"):
                    rna_seq = transcripts.get(orf.name[4:], "")
                elif "uproteins." in orf.name:
                    short_name = ".".join(orf.name.split(".")[:2])
                    rna_seq = transcripts.get(short_name, "")
                else:
                    rna_seq = transcripts.get(key, "")
                orf.transcript = rna_seq
                self.tORFs[orf.name] = orf
        return self

    def __extract_upstream(self):
        """Populate `alt.upstream` (exact <=21 nt strictly upstream of the start codon)
        and `alt.utr_length` while preserving stop-keyed `ORFCollection` groupings."""
        upstream_seqs = {}
        for stop_key, collection in self.alternatives.items():
            out_col = ORFCollection()
            for alt in collection:
                if self.subset == "Transcriptome":
                    tx_seq = getattr(alt, "transcript", None) or ""
                    if not tx_seq and getattr(alt, "transcriptName", None) in self.tORFs:
                        tx_seq = self.tORFs[alt.transcriptName].transcript or ""
                    up_seq, utr_len = extract_transcript_upstream(
                        tx_seq,
                        alt.start,
                        alt.end,
                        alt.strand or "forward",
                        window=21,
                    )
                    alt.upstream = up_seq
                    alt.utr_length = utr_len
                else:
                    contig_seq = resolve_contig_sequence(
                        self.genome_seqs,
                        getattr(alt, "chromosome", None),
                        self.default_contig,
                    )
                    alt.upstream = extract_genomic_upstream_21nt(
                        contig_seq,
                        alt.start,
                        alt.end,
                        alt.strand or "forward",
                        circular=self.circular,
                        window=21,
                    )
                    alt.utr_length = None
                out_col.add_orf(alt)
            upstream_seqs[stop_key] = out_col
        return upstream_seqs

    @staticmethod
    def __complement(seq):
        return complement_dna(seq)

    def get_free_energy(self):
        is_tx = self.subset == "Transcriptome"
        if not self.rRNA:
            for stop_key in self.alternatives:
                for alt in self.alternatives[stop_key]:
                    utr_len = getattr(alt, "utr_length", None)
                    alt.freeEnergy = 0.0
                    if is_tx and utr_len is not None and utr_len < MIN_LEADER_UTR_NT:
                        alt.shineDalgarno = "Leaderless"
                    else:
                        alt.shineDalgarno = "Not inspected (--rrna omitted)"
            return self.alternatives

        alts = {}
        for stop_key, collection in self.alternatives.items():
            out_col = ORFCollection()
            for alt in collection:
                utr_len = getattr(alt, "utr_length", None)
                if is_tx and utr_len is not None and utr_len < MIN_LEADER_UTR_NT:
                    alt.freeEnergy = 0.0
                    alt.shineDalgarno = "Leaderless"
                elif getattr(alt, "upstream", ""):
                    energy = self._score_upstream(alt.upstream)
                    alt.freeEnergy = float(energy)
                    alt.shineDalgarno = classify_rbs_category(
                        alt.freeEnergy,
                        is_transcriptome=is_tx,
                        utr_length=utr_len,
                    )
                else:
                    alt.freeEnergy = 0.0
                    alt.shineDalgarno = classify_rbs_category(
                        0.0,
                        is_transcriptome=is_tx,
                        utr_length=utr_len,
                    )
                out_col.add_orf(alt)
            alts[stop_key] = out_col
        return alts

    def save_data(self):
        out = f"{self.folder}/post_perc/{self.filetype}_results_03.txt"
        self.results.to_csv(out, sep="\t", index=False)
        return self
