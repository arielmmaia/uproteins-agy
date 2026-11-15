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


import re

import pandas as pd

from . import ORF, ORFCollection
from ..conversion import Translator, complement_dna, reverse_complement_dna
from ..locus import (
    StringTieGFF,
    format_orf_display_name,
    load_genome_contigs,
    parse_coordinate_token,
    parse_gorf_header,
    parse_torf_header,
    resolve_contig_sequence,
)
from ..transcriptomics import TranscriptExtractor
from ...upstream.duplex_dg import MODERATE_SD_THRESHOLD, STRONG_SD_THRESHOLD


_FLANK_RE = re.compile(r"^[A-Za-z\-]\.(.*)\.[A-Za-z\-]$")
_MOD_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_NON_ALPHA_RE = re.compile(r"[^A-Za-z]")


def _clean_ms_peptide(raw_pep: object) -> str:
    if pd.isna(raw_pep):
        return ""
    s = str(raw_pep).strip()
    m = _FLANK_RE.match(s)
    if m:
        s = m.group(1)
    s = _MOD_RE.sub("", s)
    return _NON_ALPHA_RE.sub("", s).upper()


def _translate_candidate_dna(
    dna_seq: str,
    stops: tuple[str, ...] = ("TAA", "TAG", "TGA"),
    genetic_code: int = 11,
) -> str:
    raw_aa = Translator(
        dna_seq, stops=stops, genetic_code=genetic_code
    ).translate(initiator=True)
    return raw_aa.split("*", 1)[0]


def _orf_covers_peptide(protein_seq: str, peptide: str) -> bool:
    if not protein_seq or not peptide:
        return False
    return peptide in protein_seq


def _format_candidate_summary(candidates: list[ORF]) -> str:
    parts: list[str] = []
    for cand in candidates:
        lo = min(int(cand.start), int(cand.end))
        hi = max(int(cand.start), int(cand.end))
        codon = cand.start_codon or "NA"
        dg = cand.freeEnergy
        dg_str = f"{float(dg):.2f}" if dg is not None else "NA"
        parts.append(f"{codon}:{lo}-{hi}(dG={dg_str})")
    return ";".join(parts)


class AltCodons(object):
    """Bounded, terminating bacterial start-codon candidate enumeration and
    deterministic biological start-site ranking."""

    def __init__(
        self,
        file,
        genome,
        maxsize,
        subset="Genome",
        transcriptome_gff=None,
        assembly=None,
        testing=False,
    ):
        self.subset = subset
        self.testing = testing
        self.tORFs = self.__check_transcriptome(transcriptome_gff, assembly)
        self.maxsize = int(maxsize) if maxsize is not None else 300
        self.stops: tuple[str, ...] = ("TAA", "TAG", "TGA")
        self.genetic_code: int = 11
        self.start_selection: str = "upstream"

        self.df = pd.read_csv(file, sep="\t")
        self.__check_testing()
        self.coordinates = (
            self.df["Genome Coordinates"].tolist() if not self.df.empty else []
        )
        self.names = self.df["Protein"].tolist() if not self.df.empty else []
        self.proteinSequences = (
            self.df["ORF Sequence"].tolist() if not self.df.empty else []
        )

        self.genome_seqs, self.default_contig = load_genome_contigs(genome)
        self.genome_seq = list(self.genome_seqs.values())

        self.locus_peptides: dict[str, tuple[str, ...]] = {}
        self.locus_source_names: dict[str, set[str]] = {}
        self.alternatives = self.__fetch_orfs()
        self.alternatives = self.__extract_peptides()

    def __check_testing(self):
        if self.testing:
            self.df = self.df.head(20)

    def __check_transcriptome(self, gff, transcripts_fasta):
        if gff is not None and transcripts_fasta is not None:
            self.subset = "Transcriptome"
            assembly = TranscriptExtractor(assembly=transcripts_fasta)
            transcripts = assembly.get_transcripts()
            gff_reader = StringTieGFF(gff)
            orf_dict = gff_reader.get_dict()
            for gene, orf in orf_dict.items():
                if orf.name in transcripts:
                    rna_seq = transcripts[orf.name]
                elif "uproteins" in orf.name:
                    short_rna = ".".join(orf.name.split(".")[:2])
                    rna_seq = transcripts.get(short_rna, "")
                elif orf.name.startswith("gene-"):
                    rna_seq = transcripts.get(orf.name[5:], "")
                elif orf.name.startswith("rna-"):
                    rna_seq = transcripts.get(orf.name[4:], "")
                else:
                    rna_seq = transcripts.get(gene, "")
                orf.transcript = rna_seq
            return orf_dict
        return {}

    def __split_coords(self, i: int) -> tuple[int, int, str, str]:
        parsed_coord = parse_coordinate_token(self.coordinates[i])
        g_parsed = parse_gorf_header(self.names[i])
        contig = None
        if g_parsed is not None and g_parsed[0]:
            contig = g_parsed[0]
        if parsed_coord is not None:
            c_contig, lo, hi, c_strand = parsed_coord
            if c_contig:
                contig = c_contig
        else:
            splat = str(self.coordinates[i]).split("-")
            a, b = int(splat[0]), int(splat[1])
            lo, hi = (a, b) if a <= b else (b, a)
            c_strand = None

        is_rev = ("reverse" in str(self.names[i])) or (c_strand == "-")
        if is_rev:
            start = hi
            end = lo
            strand = "reverse"
        else:
            start = lo
            end = hi
            strand = "forward"
        return int(start), int(end), strand, (contig or self.default_contig)

    def __get_transcript_name(self, entry: str) -> str:
        t_parsed = parse_torf_header(entry)
        if t_parsed is not None:
            tx_id = t_parsed[0]
            if tx_id in self.tORFs:
                return tx_id
            if tx_id.startswith("uproteins."):
                short_id = ".".join(tx_id.split(".")[:2])
                if short_id in self.tORFs:
                    return short_id
            return tx_id
        gene = entry.split("_")[1]
        if "gene" in gene or "rna" in gene:
            return gene
        splat = gene.split(".")
        return ".".join(splat[:2])

    def __get_transcript_coordinates(self, entry: str) -> tuple[int, int, str]:
        t_parsed = parse_torf_header(entry)
        if t_parsed is not None:
            return int(t_parsed[1]), int(t_parsed[2]), t_parsed[3]
        splat = entry.split("_")
        coords = splat[len(splat) - 2].split("-")
        start, end = coords[0], coords[1]
        return int(start), int(end), "forward"

    def _get_contig_seq(self, orf: ORF) -> str:
        return resolve_contig_sequence(
            self.genome_seqs,
            getattr(orf, "chromosome", None),
            self.default_contig,
        )

    def __define_identifier(
        self, orf_end: int, transcript_name: str | None = None, contig: str | None = None, strand: str = "forward"
    ) -> str:
        if self.subset == "Genome":
            base = (
                f"{contig}:{orf_end}"
                if (len(self.genome_seqs) > 1 and contig)
                else str(orf_end)
            )
            return f"{base}(-)" if strand == "reverse" else base
        base = f"{transcript_name}_{orf_end}"
        return f"{base}(-)" if strand == "reverse" else base

    def _extract_orf_dna(self, sequence: str, start: int, end: int, strand: str) -> str:
        if strand == "forward":
            return sequence[int(start) - 1 : int(end)]
        return reverse_complement_dna(sequence[int(end) - 1 : int(start)])

    def __fetch_codons(self, orf: ORF) -> ORF:
        sequence = self.__check_subset(orf)
        if orf.strand == "forward":
            s_codon = sequence[int(orf.start) - 1 : int(orf.start) + 2].upper()
        else:
            s_codon = reverse_complement_dna(sequence[int(orf.start) - 3 : int(orf.start)])
        orf.start_codon = s_codon
        return orf

    def __check_subset(self, orf: ORF) -> str:
        if self.subset == "Genome":
            return self._get_contig_seq(orf)
        return orf.transcript or ""

    def __fetch_orfs(self) -> dict[str, ORFCollection]:
        alt_seen_starts: dict[str, set[tuple[int, int, str]]] = {}
        alternatives: dict[str, ORFCollection] = {}
        raw_peps = (
            self.df["Peptide"].tolist()
            if "Peptide" in self.df.columns and not self.df.empty
            else [""] * len(self.names)
        )
        locus_pep_lists: dict[str, list[str]] = {}

        for i in range(len(self.names)):
            raw_prot_name = str(self.names[i]).strip()
            if self.subset == "Genome":
                start, end, strand, contig = self.__split_coords(i)
                tx_name = None
                transcript = None
                contig_seq = resolve_contig_sequence(
                    self.genome_seqs, contig, self.default_contig
                )
                seq = self._extract_orf_dna(contig_seq, start, end, strand)
            else:
                tx_name = self.__get_transcript_name(raw_prot_name)
                t_start, t_end, t_strand = self.__get_transcript_coordinates(raw_prot_name)
                tx_obj = self.tORFs[tx_name]
                contig = getattr(tx_obj, "chromosome", None) or self.default_contig
                transcript = tx_obj.transcript or ""
                lo_t, hi_t = min(t_start, t_end), max(t_start, t_end)
                if t_strand == "reverse":
                    start, end, strand = hi_t, lo_t, "reverse"
                else:
                    start, end, strand = lo_t, hi_t, "forward"
                seq = self._extract_orf_dna(transcript, start, end, strand)

            identifier = self.__define_identifier(
                end, transcript_name=tx_name, contig=contig, strand=strand
            )
            self.locus_source_names.setdefault(identifier, set()).add(raw_prot_name)

            pep_clean = _clean_ms_peptide(raw_peps[i])
            if pep_clean:
                pep_bucket = locus_pep_lists.setdefault(identifier, [])
                if pep_clean not in pep_bucket:
                    pep_bucket.append(pep_clean)

            # Never carry comma-concatenated "ORF Sequence" strings; translate cleanly
            translated_prot = _translate_candidate_dna(
                seq, stops=self.stops, genetic_code=self.genetic_code
            )
            if not translated_prot and i < len(self.proteinSequences):
                first_tok = str(self.proteinSequences[i]).split(",")[0].strip().rstrip("*")
                translated_prot = first_tok

            orf = ORF(
                name=raw_prot_name,
                start=int(start),
                end=int(end),
                seq=seq,
                strand=strand,
                chromosome=contig,
                transcript=transcript,
                protein_sequence=translated_prot,
            )
            orf.transcriptName = tx_name
            orf = self.__fetch_codons(orf)

            if identifier not in alternatives:
                col = ORFCollection()
                col.source_protein_names = self.locus_source_names[identifier]
                alternatives[identifier] = col
                alt_seen_starts[identifier] = set()

            key = (int(start), int(end), strand)
            if key not in alt_seen_starts[identifier]:
                alternatives[identifier].add_orf(orf)
                alt_seen_starts[identifier].add(key)

        for identifier, p_list in locus_pep_lists.items():
            self.locus_peptides[identifier] = tuple(p_list)

        return alternatives

    def __extract_peptides(self) -> dict[str, ORFCollection]:
        all_peps: list[str] = []
        if "Peptide" in self.df.columns and not self.df.empty:
            for raw_p in self.df["Peptide"].tolist():
                cp = _clean_ms_peptide(raw_p)
                if cp and cp not in all_peps:
                    all_peps.append(cp)

        for stop_id, collection in self.alternatives.items():
            locus_list = list(self.locus_peptides.get(stop_id, ()))
            for alt in collection:
                prot = alt.proteinSequence or ""
                for p in all_peps:
                    if p not in locus_list and _orf_covers_peptide(prot, p):
                        locus_list.append(p)
            locus_tuple = tuple(locus_list)
            self.locus_peptides[stop_id] = locus_tuple
            for alt in collection:
                prot = alt.proteinSequence or ""
                # Fresh list per ORF — NEVER alias across candidates!
                alt.MSPeptides = [p for p in locus_tuple if _orf_covers_peptide(prot, p)]
            collection.source_protein_names = self.locus_source_names.get(
                stop_id, {a.name for a in collection if a.name}
            )
        return self.alternatives

    def _build_candidate_orf(
        self,
        template: ORF,
        start_pos: int,
        end_pos: int,
        sequence: str,
        locus_peps: tuple[str, ...],
    ) -> ORF:
        dna_seq = self._extract_orf_dna(sequence, start_pos, end_pos, template.strand)
        prot_seq = _translate_candidate_dna(
            dna_seq, stops=self.stops, genetic_code=self.genetic_code
        )
        base_prefix = format_orf_display_name(template.name or "ORF")
        lo_c, hi_c = min(int(start_pos), int(end_pos)), max(int(start_pos), int(end_pos))
        if int(start_pos) == int(template.start) and int(end_pos) == int(template.end):
            cand_name = template.name
        else:
            cand_name = f"{base_prefix}_extended_{lo_c}-{hi_c}_{template.strand}"

        cand = ORF(
            name=cand_name,
            start=int(start_pos),
            end=int(end_pos),
            seq=dna_seq,
            strand=template.strand,
            chromosome=getattr(template, "chromosome", None),
            transcript=getattr(template, "transcript", None),
            protein_sequence=prot_seq,
        )
        cand.transcriptName = getattr(template, "transcriptName", None)
        cand = self.__fetch_codons(cand)
        cand.MSPeptides = [p for p in locus_peps if _orf_covers_peptide(prot_seq, p)]
        return cand

    def extend_orfs(self, args):
        """Enumerate all valid in-frame start codons (both upstream up to the nearest
        in-frame stop / `maxsize` limit AND internal starts upstream of the MS peptides)
        with guaranteed termination on both strands."""
        starts_set = {
            s.upper()
            for s in getattr(
                args, "starts", ("TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG")
            )
        }
        stops_raw = getattr(args, "stops", ("TAA", "TAG", "TGA"))
        self.stops = tuple(s.upper() for s in stops_raw)
        stops_set = set(self.stops)
        self.genetic_code = int(getattr(args, "genetic_code", 11) or 11)
        self.start_selection = str(
            getattr(args, "start_selection", "upstream") or "upstream"
        ).lower()
        maxsize_nt = int(getattr(args, "maxsize", self.maxsize) or self.maxsize or 300)
        minsize_nt = int(getattr(args, "minsize", 18) or 18)

        updated_collections: dict[str, ORFCollection] = {}
        for stop_id, collection in self.alternatives.items():
            locus_peps = self.locus_peptides.get(stop_id, ())
            source_names = getattr(
                collection,
                "source_protein_names",
                {a.name for a in collection if a.name},
            )
            candidates_by_pos: dict[tuple[int, int, str], ORF] = {}

            for alt in collection:
                sequence = self.__check_subset(alt)
                seq_len = len(sequence)
                if seq_len < 3:
                    key = (int(alt.start), int(alt.end), alt.strand)
                    candidates_by_pos[key] = alt
                    continue

                if alt.strand == "forward":
                    stop_end = min(seq_len, int(alt.end))
                    orig_start = max(1, int(alt.start))
                    has_stop = (
                        stop_end >= 3
                        and sequence[stop_end - 3 : stop_end].upper() in stops_set
                    )
                    stop_pad = 3 if has_stop else 0
                    orig_span = stop_end - orig_start + 1
                    max_span = max(maxsize_nt + stop_pad, orig_span)
                    min_span = minsize_nt + stop_pad

                    # Always retain the original MS-matched candidate
                    orig_cand = self._build_candidate_orf(
                        alt, orig_start, stop_end, sequence, locus_peps
                    )
                    candidates_by_pos[(orig_start, stop_end, "forward")] = orig_cand

                    # 1. Walk UPSTREAM in 3-nt steps down to 1-based position 1 (inclusive!)
                    pos = orig_start - 3
                    while pos >= 1:
                        if (stop_end - pos + 1) > max_span:
                            break
                        codon = sequence[pos - 1 : pos + 2].upper()
                        if len(codon) < 3 or codon in stops_set:
                            break
                        if codon in starts_set:
                            key = (pos, stop_end, "forward")
                            if key not in candidates_by_pos:
                                candidates_by_pos[key] = self._build_candidate_orf(
                                    alt, pos, stop_end, sequence, locus_peps
                                )
                        pos -= 3

                    # 2. Walk DOWNSTREAM (internal starts) in 3-nt steps while >= min_span
                    pos = orig_start + 3
                    max_pos = stop_end - min_span + 1
                    while pos <= max_pos:
                        codon = sequence[pos - 1 : pos + 2].upper()
                        if len(codon) < 3 or codon in stops_set:
                            break
                        if codon in starts_set:
                            key = (pos, stop_end, "forward")
                            if key not in candidates_by_pos:
                                candidates_by_pos[key] = self._build_candidate_orf(
                                    alt, pos, stop_end, sequence, locus_peps
                                )
                        pos += 3

                else:
                    # Reverse strand: alt.start is hi (5' start on minus strand), alt.end is lo (3' stop)
                    lo = max(1, min(int(alt.start), int(alt.end)))
                    orig_hi = min(seq_len, max(int(alt.start), int(alt.end)))
                    has_stop = (
                        (lo + 2) <= seq_len
                        and reverse_complement_dna(sequence[lo - 1 : lo + 2]) in stops_set
                    )
                    stop_pad = 3 if has_stop else 0
                    orig_span = orig_hi - lo + 1
                    max_span = max(maxsize_nt + stop_pad, orig_span)
                    min_span = minsize_nt + stop_pad

                    orig_cand = self._build_candidate_orf(
                        alt, orig_hi, lo, sequence, locus_peps
                    )
                    candidates_by_pos[(orig_hi, lo, "reverse")] = orig_cand

                    # 1. Walk UPSTREAM on minus strand (increasing genomic coordinate by +3)
                    #    Mathematically bounded by `pos_hi <= seq_len` -> NEVER infinite-loops!
                    pos_hi = orig_hi + 3
                    while pos_hi <= seq_len:
                        if (pos_hi - lo + 1) > max_span:
                            break
                        codon = reverse_complement_dna(sequence[pos_hi - 3 : pos_hi])
                        if len(codon) < 3 or codon in stops_set:
                            break
                        if codon in starts_set:
                            key = (pos_hi, lo, "reverse")
                            if key not in candidates_by_pos:
                                candidates_by_pos[key] = self._build_candidate_orf(
                                    alt, pos_hi, lo, sequence, locus_peps
                                )
                        pos_hi += 3

                    # 2. Walk DOWNSTREAM on minus strand (decreasing genomic coordinate by -3)
                    pos_hi = orig_hi - 3
                    min_pos_hi = lo + min_span - 1
                    while pos_hi >= min_pos_hi:
                        codon = reverse_complement_dna(sequence[pos_hi - 3 : pos_hi])
                        if len(codon) < 3 or codon in stops_set:
                            break
                        if codon in starts_set:
                            key = (pos_hi, lo, "reverse")
                            if key not in candidates_by_pos:
                                candidates_by_pos[key] = self._build_candidate_orf(
                                    alt, pos_hi, lo, sequence, locus_peps
                                )
                        pos_hi -= 3

            # Strictly prevent N-terminal MS peptide truncation (e.g. Mtb Rv2975a):
            # retain only candidates that cover the maximum achievable count of MS peptides!
            all_cands = list(candidates_by_pos.values())
            max_pep_cov = max((len(c.MSPeptides) for c in all_cands), default=0)
            if max_pep_cov > 0:
                valid_cands = [c for c in all_cands if len(c.MSPeptides) == max_pep_cov]
            else:
                valid_cands = all_cands

            new_col = ORFCollection().add_orfs(valid_cands)
            new_col.source_protein_names = source_names
            updated_collections[stop_id] = new_col

        self.alternatives = updated_collections
        return self

    def _deduplicate_collection(self, collection: ORFCollection) -> list[ORF]:
        seen: set[tuple[int, int, str]] = set()
        unique_orfs: list[ORF] = []
        for orf in collection:
            key = (int(orf.start), int(orf.end), str(orf.strand))
            if key not in seen:
                seen.add(key)
                unique_orfs.append(orf)
        return unique_orfs

    def _coordinate_distance_score(self, alt: ORF) -> int:
        span = abs(int(alt.end) - int(alt.start)) + 1
        return -span if self.start_selection == "proximal" else span

    def _candidate_rank_key(
        self, alt: ORF, *, include_rbs: bool = True
    ) -> tuple[int, int, int, int, float, int]:
        pep_count = len(getattr(alt, "MSPeptides", ()) or ())
        is_atg = 1 if str(getattr(alt, "start_codon", "")).upper() == "ATG" else 0

        if include_rbs:
            raw_dg = getattr(alt, "freeEnergy", None)
            dg = float(raw_dg) if raw_dg is not None and pd.notna(raw_dg) else 0.0
            sd_label = str(getattr(alt, "shineDalgarno", "") or "")
            is_leaderless = sd_label == "Leaderless"
            has_func_rbs = 1 if (dg <= MODERATE_SD_THRESHOLD or is_leaderless) else 0
            if dg <= STRONG_SD_THRESHOLD:
                rbs_bucket = 2
            elif dg <= MODERATE_SD_THRESHOLD or is_leaderless:
                rbs_bucket = 1
            else:
                rbs_bucket = 0
            # Never let thermal noise (e.g. -0.5 kcal/mol > -3.4535) override ATG or coordinates!
            func_dg_strength = -dg if dg <= MODERATE_SD_THRESHOLD else 0.0
        else:
            has_func_rbs = 0
            rbs_bucket = 0
            func_dg_strength = 0.0

        return (
            pep_count,
            has_func_rbs,
            is_atg,
            rbs_bucket,
            func_dg_strength,
            self._coordinate_distance_score(alt),
        )

    def sort_by_coordinates(self):
        """Deterministically order deduplicated start candidates by coordinate span."""
        for stop_id, collection in list(self.alternatives.items()):
            source_names = getattr(
                collection,
                "source_protein_names",
                {a.name for a in collection if a.name},
            )
            ordered = sorted(
                self._deduplicate_collection(collection),
                key=lambda o: self._coordinate_distance_score(o),
                reverse=True,
            )
            new_col = ORFCollection().add_orfs(ordered)
            new_col.source_protein_names = source_names
            self.alternatives[stop_id] = new_col
        return self.alternatives

    def sort_by_atg(self):
        """Deterministically prioritize canonical `ATG` starts over non-`ATG` starts."""
        for stop_id, collection in list(self.alternatives.items()):
            source_names = getattr(
                collection,
                "source_protein_names",
                {a.name for a in collection if a.name},
            )
            ordered = sorted(
                self._deduplicate_collection(collection),
                key=lambda o: (
                    1 if str(getattr(o, "start_codon", "")).upper() == "ATG" else 0,
                    self._coordinate_distance_score(o),
                ),
                reverse=True,
            )
            new_col = ORFCollection().add_orfs(ordered)
            new_col.source_protein_names = source_names
            self.alternatives[stop_id] = new_col
        return self.alternatives

    def sort_by_shine(self):
        """Deterministically rank candidates using functional Shine-Dalgarno / Leaderless
        evidence (`dG <= -3.4535 kcal/mol` or true `< 5 nt` Leaderless) without 2x list
        duplication or thermal-noise overrides."""
        for stop_id, collection in list(self.alternatives.items()):
            source_names = getattr(
                collection,
                "source_protein_names",
                {a.name for a in collection if a.name},
            )
            ordered = sorted(
                self._deduplicate_collection(collection),
                key=lambda o: self._candidate_rank_key(o, include_rbs=True),
                reverse=True,
            )
            new_col = ORFCollection().add_orfs(ordered)
            new_col.source_protein_names = source_names
            self.alternatives[stop_id] = new_col
        return self.alternatives

    def sort_by_peptides(self):
        """Finalize candidate translation, verify 100% MS-peptide retention, and sort
        by the full 4-criteria hierarchy without writing debug files to disk."""
        for stop_id, collection in list(self.alternatives.items()):
            locus_peps = self.locus_peptides.get(stop_id, ())
            source_names = getattr(
                collection,
                "source_protein_names",
                {a.name for a in collection if a.name},
            )
            deduped = self._deduplicate_collection(collection)
            for alt in deduped:
                if alt.seq:
                    alt.proteinSequence = _translate_candidate_dna(
                        alt.seq, stops=self.stops, genetic_code=self.genetic_code
                    )
                prot = alt.proteinSequence or ""
                if locus_peps:
                    alt.MSPeptides = [
                        p for p in locus_peps if _orf_covers_peptide(prot, p)
                    ]
                else:
                    alt.filter_peptides()

            has_any_rbs = any(
                getattr(a, "freeEnergy", None) is not None for a in deduped
            )
            ordered = sorted(
                deduped,
                key=lambda o: self._candidate_rank_key(o, include_rbs=has_any_rbs),
                reverse=True,
            )
            max_cov = max((len(a.MSPeptides) for a in ordered), default=0)
            if max_cov > 0:
                ordered = [a for a in ordered if len(a.MSPeptides) == max_cov]

            if ordered:
                summary = _format_candidate_summary(ordered)
                for cand in ordered:
                    cand.alternative_starts = summary

            new_col = ORFCollection().add_orfs(ordered)
            new_col.source_protein_names = source_names
            self.alternatives[stop_id] = new_col
        return self.alternatives

    def get_priorities(self) -> dict[str, list[ORF]]:
        """Return the top-ranked representative start-codon ORF for each stop locus."""
        pep_list: dict[str, list[ORF]] = {}
        for stop_id, collection in self.alternatives.items():
            for orf in collection:
                pep_list[stop_id] = [orf]
                break
        return pep_list

    @staticmethod
    def complement(seq: str) -> str:
        return complement_dna(seq)
