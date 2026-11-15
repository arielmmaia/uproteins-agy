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


import pandas as pd
from Bio import SeqIO

from . import ORF, ORFCollection
from ..conversion import Translator, complement_dna
from ..locus import (
    StringTieGFF,
    load_genome_contigs,
    parse_coordinate_token,
    parse_gorf_header,
    parse_torf_header,
    resolve_contig_sequence,
)
from ..transcriptomics import TranscriptExtractor


class AltCodons(object):
    def __init__(self, file, genome, maxsize, subset="Genome", transcriptome_gff=None, assembly=None, testing=False):
        self.subset = subset
        self.testing = testing
        self.tORFs = self.__check_transcriptome(transcriptome_gff, assembly)
        self.maxsize = maxsize
        self.df = pd.read_csv(file, sep='\t')
        self.__check_testing()
        self.coordinates = self.df["Genome Coordinates"].tolist() if not self.df.empty else []
        self.names = self.df["Protein"].tolist() if not self.df.empty else []
        self.proteinSequences = self.df["ORF Sequence"].tolist() if not self.df.empty else []

        self.genome_seqs, self.default_contig = load_genome_contigs(genome)
        self.genome_seq = list(self.genome_seqs.values())

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
                elif 'uproteins' in orf.name:
                    short_rna = '.'.join(orf.name.split(".")[:2])
                    rna_seq = transcripts.get(short_rna, "")
                elif orf.name.startswith('gene-'):
                    rna_seq = transcripts.get(orf.name[5:], "")
                elif orf.name.startswith('rna-'):
                    rna_seq = transcripts.get(orf.name[4:], "")
                else:
                    rna_seq = transcripts.get(gene, "")
                orf.transcript = rna_seq
            return orf_dict
        return {}

    def __split_coords(self, i):
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

        is_rev = ('reverse' in self.names[i]) or (c_strand == '-')
        if is_rev:
            start = hi
            end = lo
            strand = 'reverse'
        else:
            start = lo
            end = hi
            strand = 'forward'
        return int(start), int(end), strand, (contig or self.default_contig)

    def __get_transcript_name(self, entry):
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
        if 'gene' in gene or 'rna' in gene:
            return gene
        splat = gene.split(".")
        return '.'.join(splat[:2])

    def __get_transcript_coordinates(self, entry):
        t_parsed = parse_torf_header(entry)
        if t_parsed is not None:
            return int(t_parsed[1]), int(t_parsed[2]), t_parsed[3]
        splat = entry.split("_")
        coords = splat[len(splat) - 2].split("-")
        start, end = coords[0], coords[1]
        return int(start), int(end), "forward"

    def _get_contig_seq(self, orf) -> str:
        return resolve_contig_sequence(
            self.genome_seqs,
            getattr(orf, "chromosome", None),
            self.default_contig,
        )

    def __fetch_orfs(self):
        """
        :returns a dictionary containing all ORFs with alternative START codons for a given STOP codon.
        """
        alt_check = {}
        alternatives = {}
        print('fetching orfs')
        for i in range(len(self.names)):
            if self.subset == "Genome":
                name = self.names[i]
                start, end, strand, contig = self.__split_coords(i)
                transcript = None
                contig_seq = resolve_contig_sequence(
                    self.genome_seqs, contig, self.default_contig
                )
                if strand == 'forward':
                    seq = contig_seq[start - 1 : end]
                else:
                    seq = complement_dna(contig_seq[end - 1 : start][::-1])
            elif self.subset == "Transcriptome":
                name = self.__get_transcript_name(self.names[i])
                t_start, t_end, t_strand = self.__get_transcript_coordinates(self.names[i])
                tx_obj = self.tORFs[name]
                contig = getattr(tx_obj, "chromosome", None) or self.default_contig
                transcript = tx_obj.transcript
                if t_strand == "reverse":
                    lo_t, hi_t = min(t_start, t_end), max(t_start, t_end)
                    start, end = hi_t, lo_t
                    strand = "reverse"
                    seq = complement_dna(transcript[end - 1 : start][::-1])
                else:
                    lo_t, hi_t = min(t_start, t_end), max(t_start, t_end)
                    start, end = lo_t, hi_t
                    strand = "forward"
                    seq = transcript[start - 1 : end]
            else:
                continue

            orf = ORF(
                name=self.names[i],
                start=int(start),
                end=int(end),
                seq=seq,
                strand=strand,
                chromosome=contig,
                protein_sequence=self.proteinSequences[i],
            )
            orf.transcript = transcript
            orf = self.__fetch_codons(orf)
            orf.transcriptName = name
            identifier = self.__define_identifier(
                end, transcript_name=orf.transcriptName, contig=contig
            )
            if identifier not in alt_check:
                alt_check[identifier] = []
            if identifier not in alternatives:
                alternatives[identifier] = ORFCollection()
            if start not in alt_check[identifier]:
                alternatives[identifier].add_orf(orf)
                alt_check[identifier].append(start)
        print('done fetching')
        return alternatives

    def __define_identifier(self, orf_end, transcript_name=None, contig=None):
        if self.subset == "Genome":
            if len(self.genome_seqs) > 1 and contig:
                return f"{contig}:{orf_end}"
            return orf_end
        return f'{transcript_name}_{orf_end}'

    def sort_by_coordinates(self):
        """ Sorts the alternative ORFs inside self.alternatives by their start codons. """
        print('sorting by coordinates')
        for alts in self.alternatives:
            starts = []
            for alt in self.alternatives[alts]:
                if len(starts) == 0:
                    starts.append(alt)
                else:
                    if alt.strand == 'forward':
                        if starts[0].start > alt.start:
                            starts.insert(0, alt)
                        else:
                            starts.append(alt)
                    else:
                        if starts[0].start < alt.start:
                            starts.insert(0, alt)
                        else:
                            starts.append(alt)
            self.alternatives[alts] = ORFCollection()
            self.alternatives[alts].add_orfs(starts)
        # total = []
        # for stop in self.alternatives:
        #     for i in self.alternatives[stop]:
        #         total.append(f'{i.name} {i.MSPeptides} {i.start_codon} {i.proteinSequence}\n')
        # with open('teste_peptide_sort.txt', 'w') as out:
        #     out.writelines(total)

    def sort_by_atg(self):
        """ Gives priority to ATG when sorting ORFs. """
        print('sorting by atgs')
        for alts in self.alternatives:
            # print(f'Alts: {alts}')
            atgs = []
            for alt in self.alternatives[alts]:
                if len(atgs) == 0:
                    atgs.append(alt)
                else:
                    if alt.start_codon == "ATG":
                        if atgs[0].start_codon != "ATG":  # if the alt start at position 0 is not ATG, changes priority
                            atgs.insert(0, alt)  # to this ORF in the collection
                        else:  # if it is an ATG, check which one is closer to the closest upstream STOP codon
                            if alt.strand == 'forward':
                                if atgs[0].start > alt.start:
                                    atgs.insert(0, alt)
                                else:
                                    atgs.append(alt)
                            elif alt.strand == 'reverse':
                                if atgs[0].start < alt.start:
                                    atgs.insert(0, alt)
                                else:
                                    atgs.append(alt)
                    else:       # checks priority for alternative START codons. In the absence of an ATG, checks which
                        if atgs[0].start_codon != "ATG":  # alternative is closer to the closest upstream STOP codon
                            if alt.strand == 'forward':
                                if atgs[0].start > alt.start:
                                    atgs.insert(0, alt)
                                else:
                                    atgs.append(alt)
                            elif alt.strand == 'reverse':
                                if atgs[0].start < alt.start:
                                    atgs.insert(0, alt)
                                else:
                                    atgs.append(alt)
                        else:
                            atgs.append(alt)
                # print([(alt.name, alt.start, alt.start_codon) for alt in atgs])
            self.alternatives[alts] = ORFCollection()
            self.alternatives[alts].add_orfs(atgs)
            # print([(alt.name, alt.start, alt.start_codon, alt.seq) for alt in self.alternatives[alts]])
        # total = []
        # for stop in self.alternatives:
        #     for i in self.alternatives[stop]:
        #         total.append(f'{i.name} {i.MSPeptides} {i.start_codon} {i.proteinSequence}\n')
        # with open('teste_peptide_sort.txt', 'w') as out:
        #     out.writelines(total)
            # print(len(self.alternatives))

    def __fetch_codons(self, orf):
        """Return the nucleotide sequence of the start codon for `orf`."""
        sequence = self.__check_subset(orf)
        if orf.strand == 'forward':
            s_codon = sequence[orf.start - 1 : orf.start + 2]
        else:
            s_codon = complement_dna(sequence[orf.start - 3 : orf.start][::-1])
        orf.start_codon = s_codon
        return orf

    def __check_subset(self, orf):
        if self.subset == "Genome":
            return self._get_contig_seq(orf)
        return orf.transcript

    def __check_transcript(self, orf):
        if self.subset == "Genome":
            return None
        return orf.transcript

    def extend_orfs(self, args):
        print("extending ORFS")
        new_alts = {}
        for alts in self.alternatives:
            for alt in self.alternatives[alts]:
                sequence = self.__check_subset(alt)
                transcript = self.__check_transcript(alt)
                if alt.strand == 'forward':
                    i = 3
                    extend = True
                    extended = 'same'
                    position = 0
                    seq = "dummy"
                    while extend and (alt.start - i) > 0:
                        s_codon = sequence[alt.start - i - 1 : alt.start - i + 2]
                        real_start = sequence[alt.start + 2 - i : alt.start - i + 5]
                        if real_start in args.starts:
                            extended = real_start
                            position = alt.start - i
                            seq = sequence[alt.start - i + 2 : alt.end]
                            self.__add_extended(
                                new_alts,
                                position,
                                alt,
                                extended,
                                seq,
                                transcript=transcript,
                                transcript_name=alt.transcriptName,
                            )
                        if s_codon in args.stops:
                            extend = False
                        i += 3
                else:
                    i = 3
                    extend = True
                    while extend and (alt.start + i) <= len(sequence):
                        ex_start = sequence[alt.end - 1 : alt.start + i][::-1]
                        ex_seq = complement_dna(ex_start)
                        ex_codon = complement_dna(
                            sequence[alt.start + i - 3 : alt.start + i][::-1]
                        )
                        i += 3
                        if ex_codon in args.starts:
                            self.__add_extended(
                                alt=alt,
                                new_alts=new_alts,
                                s_codon=ex_codon,
                                seq=ex_seq,
                                start_pos=alt.start + i - 6,
                                transcript=transcript,
                                transcript_name=getattr(alt, "transcriptName", None),
                            )
                        if ex_codon in args.stops:
                            extend = False
        self.__add_new_alts(new_alts)
        print('done extending')

    @staticmethod
    def complement(seq):
        return complement_dna(seq)

    def __add_new_alts(self, new_alts):
        for alt in new_alts:
            key = alt if alt in self.alternatives else str(alt)
            if key in self.alternatives:
                self.alternatives[key].add_orfs(new_alts[alt])
        return self

    def __check_length(self, seq, alt, new_alts, i):
        """ Deprecated """
        if i > self.maxsize:
            orf = ORF(
                name='Discard',
                end=alt.end,
                start=alt.start,
                seq=seq,
                strand=alt.strand,
                chromosome=getattr(alt, "chromosome", None),
            )
            orf.shineDalgarno = alt.shineDalgarno
            orf.freeEnergy = alt.freeEnergy
            if alt.end not in new_alts:
                new_alts[alt.end] = [orf]
            else:
                new_alts[alt.end].insert(0, orf)
        return new_alts

    def __add_extended(self, new_alts, start_pos, alt, s_codon, seq, transcript, transcript_name=None):
        orf = ORF(
            name=f'{alt.name[:5]}_extended_{start_pos+3}-{alt.end}_{alt.strand}',
            strand=alt.strand,
            start=start_pos + 3,
            end=alt.end,
            seq=seq,
            transcript=transcript,
            chromosome=getattr(alt, "chromosome", None),
        )
        orf.start_codon = s_codon
        orf.MSPeptides = list(alt.MSPeptides)
        orf.transcriptName = transcript_name
        identifier = self.__define_identifier(
            orf_end=orf.end,
            transcript_name=transcript_name,
            contig=getattr(alt, "chromosome", None),
        )
        if identifier not in new_alts:
            new_alts[identifier] = [orf]
        else:
            new_alts[identifier].append(orf)
        return new_alts

    def sort_by_shine(self):
        """
        Sorts the ORFs by their free energy when binding to the 16S rRNA. If no alternative START of a given STOP has
        any Shine-Dalgarno sequences (free energy >= 0) upstream from their START codon, these ORFs are not sorted at
        all.
        :return: dictionary containing ORFCollections with ORFs sorted by the presence or absence of Shine-Dalgarno
        sequences upstream from their START.
        """
        print('sorting by RBS')
        alternatives = {}
        for stop in self.alternatives:
            ignore = False
            sorted_orfs = [orf for orf in self.alternatives[stop]]
            no_sd = 0  # marks the number of sequences without a SD sequence. If this is equal to the number of
            # alternative starts, change ignore to True. Then, these ORFs should not be sorted by SD.
            for alt in self.alternatives[stop]:
                if alt.freeEnergy < 0:
                    if len(sorted_orfs) > 0:
                        if alt.freeEnergy < sorted_orfs[0].freeEnergy:
                            sorted_orfs.insert(0, alt)
                    else:
                        sorted_orfs.append(alt)
                else:
                    sorted_orfs.append(alt)
                    no_sd += 1
                if no_sd == len(self.alternatives[stop]):
                    ignore = True
            if stop not in alternatives:
                if not ignore:
                    alternatives[stop] = ORFCollection()
                    alternatives[stop].add_orfs(sorted_orfs)
                else:
                    alternatives[stop] = self.alternatives[stop]
            else:
                if not ignore:
                    alternatives[stop].add_orfs(sorted_orfs)
                else:
                    alternatives[stop] = self.alternatives[stop]
        # for stop in alternatives:
            # for alt in alternatives[stop]:
            # print('shine:', [(stop, alt.freeEnergy, alt.name, alt.start, alt.shineDalgarno) for alt in alternatives[stop]])
        # total = []
        # for stop in alternatives:
        #     i = 0
        #     for i in alternatives[stop]:
        #         if i == 0:
        #             total.append('__________________________________________\n')
        #         total.append(f'{i.name} {i.MSPeptides} {i.start_codon} {i.proteinSequence} {i.shineDalgarno} '
        #                      f'{i.freeEnergy}\n')
        #     total.append('__________________________________________\n')
        # with open('teste_peptide_sort.txt', 'w') as out:
        #     out.writelines(total)
        self.alternatives = alternatives
        return alternatives


    def __extract_peptides(self):
        peptides = self.df["Peptide"].tolist()
        # print(peptides)
        def format_pep(pep):
            fixed = ""
            # print(pep)
            for i in pep:
                if i.isalpha():
                    fixed += i
            return fixed

        fixed_peps = []
        for pep in peptides:
            fixed = format_pep(pep)
            fixed_peps.append(fixed)
        alts_with_peps = {}
        for stop in self.alternatives:
            for alt in self.alternatives[stop]:
                for pep in fixed_peps:
                    if pep not in alt.MSPeptides and pep in alt.proteinSequence:
                        # if pep in alt.proteinSequence:
                        alt.MSPeptides.append(pep)
                            # print(alt.MSPeptides, alt.proteinSequence, pep)
                            # break
                if str(stop) not in alts_with_peps:
                    alts_with_peps[str(stop)] = ORFCollection()
                    alts_with_peps[str(stop)].add_orf(alt)
                elif str(stop) in alts_with_peps:
                    alts_with_peps[str(stop)].add_orf(alt)
        return alts_with_peps

    def sort_by_peptides(self):
        print('sorting by ms peptides')
        pep_sorted = {}
        total = []
        for stop in self.alternatives:
            orfs = []
            for alt in self.alternatives[stop]:
                to_trans = Translator(alt.seq)
                alt.proteinSequence = to_trans.translate()
                alt.filter_peptides()
                if len(orfs) > 0:
                    if len(alt.MSPeptides) > len(orfs[0].MSPeptides):
                        orfs.insert(0, alt)
                    else:
                        orfs.append(alt)
                else:
                    orfs.append(alt)
            if stop not in pep_sorted:
                pep_sorted[stop] = ORFCollection()
                pep_sorted[stop].add_orfs(orfs)
            else:
                pep_sorted[stop].add_orfs(orfs)
        """ FOR DEBUGGING """
        for stop in pep_sorted:
            i = 0
            for i in pep_sorted[stop]:
                if i == 0:
                    total.append('__________________________________________\n')
                total.append(f'{i.name} {i.MSPeptides} {i.start_codon} {i.proteinSequence} {i.shineDalgarno} '
                             f'{i.freeEnergy}\n')

            total.append('__________________________________________\n')

        with open('teste_peptide_sort.txt', 'w') as out:
            out.writelines(total)
            # print([('start', orf.name, orf.MSPeptides) for orf in self.alternatives[stop]])

        self.alternatives = pep_sorted
        return pep_sorted

    def get_priorities(self):
        """ :returns the entries that have top priority for their START codons. """
        pep_list = {}
        for stop in self.alternatives:
            # pep_list.append(self.alternatives[stop])
            for orf in self.alternatives[stop]:
                pep_list[stop] = [orf]
                break
        return pep_list
