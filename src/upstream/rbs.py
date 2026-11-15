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


import os

from Bio import SeqIO
import pandas as pd

from ..sequtils.orflib import ORF, ORFCollection
from ..sequtils.locus import StringTieGFF
from ..sequtils.transcriptomics import TranscriptExtractor
from ..utils import REPO_ROOT, resolve_executable, resolve_free2bind_script, run_cmd


class SDInspection(object):
    def __init__(self, args, filetype, folder, alternatives, subset="Genome", transcriptome_gff=None, transcripts=None,
                 testing=False):
        """
        Looks for a Shine-Dalgarno sequence using the script free_align.pl from free2bind package.
        :param args: uProteInS arguments
        :param filetype: either genome or transcriptome. Lowercase.
        :param alternatives: the dictionary containing alternative START codons for a given STOP codon, after extending
        it with AltORF method extend_orfs(), preferably.
        """
        self.args = args
        self.subset = subset
        self.tORFs = {}
        self.testing = testing
        self.__get_transcripts(transcriptome_gff, assembly=transcripts)
        self.rRNA = self.__get_sequences(args.rrna)[::-1][:13]  # one of the RNA strings must be 3'-5'.
        self.filetype = filetype
        self.folder = folder
        self.genome = self.__get_sequences(args.genome)
        self.pyPath = str(REPO_ROOT)
        self.freeAlignPath = None

        self.alternatives = alternatives
        self.alternatives = self.__extract_upstream()

    def _run_free_align(self, upstream_seq: str) -> str:
        if self.freeAlignPath is None:
            self.freeAlignPath = resolve_free2bind_script(
                getattr(self.args, 'free2bind_dir', None)
            )
        perl_bin = resolve_executable(None, 'perl', env_var='UPROTEINS_PERL')
        res = run_cmd(
            [str(perl_bin), str(self.freeAlignPath), '-e', str(upstream_seq), str(self.rRNA)],
            capture_output=True,
            text=True,
        )
        return res.stdout.strip()

    def __get_transcripts(self, transcriptome_gff, assembly):
        """

        :param transcriptome_gff: GFF containing the StringTie assembly
        :param transcripts: fasta file containing the transcripts
        :return:
        """
        if transcriptome_gff is not None and assembly is not None:
            self.subset = "Transcriptome"
            gff = StringTieGFF(gff=transcriptome_gff)
            torfs = gff.get_dict()
            sequences = TranscriptExtractor(assembly=assembly)
            transcripts = sequences.get_transcripts()
            for i in torfs:
                orf = torfs[i]
                if 'gene' in orf.name:
                    name = orf.name[5:]
                elif 'rna' in orf.name:
                    name = orf.name[4:]
                else:
                    splat = orf.name.split(".")
                    name = '.'.join(splat[:2])
                orf.transcript = transcripts[name]
                self.tORFs[orf.name] = orf
        return self

    @staticmethod
    def __get_sequences(fasta):
        records = SeqIO.parse(fasta, 'fasta')
        seqs = [str(record.seq) for record in records][0]
        return seqs

    def __extract_upstream_old(self):
        """
        Extracts the 21 nucleotides upstream from the START codon of each ORF present in self.results.
        :return:
        """
        upstream_seqs = []
        for coords, entry in zip(self.coordinates, self.entries):
            splat_coords = coords.split("-")
            if 'forward' in entry:
                start = int(splat_coords[0])
                upstream = self.genome[start-22: start]  # index starts at 0. Need the 21 nucleotides upstream
                upstream_seqs.append(upstream)
            elif 'reverse' in entry:
                start = int(splat_coords[1])
                upstream = self.__complement(self.genome[start: start+22][::-1])
                upstream_seqs.append(upstream)
        return upstream_seqs

    def __extract_upstream(self):
        """
        Extracts the 21 nucleotides upstream from the START codon of each ORF present in a ORFCollection().
        :return: Dictionary with ORFCollections containing the alternative START codons for a given STOP.
        """
        upstream_seqs = {}
        for stop in self.alternatives:
            for alt in self.alternatives[stop]:
                if alt.strand == 'forward':
                    if self.subset == "Genome":
                        upstream = self.genome[alt.start-22: alt.start]
                    else:
                        if alt.start-22 > 0:
                            upstream = alt.transcript[alt.start-22: alt.start]
                        else:
                            upstream = ''
                    alt.upstream = upstream
                elif alt.strand == 'reverse':
                    upstream = self.__complement(self.genome[alt.start: alt.start+22][::-1])
                    alt.upstream = upstream
                if self.subset == 'Genome':
                    if str(alt.end) not in upstream_seqs:
                        upstream_seqs[str(alt.end)] = ORFCollection()
                        upstream_seqs[str(alt.end)].add_orf(alt)
                    elif str(alt.end) in upstream_seqs:
                        upstream_seqs[str(alt.end)].add_orf(alt)
                elif self.subset == "Transcriptome":
                    identifier = f'{alt.transcriptName}_{alt.end}'
                    if identifier not in upstream_seqs:
                        upstream_seqs[identifier] = ORFCollection()
                        upstream_seqs[identifier].add_orf(alt)
                    elif identifier in upstream_seqs:
                        upstream_seqs[identifier].add_orf(alt)

        return upstream_seqs

    def __define_identifier(self, orf_end, transcript_name=None):
        if self.subset == "Genome":
            identifier = orf_end
        else:
            identifier = f'{transcript_name}_{orf_end}'
        return identifier

    @staticmethod
    def __complement(seq):
        """
        :param seq: sequence to generate the reverse complement for
        :return: complementary sequence
        """
        nucs = {"A": "T", "T": "A", "G": "C", "C": "G"}
        compl = ""
        for nuc in seq:
            compl += nucs[nuc]
        return compl

    def get_free_energy(self):
        alts = {}
        for stop in self.alternatives:
            for alt in self.alternatives[stop]:
                if alt.upstream != '':
                    energy = self._run_free_align(alt.upstream)
                    alt.freeEnergy = float(energy)
                    sd_seq = self.__check_rbs(energy)
                    alt.shineDalgarno = sd_seq
                else:
                    alt.freeEnergy = 0
                    alt.shineDalgarno = self.__check_rbs(0)
                identifier = self.__define_identifier(alt.end, transcript_name=alt.transcriptName)
                if identifier not in alts:
                    alts[identifier] = ORFCollection()
                    alts[identifier].add_orf(alt)
                else:
                    alts[identifier].add_orf(alt)
        return alts

    def get_free_energy_old(self):
        """
        Runs free_align.pl to check which part of the sequence has the minimum free energy when binding to the rRNA,
        i.e, has the most stable binding. """
        energies = []
        sd = []
        for upstream_seq in self.upstreamSequences:
            energy = self._run_free_align(upstream_seq)
            energies.append(f'{energy}')
            sd_seq = self.__check_rbs(energy)
            sd.append(sd_seq)
        self.results.insert(5, "Free Energy", energies)
        self.results.insert(6, "Shine-Dalgarno", sd)
        return self

    @staticmethod
    def __check_rbs(rbs):
        if float(rbs) >= -3.4535:
            sd_seq = "Leaderless"
        elif -8.4 < float(rbs) < -3.4535:
            sd_seq = "Present. Low to moderate binding."
        elif float(rbs) <= -8.4:
            sd_seq = "Present. Strong binding."
        else:
            sd_seq = "SD absent"
        return sd_seq

    def save_data(self):
        out = f'{self.folder}/post_perc/{self.filetype}_results_03.txt'
        self.results.to_csv(out, sep='\t', index=False)
        return self


