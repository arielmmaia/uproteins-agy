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
import pathlib
import shutil
import subprocess

from src import cli


class Error(Exception):
    pass


class ReadError(Error):
    def __init__(
        self,
        message="Inconsistent read information. Provide a set of paired-end "
        "reads, or a set of single end reads."
    ):
        self.message = message
        super().__init__(self.message)


def hisat2_strand_args(strandness: str | None) -> list[str]:
    """Return HISAT2 strand-specificity CLI arguments ([] when unstranded)."""
    if strandness is None:
        return []
    if strandness in ("RF", "FR", "R", "F"):
        return ["--rna-strandness", strandness]
    raise ValueError(f"Unsupported HISAT2 strandness value: {strandness!r}")


def stringtie_strand_args(strandness: str | None) -> list[str]:
    """Return StringTie strand-specificity CLI arguments ([] when unstranded).

    Both paired-end first-strand ('RF') and single-end first-strand/dUTP ('R')
    use StringTie's '--rf'; both 'FR' and 'F' use '--fr'.
    """
    if strandness is None:
        return []
    if strandness in ("RF", "R"):
        return ["--rf"]
    if strandness in ("FR", "F"):
        return ["--fr"]
    raise ValueError(f"Unsupported StringTie strandness value: {strandness!r}")


class ReadMapper:
    def __init__(self, args):
        self.args = args
        self.gtf = args.gtf
        self.genome = args.genome
        self.threads = self.args.threads
        self.memory = self.args.memory
        self.allow_splicing = bool(getattr(self.args, 'allow_splicing', False))
        self.is_paired = self._check_read_type()
        self.strandness = self.args.strandness
        if self.is_paired and (
            self.args.reads1 is None
            or self.args.reads2 is None
            or len(self.args.reads1) != len(self.args.reads2)
        ):
            raise ReadError(
                "Paired-end reads (--reads1 and --reads2) must both be provided "
                "with identical file counts."
            )

    def _check_read_type(self):
        if self.args.single is None:
            return True
        return False

    def create_indexes(self):
        from src.utils import resolve_executable

        hisat2_build = resolve_executable(None, 'hisat2-build', env_var='UPROTEINS_HISAT2_BUILD')
        cmd_index = [str(hisat2_build)]
        if self.threads is not None:
            cmd_index.extend(['-p', str(self.threads)])
        cmd_index.extend([str(self.genome), 'genome'])
        result = subprocess.run(cmd_index)
        if (code := result.returncode) != 0:
            err = f'hisat2-build finished with non-zero return value: {code}'
            cli.exit(3, err)
        return self

    def _build_hisat2_cmd(self, hisat2_bin: pathlib.Path, read_data) -> list[str]:
        cmd_hisat = [
            str(hisat2_bin),
            '-x', 'genome',
            '--dta',
        ]
        if not self.allow_splicing:
            cmd_hisat.append('--no-spliced-alignment')
        if self.threads is not None:
            cmd_hisat.extend(['-p', str(self.threads)])
        cmd_hisat.extend(hisat2_strand_args(self.strandness))

        if self.is_paired:
            read_one, read_two = read_data
            cmd_hisat.extend(['-1', str(read_one), '-2', str(read_two)])
        else:
            cmd_hisat.extend(['-U', str(read_data)])
        return cmd_hisat

    def _build_samtools_sort_cmd(self, samtools_bin: pathlib.Path, read_name: str) -> list[str]:
        cmd_sort = [
            str(samtools_bin),
            'sort',
            '-o',
            f'HISAT/bam/{read_name}.bam',
        ]
        if self.threads is not None:
            cmd_sort.extend(['-@', str(self.threads)])
        if self.memory is not None:
            cmd_sort.extend(['-m', self.memory])
        return cmd_sort

    @staticmethod
    def _stream_align_and_sort(cmd_hisat: list[str], cmd_sort: list[str]) -> None:
        """Stream HISAT2 SAM output directly into ``samtools sort`` via an OS pipe."""
        with subprocess.Popen(cmd_hisat, stdout=subprocess.PIPE) as hisat_proc:
            assert hisat_proc.stdout is not None
            with subprocess.Popen(cmd_sort, stdin=hisat_proc.stdout) as sort_proc:
                # Allow sort_proc to own the read end so SIGPIPE propagates cleanly
                hisat_proc.stdout.close()
                sort_code = sort_proc.wait()
            hisat_code = hisat_proc.wait()

        if hisat_code != 0:
            cli.exit(3, f'hisat2 finished with non-zero return value: {hisat_code}')
        if sort_code != 0:
            cli.exit(3, f'samtools sort finished with non-zero return value: {sort_code}')

    def align_reads(self):
        from src.utils import resolve_executable

        hisat2_bin = resolve_executable(None, 'hisat2', env_var='UPROTEINS_HISAT2')
        samtools_bin = resolve_executable(None, 'samtools', env_var='UPROTEINS_SAMTOOLS')
        pathlib.Path('HISAT/bam').mkdir(exist_ok=True, parents=True)

        reads = self.args.single
        if self.is_paired:
            reads = zip(self.args.reads1, self.args.reads2)

        for i, read_data in enumerate(reads):
            read_name = f'aligned_{i}'
            cmd_hisat = self._build_hisat2_cmd(hisat2_bin, read_data)
            cmd_sort = self._build_samtools_sort_cmd(samtools_bin, read_name)
            self._stream_align_and_sort(cmd_hisat, cmd_sort)

        return self


class TranscriptAssembly:
    def __init__(self, args):
        self.args = args
        self.threads = args.threads
        self.gtf = self.args.gtf
        self.strand_args = stringtie_strand_args(self.args.strandness)
        self.strandness = self.strand_args[0] if self.strand_args else None

    def _check_strand(self):
        return self.strand_args[0] if self.strand_args else None

    def assemble(self):
        """Run StringTie to assemble transcripts."""
        from src.utils import resolve_executable

        stringtie_bin = resolve_executable(None, 'stringtie', env_var='UPROTEINS_STRINGTIE')
        path = pathlib.Path('StringTie/gtf_intermediates')
        path.mkdir(exist_ok=True, parents=True)
        bam_dir = pathlib.Path("HISAT/bam")
        bam_files = sorted(
            entry for entry in bam_dir.iterdir()
            if entry.is_file() and entry.suffix == '.bam'
        )
        for bam_path in bam_files:
            stem = bam_path.stem
            cmd_str = [
                str(stringtie_bin),
                f'HISAT/bam/{bam_path.name}',
                '-o', f'StringTie/gtf_intermediates/{stem}.gtf',
                '-G', str(self.gtf),
                *self.strand_args,
                '-l', 'uproteins',
                '-m', '30',
                '-A', f'StringTie/gene_abundances_{stem}.tab',
            ]
            if self.threads is not None:
                cmd_str.extend(['-p', str(self.threads)])

            result = subprocess.run(cmd_str)
            if (code := result.returncode) != 0:
                err = f'stringtie finished with non-zero return value: {code}'
                cli.exit(3, err)
        return self

    def create_gtf_list(self):
        gtf_dir = pathlib.Path("StringTie/gtf_intermediates")
        all_gtfs = [
            f"StringTie/gtf_intermediates/{entry.name}\n"
            for entry in sorted(gtf_dir.iterdir())
            if entry.is_file() and entry.suffix == '.gtf'
        ]
        with open("StringTie/gtf_list.txt", 'w') as out:
            out.writelines(all_gtfs)
        return self

    def merge_transcripts(self):
        from src.utils import resolve_executable

        stringtie_bin = resolve_executable(None, 'stringtie', env_var='UPROTEINS_STRINGTIE')
        cmd_str = [
            str(stringtie_bin),
            '--merge',
            'StringTie/gtf_list.txt',
            '-o', 'assembled.gtf',
            '-G', str(self.gtf),
            '-m', '30',
            '-l', 'uproteins'
        ]
        if self.threads is not None:
            cmd_str.extend(['-p', str(self.threads)])

        result = subprocess.run(cmd_str)
        if (code := result.returncode) != 0:
            err = (
                'stringtie --merge finished with non-zero '
                f'return value: {code}'
            )
            cli.exit(3, err)
        gtf_path = 'assembled.gtf'
        return gtf_path


class CompareTranscripts:
    def __init__(self, args, assembled_gtf):
        from src.utils import resolve_executable

        self.assembled_gtf = assembled_gtf
        self.args = args
        self.gffcompare_path = resolve_executable(
            getattr(self.args, 'gffcompare_path', None),
            'gffcompare',
            env_var='UPROTEINS_GFFCOMPARE',
            cli_flag='--gffcompare_path',
        )
        self.gffread_path = resolve_executable(
            getattr(self.args, 'gffread_path', None),
            'gffread',
            env_var='UPROTEINS_GFFREAD',
            cli_flag='--gffread_path',
        )
        self.copy_genome()
        self.index_genome()

    def copy_genome(self) -> None:
        src = pathlib.Path(self.args.genome).resolve()
        dst = pathlib.Path('./genome.fasta').resolve()
        if src != dst:
            shutil.copy2(src, dst)

    def index_genome(self):
        from src.utils import resolve_executable

        samtools_bin = resolve_executable(None, 'samtools', env_var='UPROTEINS_SAMTOOLS')
        cmd_sam = [str(samtools_bin), 'faidx', 'genome.fasta']
        result = subprocess.run(cmd_sam)
        if (code := result.returncode) != 0:
            err = f'samtools faidx finished with non-zero status code: {code}'
            cli.exit(3, err)

    def run_gffcompare(self):
        pathlib.Path('RNA_comparisons').mkdir(exist_ok=True)
        cmd_compare = [
            str(self.gffcompare_path),
            '-r', str(self.args.gtf),
            str(self.assembled_gtf),
            '-o', 'RNA_comparisons/compared_assembly',
            '-s', 'genome.fasta'
        ]

        result = subprocess.run(cmd_compare)
        if (code := result.returncode) != 0:
            err = f'gffcompare finished with non-zero return code: {code}'
            cli.exit(3, err)
        return 'RNA_comparisons'

    def extract_sequences(self):
        cmd_read = [
            str(self.gffread_path),
            '-w', 'HISAT/transcripts.fasta',
            '-g', 'genome.fasta',
            str(self.assembled_gtf)
        ]
        result = subprocess.run(cmd_read)
        if (code := result.returncode) != 0:
            err = f'gffread finished with non-zero return code: {code}'
            cli.exit(3, err)
        return 'HISAT/transcripts.fasta'
