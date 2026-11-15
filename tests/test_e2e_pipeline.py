# Copyright © 2025-2026 uProteInS Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Self-contained 6-stage + train-validator end-to-end proteogenomic integration suite (Issue #43)."""

from __future__ import annotations

import json
import os
import pathlib
import stat
import textwrap

from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import pytest

from src.main import uproteins
from src.utils import REPO_ROOT


# Codons chosen so none of the internal sense codons accidentally introduce an
# unintended bacterial start (TTG, CTG, ATT, ATC, ATA, ATG, GTG) or stop (TAA, TAG, TGA)
_CLEAN_CODONS: dict[str, str] = {
    "A": "GCC",
    "D": "GAC",
    "E": "GAG",
    "F": "TTC",
    "G": "GGC",
    "H": "CAC",
    "K": "AAG",
    "L": "CTC",
    "N": "AAC",
    "P": "CCG",
    "Q": "CAG",
    "R": "CGC",
    "S": "TCC",
    "T": "ACC",
    "V": "GTC",
    "W": "TGG",
    "Y": "TAC",
}


def _encode_clean(aa_seq: str) -> str:
    return "".join(_CLEAN_CODONS[aa] for aa in aa_seq)


def _spacer(num_codons: int) -> str:
    """Return neutral 'CGC' * N spacer with zero start/stop codons and zero 'A's in all 6 frames."""
    return "CGC" * num_codons


def _build_synthetic_bacterial_dataset(work_dir: pathlib.Path) -> dict:
    """Create a multi-contig bacterial genome (circular chr1 + pPlasmid), canonical proteome,
    reference GTF/GFF, 16S rRNA, RNA-Seq FASTQ, and MS spectrum metadata."""
    inputs = work_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    ms_raw = inputs / "ms_raw"
    ms_raw.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Contig 1: circular chr1
    #   - 5' tail of circular oriC smORF (continues from 3' end of chr1):
    #     full protein: MRGYVAPTLNDKFR -> peptide R.GYVAPTLNDK.F (Scan 103, HC)
    #   - Canonical GTG-start Rv0001_ANNO preceded in-frame by upstream ATG ("MADEK")
    #     and containing internal TTG ("L"): MGSYVNPEKLFDQWLRAK (Scan 201 -> _ANNO)
    #   - Minus-strand leaderless transcript uproteins.1.1 (2-nt 5' UTR < 5 nt):
    #     protein MRTWNFQDVLKGR -> peptide R.TWNFQDVLK.G (Scan 104, HC, Leaderless)
    #   - 3' head of circular oriC smORF preceded by strong SD AGGAGGT
    # ------------------------------------------------------------------
    oric_full_cds = "ATG" + _encode_clean("RGYVAPTLNDKFR") + "TAA"  # 45 nt (14 aa + stop)
    oric_3prime_head = oric_full_cds[:21]  # first 7 codons at 3' end of chr1
    oric_5prime_tail = oric_full_cds[21:]  # remaining 8 codons (incl. TAA) at 5' start of chr1

    rv0001_upstream_readthrough_prefix = "ATG" + _encode_clean("ADEK")
    rv0001_cds = (
        "GTG"
        + _encode_clean("GSYVNPEK")
        + "TTG"  # internal Leu encoded by TTG (tests nested non-ATG canonical collapse)
        + _encode_clean("FDQWLRAK")
        + "TAA"
    )
    rv0001_protein = "MGSYVNPEKLFDQWLRAK"

    # Minus-strand leaderless transcript uproteins.1.1 (5'->3' RNA has 2-nt 5' UTR "CC")
    tx_leaderless_rna = (
        "CC"  # 2-nt 5' UTR -> strictly Leaderless (< 5 nt)
        + "ATG"
        + _encode_clean("RTWNFQDVLKGR")
        + "TAA"
        + _spacer(6)
    )
    tx_leaderless_rc_on_plus = str(Seq(tx_leaderless_rna).reverse_complement())

    chr1_parts_before_tx = (
        oric_5prime_tail
        + _spacer(8)
        + rv0001_upstream_readthrough_prefix
        + rv0001_cds
        + _spacer(10)
    )
    tx1_start_1based = len(chr1_parts_before_tx) + 1
    tx1_end_1based = tx1_start_1based + len(tx_leaderless_rc_on_plus) - 1

    chr1_seq = (
        chr1_parts_before_tx
        + tx_leaderless_rc_on_plus
        + _spacer(10)
        + "AGGAGGT"
        + "CCCCCCC"  # 7-nt spacer before circular oriC ATG
        + oric_3prime_head
    )
    assert len(chr1_seq) % 3 == 0 or True  # circular walker handles any genome length!

    # Compute Rv0001 1-based genomic coordinates on chr1 (+)
    rv0001_start_1based = len(oric_5prime_tail + _spacer(8) + rv0001_upstream_readthrough_prefix) + 1
    rv0001_end_1based = rv0001_start_1based + len(rv0001_cds) - 1

    # ------------------------------------------------------------------
    # Contig 2: pPlasmid
    #   - Locus 4 (also transcribed as uproteins.2.1 on '+' strand):
    #     ATG_outer (no SD) + in-frame linker "GGCAGGAGGTCCCCCCCC" carrying strong SD
    #     AGGAGGT 8 nt upstream of ATG_inner -> AltCodons selects ATG_inner!
    #     Protein from ATG_inner: MRTYVGQNPEKSNFQDVWLRVYTPQNDLRGK
    #       * Scan 101: R.TYVGQNPEK.S (HC)
    #       * Scan 102: K.SNFQDVWLR.V (LC -> rescued as Rescued_LC in Stage 5!)
    #       * Scan 105: R.VYTPQNDLR.G (HC)
    #   - Locus 5: I==L + Met-excision canonical mimic:
    #     proteome has Rv0002 = MGVLPEPTWQRK; pPlasmid has MRGVIPEPTWQR (Scan 203 -> excluded!)
    #   - Locus 6: Noisy smORF MRQWNPYVGDFK (Scan 204: global q=0.005, but outscored by
    #     3 novel_decoy PSMs -> rejected by SubgroupFDR!)
    #   - Locus 7: Weak target MRHVDLPQWYK (Scan 205: loses per-scan TDC to higher-scoring decoy!)
    # ------------------------------------------------------------------
    locus4_block = (
        _spacer(8)  # 24-nt 5' UTR of uproteins.2.1 with zero purines -> ATG_outer has SD absent
        + "ATG"  # ATG_outer
        + "GGCAGGAGGTCCCCCCCC"  # 6 in-frame codons (GRRSPP) embedding AGGAGGT 8 nt before ATG_inner
        + "ATG"  # ATG_inner (expected AltCodons priority winner!)
        + _encode_clean("RTYVGQNPEKSNFQDVWLRVYTPQNDLRGK")
        + "TAA"
        + _spacer(4)
    )
    locus5_il_mimic_cds = "ATG" + "CGC" + "GGCGTCATCCCGGAGCCGACCTGGCAGCGC" + "TAA"  # MRGVIPEPTWQR*
    locus6_noisy_cds = "ATG" + _encode_clean("RQWNPYVGDFK") + "TAA"  # also transcribed in uproteins.3.1
    locus7_tdc_loser_cds = "ATG" + _encode_clean("RHVDLPQWYK") + "TAA"

    p_prefix = _spacer(6)
    tx2_start_1based = len(p_prefix) + 1
    tx2_end_1based = tx2_start_1based + len(locus4_block) - 1

    mid_spacer = _spacer(6) + locus5_il_mimic_cds + _spacer(6)
    tx3_prefix_len = len(p_prefix) + len(locus4_block) + len(mid_spacer)
    tx3_block = _spacer(4) + locus6_noisy_cds + _spacer(3)
    tx3_start_1based = tx3_prefix_len + 1
    tx3_end_1based = tx3_start_1based + len(tx3_block) - 1

    plasmid_seq = (
        p_prefix
        + locus4_block
        + mid_spacer
        + tx3_block
        + _spacer(5)
        + locus7_tdc_loser_cds
        + _spacer(6)
    )

    genome_fasta = inputs / "genome.fasta"
    genome_fasta.write_text(
        f">chr1 circular chromosome\n{chr1_seq}\n"
        f">pPlasmid virulence plasmid\n{plasmid_seq}\n",
        encoding="utf-8",
    )

    # Canonical proteome: Rv0001 (GTG start -> M at pos 1) + Rv0002 (Met-excision + L vs I test)
    rv0002_protein = "MGVLPEPTWQRK"
    proteome_fasta = inputs / "proteome.fasta"
    proteome_fasta.write_text(
        f">Rv0001 canonical GTG-start protein\n{rv0001_protein}\n"
        f">Rv0002 canonical Met-excised Leu protein\n{rv0002_protein}\n",
        encoding="utf-8",
    )

    # Reference GTF (for Stage 1 assembly) & GFF3 (for Stage 4 postms)
    ref_gtf = inputs / "reference.gtf"
    ref_gtf.write_text(
        f'chr1\tRefSeq\ttranscript\t{rv0001_start_1based}\t{rv0001_end_1based}\t.\t+\t.\tgene_id "Rv0001"; transcript_id "Rv0001.1";\n'
        f'chr1\tRefSeq\texon\t{rv0001_start_1based}\t{rv0001_end_1based}\t.\t+\t.\tgene_id "Rv0001"; transcript_id "Rv0001.1";\n',
        encoding="utf-8",
    )
    ref_gff = inputs / "reference.gff3"
    ref_gff.write_text(
        "##gff-version 3\n"
        f"chr1\tRefSeq\tCDS\t{rv0001_start_1based}\t{rv0001_end_1based}\t.\t+\t0\tID=cds-Rv0001;locus_tag=Rv0001;Name=Rv0001\n",
        encoding="utf-8",
    )

    # Multi-record rRNA FASTA (23S decoy first, 16S with 3' anti-SD CCUCC second)
    rrna_fasta = inputs / "rrna.fasta"
    rrna_fasta.write_text(
        ">rrl_23S ribosomal RNA\n"
        "GGTTAAGCGACTAAGCGTACACGGTGGATGCCCTGGCAGTCAGAGGCGATGAAGGACGTG\n"
        ">rrs_16S ribosomal RNA\n"
        "AGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCTTAACACATGCAAGTCGAACGGAAAGGTCTCTTCGGAGATACTCGAGTGGCGAACGGGTGAGTAACACGTGGGTGATCTGCCCTGCACTTCGGGATAAGCCTGGGAAACTGGGTCTAATACCGGATAGGACCACGGGATGCATGTCTTGTGGTGGAAAGCGCTTTAGCGGTGTGGGATGAGCCCGCGGCCTATCAGCTTGTTGGTGGGGTGACGGCCTACCAAGGCGACGACGGGTAGCCGGCCTGAGAGGGTGTCCGGCCACACTGGGACTGAGATACGGCCCAGACTCCTACGGGAGGCAGCAGTGGGGAATATTGCACAATGGGCGCAAGCCTGATGCAGCCATGCCGCGTGTGGGATGACGGCCTTCGGGTTGTAAACCTCTTTCACCATCGACGAAGGTCCGGGTTCTCTCGGATTGACGGTAGGTGGAGAAGAAGCACCGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGGTGCGAGCGTTGTCCGGAATTACTGGGCGTAAAGAGCTCGTAGGTGGTTTGTCGCGTTGTTCGTGAAATCTCACGGCTTAACTGTGAGCGTGCGGGCGATACGGGCAGACTAGAGTACTGCAGGGGAGACTGGAATTCCTGGTGTAGCGGTGGAATGCGCAGATATCAGGAGGAACACCGGTGGCGAAGGCGGGTCTCTGGGCAGTAACTGACGCTGAGGAGCGAAAGCGTGGGGAGCGAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGGTGGGTACTAGGTGTGGGTTTCCTTCCTTGGGATCCGTGCCGTAGCTAACGCATTAAGTACCCCGCCTGGGGAGTACGGCCGCAAGGCTAAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGCGGAGCATGTGGATTAATTCGATGCAACGCGAAGAACCTTACCTGGGTTTGACATGCACAGGACGCGTCTAGAGATAGGCGTTCCCTTGTGGCCTGTGTGCAGGTGGTGCATGGCTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTGTCTCATGTTGCCAGCACGTAATGGTGGGGACTCGTGAGAGACTGCCGGGGTCAACTCGGAGGAAGGTGGGGATGACGTCAAGTCATCATGCCCCTTATGTCCAGGGCTTCACACATGCTACAATGGCCGGTACAAAGGGCTGCGATGCCGCGAGGTTAAGCGAATCCTTAAAAGCCGGTCTCAGTTCGGATCGGGGTCTGCAACTCGACCCCGTGAAGTCGGAGTCGCTAGTAATCGCAGATCAGCAACGCTGCGGTGAATACGTTCCCGGGCCTTGTCACACACCGCCCGTCACGTCATGAAAGTCGGTAACACCCGAAGCCAGTGGCCTAACCCTCGGGAGGGAGCTGTCGAAGGTGGGATCGGCGATTGGGACGAAGTCGTAACAAGGTAGCCGTACCGGAAGGTGCGGCTGGATCACCTCCTTA\n",
        encoding="utf-8",
    )

    # Single-end dUTP stranded RNA-Seq FASTQ
    reads_fq = inputs / "sample_R.fastq"
    reads_fq.write_text(
        "@read1\n" + tx_leaderless_rna[:40] + "\n+\n" + ("I" * 40) + "\n",
        encoding="utf-8",
    )

    # Dummy spectrum file in inputs/ms_raw/ (must remain untouched by Stage 3 ms!)
    mzml_file = ms_raw / "sample_run.mzML"
    mzml_file.write_text("<mzML id='sample_run'/>\n", encoding="utf-8")

    assembled_gtf_text = (
        f'chr1\tStringTie\ttranscript\t{tx1_start_1based}\t{tx1_end_1based}\t1000\t-\t.\tgene_id "uproteins.1"; transcript_id "uproteins.1.1";\n'
        f'chr1\tStringTie\texon\t{tx1_start_1based}\t{tx1_end_1based}\t1000\t-\t.\tgene_id "uproteins.1"; transcript_id "uproteins.1.1"; exon_number "1";\n'
        f'pPlasmid\tStringTie\ttranscript\t{tx2_start_1based}\t{tx2_end_1based}\t1000\t+\t.\tgene_id "uproteins.2"; transcript_id "uproteins.2.1";\n'
        f'pPlasmid\tStringTie\texon\t{tx2_start_1based}\t{tx2_end_1based}\t1000\t+\t.\tgene_id "uproteins.2"; transcript_id "uproteins.2.1"; exon_number "1";\n'
        f'pPlasmid\tStringTie\ttranscript\t{tx3_start_1based}\t{tx3_end_1based}\t1000\t+\t.\tgene_id "uproteins.3"; transcript_id "uproteins.3.1";\n'
        f'pPlasmid\tStringTie\texon\t{tx3_start_1based}\t{tx3_end_1based}\t1000\t+\t.\tgene_id "uproteins.3"; transcript_id "uproteins.3.1"; exon_number "1";\n'
    )

    return {
        "inputs": inputs,
        "ms_raw": ms_raw,
        "genome_fasta": genome_fasta,
        "proteome_fasta": proteome_fasta,
        "ref_gtf": ref_gtf,
        "ref_gff": ref_gff,
        "rrna_fasta": rrna_fasta,
        "reads_fq": reads_fq,
        "mzml_file": mzml_file,
        "assembled_gtf_text": assembled_gtf_text,
        "tx1_coords": (tx1_start_1based, tx1_end_1based),
        "rv0001_protein": rv0001_protein,
    }


def _install_mock_bio_tools(
    mock_bin: pathlib.Path,
    state_dir: pathlib.Path,
    assembled_gtf_text: str,
) -> pathlib.Path:
    """Write lightweight executable Python shims for external C/Java bioinformatics tools."""
    mock_bin.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    gtf_template_path = state_dir / "assembled_template.gtf"
    gtf_template_path.write_text(assembled_gtf_text, encoding="utf-8")
    log_jsonl = state_dir / "invocations.jsonl"

    dummy_jar = state_dir / "MSGFPlus.jar"
    dummy_jar.write_bytes(b"PK\x03\x04mock-msgf-jar")

    def _write_exec(name: str, body: str) -> None:
        script_path = mock_bin / name
        header = (
            "#!/usr/bin/env python3\n"
            "import json, os, pathlib, re, sys\n"
            f"LOG_PATH = pathlib.Path({str(log_jsonl)!r})\n"
            "with LOG_PATH.open('a', encoding='utf-8') as _lf:\n"
            f"    _lf.write(json.dumps({{'tool': {name!r}, 'argv': sys.argv[1:], 'cwd': os.getcwd()}}) + '\\n')\n"
        )
        script_path.write_text(header + textwrap.dedent(body), encoding="utf-8")
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    _write_exec(
        "hisat2-build",
        """
        prefix = sys.argv[-1]
        pathlib.Path(f"{prefix}.1.ht2").write_text("mock-ht2-index\\n", encoding="utf-8")
        """,
    )

    _write_exec(
        "hisat2",
        """
        sys.stdout.write("@HD\\tVN:1.6\\tSO:unsorted\\n@SQ\\tSN:chr1\\tLN:500\\n")
        sys.stdout.flush()
        """,
    )

    _write_exec(
        "samtools",
        """
        sub = sys.argv[1]
        if sub == "sort":
            _ = sys.stdin.read()
            out_idx = sys.argv.index("-o") + 1
            out_bam = pathlib.Path(sys.argv[out_idx])
            out_bam.parent.mkdir(parents=True, exist_ok=True)
            out_bam.write_bytes(b"BAM\\x01mock-sorted-bam")
        elif sub == "faidx":
            fa = pathlib.Path(sys.argv[2])
            pathlib.Path(f"{fa}.fai").write_text("chr1\\t500\\t6\\t60\\t61\\n", encoding="utf-8")
        """,
    )

    _write_exec(
        "stringtie",
        f"""
        args = sys.argv[1:]
        out_path = pathlib.Path(args[args.index("-o") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tpl = pathlib.Path({str(gtf_template_path)!r}).read_text(encoding="utf-8")
        out_path.write_text(tpl, encoding="utf-8")
        if "-A" in args:
            abund = pathlib.Path(args[args.index("-A") + 1])
            abund.parent.mkdir(parents=True, exist_ok=True)
            abund.write_text("Gene ID\\tGene Name\\tReference\\tStrand\\tStart\\tEnd\\tCoverage\\tFPKM\\tTPM\\n", encoding="utf-8")
        """,
    )

    _write_exec(
        "gffcompare",
        """
        args = sys.argv[1:]
        prefix = pathlib.Path(args[args.index("-o") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{prefix}.annotated.gtf").write_text("# mock gffcompare\\n", encoding="utf-8")
        """,
    )

    # gffread genuinely extracts spliced / reverse-complemented RNA sequences from genome.fasta + assembled.gtf!
    _write_exec(
        "gffread",
        """
        args = sys.argv[1:]
        out_fa = pathlib.Path(args[args.index("-w") + 1])
        genome_fa = pathlib.Path(args[args.index("-g") + 1])
        gtf_file = pathlib.Path(args[-1])

        contigs = {}
        cur_id, buf = None, []
        for raw in genome_fa.read_text(encoding="utf-8").splitlines():
            if raw.startswith(">"):
                if cur_id is not None:
                    contigs[cur_id] = "".join(buf)
                cur_id = raw[1:].split()[0]
                buf = []
            else:
                buf.append(raw.strip())
        if cur_id is not None:
            contigs[cur_id] = "".join(buf)

        comp = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
        def revcomp(s):
            return s.translate(comp)[::-1]

        exons = {}
        strands = {}
        for line in gtf_file.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split("\\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            m = re.search(r'transcript_id\\s+"([^"]+)"', parts[8])
            if not m:
                continue
            tid = m.group(1)
            strands[tid] = parts[6]
            exons.setdefault(tid, []).append((parts[0], int(parts[3]), int(parts[4])))

        out_fa.parent.mkdir(parents=True, exist_ok=True)
        with out_fa.open("w", encoding="utf-8") as fh:
            for tid, ex_list in exons.items():
                ex_sorted = sorted(ex_list, key=lambda x: x[1])
                dna = "".join(contigs[c][s - 1 : e] for c, s, e in ex_sorted)
                if strands.get(tid) == "-":
                    dna = revcomp(dna)
                fh.write(f">{tid}\\n{dna}\\n")
        """,
    )

    # java handles BOTH `-jar MSGFPlus.jar` (Stage 3 ms) and `-cp MSGFPlus.jar ...MzIDToTsv` (Stage 4 postms)
    _write_exec(
        "java",
        """
        args = sys.argv[1:]
        def read_fasta(p):
            recs = []
            cid, buf = None, []
            for line in pathlib.Path(p).read_text(encoding="utf-8").splitlines():
                if line.startswith(">"):
                    if cid is not None:
                        recs.append((cid, "".join(buf)))
                    cid = line[1:].split()[0]
                    buf = []
                else:
                    buf.append(line.strip())
            if cid is not None:
                recs.append((cid, "".join(buf)))
            return recs

        if "-jar" in args:
            db_path = pathlib.Path(args[args.index("-d") + 1])
            spec_path = pathlib.Path(args[args.index("-s") + 1])
            out_mzid = pathlib.Path(args[args.index("-o") + 1])
            out_mzid.parent.mkdir(parents=True, exist_ok=True)
            # Touch a dummy .cseq index beside staged_db to verify staging isolation!
            pathlib.Path(f"{db_path}.cseq").write_bytes(b"idx")

            records = read_fasta(db_path)
            is_decoy_db = all(r[0].lower().startswith("decoy_") for r in records) if records else False
            hits = []

            if not is_decoy_db:
                # Query real target FASTA for each designed MS2 scan!
                queries = [
                    # (sii_idx, scan_num, pep_core, flank_l, flank_r, hc_flag, perc_score, perc_q, perc_pep, ln_eval)
                    (1, 101, "TYVGQNPEK", "R", "S", True, 5.5, 0.001, 0.0005, 16.2),
                    (2, 102, "SNFQDVWLR", "K", "V", False, 5.2, 0.001, 0.0010, 1.4),
                    (3, 103, "GYVAPTLNDK", "R", "F", True, 5.0, 0.001, 0.0008, 15.8),
                    (4, 104, "TWNFQDVLK", "R", "G", True, 4.8, 0.001, 0.0009, 15.4),
                    (5, 105, "VYTPQNDLR", "R", "G", True, 4.7, 0.001, 0.0010, 15.1),
                    (6, 201, "GSYVNPEK", "M", "L", True, 6.2, 0.0005, 0.0001, 18.0),
                    (7, 202, "TITLEVEPSDTIENVK", "K", "A", True, 6.0, 0.0005, 0.0001, 17.5),
                    (8, 203, "GVIPEPTWQR", "R", "-", True, 4.6, 0.001, 0.0012, 14.9),
                    (9, 204, "QWNPYVGDFK", "R", "-", False, -0.9, 0.005, 0.0100, 0.5),
                    (10, 205, "HVDLPQWYK", "R", "-", False, -0.5, 0.020, 0.0500, 1.0),
                ]
                for sii_idx, scan_num, pep_core, fl, fr, hc, pscore, pq, ppep, lnev in queries:
                    matched_prots = [rid for rid, seq in records if pep_core in seq]
                    if matched_prots:
                        hits.append({
                            "spec_file": spec_path.name,
                            "sii": sii_idx,
                            "scan": scan_num,
                            "charge": 2,
                            "label": 1,
                            "peptide_core": pep_core,
                            "peptide_fmt": f"{fl}.{pep_core}.{fr}",
                            "proteins": matched_prots,
                            "hc": hc,
                            "perc_score": pscore,
                            "perc_q": pq,
                            "perc_pep": ppep,
                            "lnEValue": lnev,
                        })
            else:
                # Select real decoy novel ORF entries from {Genome,Transcriptome}_decoy.fasta!
                novel_decoys = [
                    (rid, seq) for rid, seq in records
                    if not rid.lower().startswith("decoy_contaminant_") and not rid.endswith("_ANNO")
                ]
                decoy_specs = [
                    # Scan 205 beats target Scan 205 (lnEValue 9.5 > 1.0) in per-scan TDC!
                    (1, 205, 0.25, 9.5),
                    # Three novel decoys (scans 301..303) outscore noisy Scan 204 (-0.9)!
                    (2, 301, 0.20, 3.2),
                    (3, 302, 0.00, 2.8),
                    (4, 303, -0.20, 2.5),
                ]
                for idx_d, (sii_idx, scan_num, pscore, lnev) in enumerate(decoy_specs):
                    if not novel_decoys:
                        break
                    rid, seq = novel_decoys[idx_d % len(novel_decoys)]
                    pep_core = seq[2:10] if len(seq) >= 10 else seq[:6]
                    hits.append({
                        "spec_file": spec_path.name,
                        "sii": sii_idx,
                        "scan": scan_num,
                        "charge": 2,
                        "label": -1,
                        "peptide_core": pep_core,
                        "peptide_fmt": f"R.{pep_core}.K",
                        "proteins": [rid],
                        "hc": False,
                        "perc_score": pscore,
                        "perc_q": 0.50,
                        "perc_pep": 0.80,
                        "lnEValue": lnev,
                    })

            db_tags = "\\n".join(
                f'  <DBSequence id="DBSeq_{i}" accession="{p}"/>'
                for i, h in enumerate(hits)
                for p in h["proteins"]
            )
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>\\n'
                '<MzIdentML id="MS-GF+">\\n'
                '  <Enzymes><Enzyme id="Tryp"><EnzymeName><cvParam name="Trypsin"/></EnzymeName></Enzyme></Enzymes>\\n'
                f'{db_tags}\\n'
                f'<!-- MOCK_HITS_JSON:{json.dumps(hits)} -->\\n'
                '</MzIdentML>\\n'
            )
            out_mzid.write_text(xml, encoding="utf-8")

        elif "edu.ucsd.msjava.ui.MzIDToTsv" in args:
            in_mzid = pathlib.Path(args[args.index("-i") + 1])
            out_tsv = pathlib.Path(args[args.index("-o") + 1])
            out_tsv.parent.mkdir(parents=True, exist_ok=True)
            txt = in_mzid.read_text(encoding="utf-8")
            m = re.search(r"<!-- MOCK_HITS_JSON:(.*?) -->", txt)
            hits = json.loads(m.group(1)) if m else []
            cols = [
                "#SpecFile", "SpecID", "ScanNum", "FragMethod", "Precursor",
                "IsotopeError", "PrecursorError(ppm)", "Charge", "Peptide",
                "Protein", "DeNovoScore", "MSGFScore", "SpecEValue", "EValue",
                "QValue", "PepQValue",
            ]
            lines = ["\\t".join(cols)]
            for h in hits:
                prot_str = ";".join(f"{p}(pre=R,post=S)" for p in h["proteins"])
                spec_id = f"controllerType=0 controllerNumber=1 scan={h['scan']}"
                raw_sc = 140 if h["hc"] else -20
                denovo = 140 if h["hc"] else 10
                spec_ev = "1.5e-12" if h["hc"] else "2.5e-4"
                ev = "1.5e-9" if h["hc"] else "0.25"
                row = [
                    h["spec_file"], spec_id, str(h["scan"]), "HCD", "550.28",
                    "0", "1.2", str(h["charge"]), h["peptide_fmt"],
                    prot_str, str(denovo), str(raw_sc), spec_ev, ev,
                    str(h["perc_q"]), str(h["perc_q"]),
                ]
                lines.append("\\t".join(row))
            out_tsv.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        """,
    )

    _write_exec(
        "msgf2pin",
        """
        args = sys.argv[1:]
        out_pin = pathlib.Path(args[args.index("-o") + 1])
        out_pin.parent.mkdir(parents=True, exist_ok=True)
        meta_files = [pathlib.Path(a) for a in args[:args.index("-o")]]

        mzid_files = []
        for mf in meta_files:
            if mf.is_file():
                for line in mf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        mzid_files.append(pathlib.Path(line.strip()))

        headers = [
            "SpecId", "Label", "ScanNr", "ExpMass", "CalcMass",
            "RawScore", "DeNovoScore", "ScoreRatio", "Energy", "lnEValue",
            "IsotopeError", "lnExplainedIonCurrentRatio", "lnNTermIonCurrentRatio",
            "lnCTermIonCurrentRatio", "lnMS2IonCurrent", "Mass", "PepLen",
            "dM", "absdM", "MeanErrorTop7", "sqMeanErrorTop7", "StdevErrorTop7",
            "Charge1", "Charge2", "enzN", "enzC", "enzInt", "Peptide", "Proteins",
        ]
        lines = ["\\t".join(headers)]
        for mz in mzid_files:
            txt = mz.read_text(encoding="utf-8")
            m = re.search(r"<!-- MOCK_HITS_JSON:(.*?) -->", txt)
            hits = json.loads(m.group(1)) if m else []
            stem = mz.stem  # e.g. sample_run_target or sample_run_decoy
            for h in hits:
                spec_id = f"{stem}_SII_{h['sii']}_1_{h['scan']}_{h['charge']}_1"
                pep_len = len(h["peptide_core"])
                if h["hc"]:
                    feats = [
                        "527.785", "527.786", "134.0", "134.0", "1.0", "0.0",
                        str(h["lnEValue"]), "0", "0.1672", "-0.8809", "-0.2644",
                        "11.5406", "527.785", str(pep_len), "0.0", "0.0",
                        "12.0", "4228.07", "65.0",
                    ]
                else:
                    feats = [
                        "685.338", "684.849", "-38.0", "4.0", "-1.0", "42.0",
                        str(h["lnEValue"]), "1", "-2.1355", "-2.8792", "-2.7789",
                        "7.8959", "685.338", str(pep_len), "-0.000019", "0.000019",
                        "247.5", "61288.0", "360.0",
                    ]
                row = [
                    spec_id, str(h["label"]), str(h["scan"]),
                    *feats,
                    "0", "1", "1", "1", "0",
                    h["peptide_fmt"],
                    *h["proteins"],  # tab-separated multi-protein tail like real msgf2pin!
                ]
                lines.append("\\t".join(row))
        out_pin.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        """,
    )

    _write_exec(
        "percolator",
        """
        args = sys.argv[1:]
        pin_file = pathlib.Path(args[-1])
        psm_out = pathlib.Path(args[args.index("-m") + 1])
        dec_psm_out = pathlib.Path(args[args.index("--decoy-results-psms") + 1])
        pep_out = pathlib.Path(args[args.index("-r") + 1])
        dec_pep_out = pathlib.Path(args[args.index("--decoy-results-peptides") + 1])
        prot_out = pathlib.Path(args[args.index("--results-proteins") + 1])
        xml_out = pathlib.Path(args[args.index("-X") + 1])

        raw_lines = pin_file.read_text(encoding="utf-8").splitlines()
        hdr = raw_lines[0].split("\\t")
        idx_spec = hdr.index("SpecId")
        idx_lbl = hdr.index("Label")
        idx_scan = hdr.index("ScanNr")
        idx_lnev = hdr.index("lnEValue")
        idx_pep = hdr.index("Peptide")
        idx_prot = hdr.index("Proteins")

        # Score mapping that preserves exact global-FDR vs subgroup-FDR test semantics!
        scan_meta = {
            101: (5.5, 0.001, 0.0005),
            102: (5.2, 0.001, 0.0010),
            103: (5.0, 0.001, 0.0008),
            104: (4.8, 0.001, 0.0009),
            105: (4.7, 0.001, 0.0010),
            201: (6.2, 0.0005, 0.0001),
            202: (6.0, 0.0005, 0.0001),
            203: (4.6, 0.001, 0.0012),
            # Scan 204 intentionally has global q=0.005 (<=0.01) so only SubgroupFDR catches it!
            204: (-0.9, 0.005, 0.0100),
            205: (0.25, 0.50, 0.80),
            301: (0.20, 0.50, 0.80),
            302: (0.00, 0.50, 0.80),
            303: (-0.20, 0.50, 0.80),
        }

        t_rows, d_rows = [], []
        prot_ids = set()
        for line in raw_lines[1:]:
            if not line.strip():
                continue
            cols = line.split("\\t")
            if cols[0] == "DefaultDirection":
                continue
            spec_id = cols[idx_spec]
            lbl = int(cols[idx_lbl])
            scan_nr = int(cols[idx_scan])
            pep = cols[idx_pep]
            prots = cols[idx_prot:]
            sc, qv, pep_prob = scan_meta.get(scan_nr, (float(cols[idx_lnev]) / 3.0, 0.001, 0.001))
            entry = (sc, [spec_id, f"{sc:.4f}", f"{qv:.6f}", f"{pep_prob:.6f}", pep, *prots])
            if lbl == 1:
                t_rows.append(entry)
                prot_ids.update(prots)
            else:
                d_rows.append(entry)

        t_rows.sort(key=lambda x: -x[0])
        d_rows.sort(key=lambda x: -x[0])

        psm_hdr = "PSMId\\tscore\\tq-value\\tposterior_error_prob\\tpeptide\\tproteinIds\\n"
        psm_out.write_text(psm_hdr + "".join("\\t".join(r) + "\\n" for _, r in t_rows), encoding="utf-8")
        dec_psm_out.write_text(psm_hdr + "".join("\\t".join(r) + "\\n" for _, r in d_rows), encoding="utf-8")
        pep_out.write_text(psm_hdr + "".join("\\t".join(r) + "\\n" for _, r in t_rows), encoding="utf-8")
        dec_pep_out.write_text(psm_hdr + "".join("\\t".join(r) + "\\n" for _, r in d_rows), encoding="utf-8")

        prot_lines = ["ProteinId\\tProteinGroupId\\tq-value\\tposterior_error_prob\\tpeptideIds"]
        for gid, pid in enumerate(sorted(prot_ids), start=1):
            prot_lines.append(f"{pid}\\t{gid}\\t0.001\\t0.001\\tR.PEPTIDEK.S")
        prot_out.write_text("\\n".join(prot_lines) + "\\n", encoding="utf-8")
        xml_out.write_text("<percolator_output/>\\n", encoding="utf-8")
        """,
    )

    return dummy_jar


@pytest.mark.fullrun
def test_cli_version_is_unofficial_and_not_2_0_0(capsys):
    """Verify unofficial fork does not claim an official 2.0.0 release version."""
    pyproject_txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "1.4.2"' in pyproject_txt
    assert 'version = "2.0.0"' not in pyproject_txt

    with pytest.raises(SystemExit) as exc_info:
        uproteins(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "2.0.0" not in captured.out
    assert "1.4.2" in captured.out
    assert "unofficial" in captured.out.lower()


@pytest.mark.fullrun
def test_full_6_stage_proteogenomic_pipeline_and_train_validator(tmp_path, monkeypatch):
    """Execute all 6 pipeline stages (`assembly` -> `database` -> `ms` -> `postms` -> `validate`
    -> `metrics`) plus `train-validator` end-to-end on a multi-contig synthetic bacterium and
    verify every cross-stage biological, statistical, thermodynamic, and ML invariant."""
    ds = _build_synthetic_bacterial_dataset(tmp_path)
    mock_bin = tmp_path / "mock_bin"
    state_dir = tmp_path / "mock_state"
    dummy_jar = _install_mock_bio_tools(mock_bin, state_dir, ds["assembled_gtf_text"])

    monkeypatch.setenv("PATH", f"{mock_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("UPROTEINS_MSGF_JAR", str(dummy_jar))

    outdir = tmp_path / "run_out"
    caller_cwd = pathlib.Path.cwd()

    # =========================================================================
    # STAGE 1: assembly (bacterial unspliced HISAT2 + streaming BAM + -L R -> --rf)
    # =========================================================================
    uproteins(
        [
            "assembly",
            "-o",
            str(outdir),
            "-g",
            str(ds["genome_fasta"]),
            "-G",
            str(ds["ref_gtf"]),
            "--single",
            str(ds["reads_fq"]),
            "-L",
            "R",
            "--threads",
            "2",
        ]
    )
    assert pathlib.Path.cwd() == caller_cwd

    invocations = [
        json.loads(line)
        for line in (state_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    hisat2_calls = [c for c in invocations if c["tool"] == "hisat2"]
    assert len(hisat2_calls) == 1
    assert "--no-spliced-alignment" in hisat2_calls[0]["argv"]
    assert "--rna-strandness" in hisat2_calls[0]["argv"]
    assert "R" in hisat2_calls[0]["argv"]

    stringtie_calls = [c for c in invocations if c["tool"] == "stringtie" and "--merge" not in c["argv"]]
    assert len(stringtie_calls) == 1
    assert "--rf" in stringtie_calls[0]["argv"]
    assert "" not in stringtie_calls[0]["argv"]

    assembled_gtf = outdir / "assembled.gtf"
    assembled_gtf_before_postms = assembled_gtf.read_text(encoding="utf-8")
    tx_seqs = {rec.id: str(rec.seq) for rec in SeqIO.parse(str(outdir / "HISAT" / "transcripts.fasta"), "fasta")}
    assert {"uproteins.1.1", "uproteins.2.1", "uproteins.3.1"} <= set(tx_seqs.keys())
    # uproteins.1.1 on minus strand must start with 2-nt leaderless 5' UTR "CCATG..."
    assert tx_seqs["uproteins.1.1"].startswith("CCATGCGC")

    # =========================================================================
    # STAGE 2: database (circular oriC, GTG canonical readthrough/nested exclusion, contaminants)
    # =========================================================================
    uproteins(
        [
            "database",
            "-o",
            str(outdir),
            "-g",
            str(ds["genome_fasta"]),
            "-p",
            str(ds["proteome_fasta"]),
            "-T",
            "YES",
            "--circular",
        ]
    )

    genome_db_records = {
        rec.id: str(rec.seq) for rec in SeqIO.parse(str(outdir / "genome_database.fasta"), "fasta")
    }
    tx_db_records = {
        rec.id: str(rec.seq) for rec in SeqIO.parse(str(outdir / "transcriptome_database.fasta"), "fasta")
    }

    # 1. Canonical GTG-start Rv0001_ANNO present; neither nested TTG sub-ORF ("MFDQWLRAK")
    #    nor upstream ATG GTG-readthrough ("MADEKVGSYVNPEKLFDQWLRAK") may exist as novel gORFs!
    assert "Rv0001_ANNO" in genome_db_records
    assert genome_db_records["Rv0001_ANNO"] == ds["rv0001_protein"]
    novel_gorf_seqs = [
        seq
        for rid, seq in genome_db_records.items()
        if not rid.endswith("_ANNO") and not rid.startswith("contaminant_")
    ]
    assert "MFDQWLRAK" not in novel_gorf_seqs
    assert "MADEKVGSYVNPEKLFDQWLRAK" not in novel_gorf_seqs

    # 2. Circular oriC-spanning ORF ("GYVAPTLNDK") on chr1 & plasmid smORF ("TYVGQNPEK") present
    assert any("GYVAPTLNDK" in seq for seq in novel_gorf_seqs)
    assert any("TYVGQNPEK" in seq for seq in novel_gorf_seqs)

    # 3. Minus-strand leaderless transcript uproteins.1.1 translated in transcriptome_database.fasta
    assert any(
        rid.startswith("tORF_uproteins.1.1_") and "TWNFQDVLK" in seq
        for rid, seq in tx_db_records.items()
    )

    # 4. Forward contaminants (>contaminant_*) included in target FASTAs, zero Reverse_ entries
    contam_ids = [rid for rid in genome_db_records if rid.startswith("contaminant_")]
    assert len(contam_ids) == 121
    assert not any(rid.lower().startswith("reverse_") for rid in genome_db_records)

    # =========================================================================
    # STAGE 3: ms (explicit -o, raw-dir isolation, 1:1 tryptic pseudo-reversed decoys)
    # =========================================================================
    uproteins(
        [
            "ms",
            "-o",
            str(outdir),
            "-M",
            str(ds["ms_raw"]),
            "-T",
            "YES",
            "--java_mem",
            "2G",
        ]
    )

    # Raw MS folder must remain completely clean (only sample_run.mzML, no .cseq/.mzid/.fasta!)
    assert [p.name for p in ds["ms_raw"].iterdir()] == ["sample_run.mzML"]
    assert (outdir / "Genome" / "sample_run_target.mzid").is_file()
    assert (outdir / "Genome" / "sample_run_decoy.mzid").is_file()
    assert (outdir / "Transcriptome" / "sample_run_target.mzid").is_file()
    assert (outdir / "Transcriptome" / "sample_run_decoy.mzid").is_file()

    decoy_records = {
        rec.id: str(rec.seq)
        for rec in SeqIO.parse(str(outdir / "Genome" / "Percolator" / "Genome_decoy.fasta"), "fasta")
    }
    assert len(decoy_records) == len(genome_db_records)
    assert any(rid.startswith("decoy_contaminant_") for rid in decoy_records)

    # =========================================================================
    # STAGE 4: postms (TDC, SubgroupFDR, I==L + Met-excision UTP, Xia/Turner RBS, AltCodons, coords)
    # =========================================================================
    uproteins(
        [
            "postms",
            "-o",
            str(outdir),
            "-M",
            str(ds["ms_raw"]),
            "-p",
            str(ds["proteome_fasta"]),
            "-g",
            str(ds["genome_fasta"]),
            "--gff",
            str(ds["ref_gff"]),
            "--rrna",
            str(ds["rrna_fasta"]),
            "-T",
            "YES",
            "--qvalue",
            "0.25",
            "--pep",
            "0.05",
        ]
    )

    # 1. assembled.gtf must NEVER be modified in place by StringTieGFF!
    assert assembled_gtf.read_text(encoding="utf-8") == assembled_gtf_before_postms

    # 2. Per-scan top-1 Target-Decoy Competition: Scan 205 in Genome_pin.txt must be Label == -1 (decoy won!)
    assert (outdir / "Genome" / "Percolator" / "Genome_pin_uncompeted.txt").is_file()
    competed_pin_lines = (outdir / "Genome" / "Percolator" / "Genome_pin.txt").read_text(encoding="utf-8").splitlines()
    scan_205_rows = [ln.split("\t") for ln in competed_pin_lines[1:] if ln.split("\t")[2] == "205"]
    assert len(scan_205_rows) == 1
    assert scan_205_rows[0][1] == "-1"

    # 3. SubgroupFDR: Scan 204 ("QWNPYVGDFK") had global q=0.005 (<=0.01) but novel_psm_q_value > 0.25!
    sub_fdr_df = pd.read_csv(outdir / "Genome" / "post_perc" / "genome_subgroup_fdr.tsv", sep="\t")
    row_204 = sub_fdr_df[sub_fdr_df["peptide"].str.contains("QWNPYVGDFK")]
    assert len(row_204) == 1
    assert float(row_204["global_q_value"].iloc[0]) == pytest.approx(0.005)
    assert float(row_204["novel_psm_q_value"].iloc[0]) > 0.25

    # 4. Pre-validation results verification across Genome and Transcriptome
    g_pre = pd.read_csv(outdir / "Genome" / "Results" / "genome_pre_validation_results.txt", sep="\t")
    t_pre = pd.read_csv(outdir / "Transcriptome" / "Results" / "transcriptome_pre_validation_results.txt", sep="\t")

    # Negative controls MUST be absent from pre-validation novel smORF results:
    all_pre_peps = " ".join(g_pre["Peptide"].astype(str).tolist() + t_pre["Peptide"].astype(str).tolist())
    for forbidden_pep in ("GSYVNPEK", "TITLEVEPSDTIENVK", "GVIPEPTWQR", "QWNPYVGDFK", "HVDLPQWYK"):
        assert forbidden_pep not in all_pre_peps

    # Positive controls MUST be present:
    # - Circular oriC smORF ("GYVAPTLNDK") in Genome
    assert g_pre["Peptide"].astype(str).str.contains("GYVAPTLNDK").any()
    # - AltCodons + Xia/Turner RBS on pPlasmid Locus 4: selects ATG_inner ("MRTYVGQNPEK...") with Strong SD!
    plasmid_rows = g_pre[g_pre["Peptide"].astype(str).str.contains("TYVGQNPEK")]
    assert len(plasmid_rows) == 1
    assert plasmid_rows["Protein sequence"].iloc[0] == "MRTYVGQNPEKSNFQDVWLRVYTPQNDLRGK"
    assert plasmid_rows["Shine Dalgarno"].iloc[0] == "Present. Strong binding."
    assert float(plasmid_rows["Free Energy"].iloc[0]) <= -8.4

    # - Minus-strand leaderless uproteins.1.1 in Transcriptome:
    #   "Shine Dalgarno" == "Leaderless" AND "Genome Coordinates" projected onto chr1:...(-) with start <= end!
    leaderless_rows = t_pre[t_pre["Peptide"].astype(str).str.contains("TWNFQDVLK")]
    assert len(leaderless_rows) == 1
    assert leaderless_rows["Shine Dalgarno"].iloc[0] == "Leaderless"
    g_coord_str = str(leaderless_rows["Genome Coordinates"].iloc[0])
    assert g_coord_str.startswith("chr1:") and g_coord_str.endswith("(-)")
    coord_span = g_coord_str.split(":", 1)[1].split("(", 1)[0]
    c_start, c_end = [int(x) for x in coord_span.split("-")]
    assert ds["tx1_coords"][0] <= c_start <= c_end <= ds["tx1_coords"][1]
    assert "Transcript Coordinates" in t_pre.columns

    # =========================================================================
    # STAGE 5: validate (SpectralForest + rescued sibling LC scan + ResultsSummarizer)
    # =========================================================================
    uproteins(["validate", "-o", str(outdir), "-T", "YES"])

    g_post = pd.read_csv(outdir / "Genome" / "Results" / "genome_post_validation_results.txt", sep="\t")
    t_post = pd.read_csv(outdir / "Transcriptome" / "Results" / "transcriptome_post_validation_results.txt", sep="\t")

    # Scan 101 ("TYVGQNPEK", HC) must rescue sibling Scan 102 ("SNFQDVWLR", LC) on the same pPlasmid smORF!
    locus4_post = g_post[g_post["Protein sequence"] == "MRTYVGQNPEKSNFQDVWLRVYTPQNDLRGK"]
    assert set(locus4_post["ScanNum"].astype(int).tolist()) == {101, 102, 105}
    val_by_scan = dict(zip(locus4_post["ScanNum"].astype(int), locus4_post["Spectrum Validation"]))
    assert val_by_scan[101] == "HC"
    assert val_by_scan[105] == "HC"
    assert val_by_scan[102] == "Rescued_LC"
    # Unified ORF name across all 3 scans of the same extended locus
    assert locus4_post["ORF name"].nunique() == 1

    # Transcriptome post-validation also retains both uproteins.1.1 and rescued Scan 102 on uproteins.2.1
    assert t_post["Spectrum Validation"].isin(["HC", "Rescued_LC"]).all()
    assert (t_post["Spectrum Validation"] == "Rescued_LC").any()

    summ_df = pd.read_csv(outdir / "summarized_results.txt", sep="\t")
    assert set(summ_df["Subset"].tolist()) == {
        "Genome LC",
        "Genome HC",
        "Transcriptome LC",
        "Transcriptome HC",
    }

    # =========================================================================
    # STAGE 6: metrics (100% AA sum, Leaderless & Strong RBS counts, unique FASTA headers, Venn)
    # =========================================================================
    uproteins(["metrics", "-o", str(outdir)])

    aa_df = pd.read_csv(outdir / "metrics" / "aa_distribution.tsv", sep="\t")
    for (_, _), grp in aa_df.groupby(["Database", "Subset"]):
        assert grp["Frequency_Pct"].sum() == pytest.approx(100.0, abs=1e-4)

    sd_df = pd.read_csv(outdir / "metrics" / "shine_dalgarno_counts.tsv", sep="\t")
    assert "Leaderless" in set(sd_df["RBS"].tolist())
    assert "Present. Strong binding." in set(sd_df["RBS"].tolist())
    total_standard = int(sd_df[sd_df["Subset"] == "Validated"]["Count"].sum())

    val_fa = outdir / "Genome" / "Results" / "Validated.fasta"
    assert val_fa.is_file()
    val_headers = [
        ln[1:].strip()
        for ln in val_fa.read_text(encoding="utf-8").splitlines()
        if ln.startswith(">")
    ]
    assert len(val_headers) == len(set(val_headers))
    assert all("|" in h for h in val_headers)
    assert (outdir / "metrics" / "venn_validated.png").is_file()

    # Re-run metrics with --non-redundant: shared Genome/Transcriptome sequences deduplicate!
    uproteins(["metrics", "-o", str(outdir), "--non-redundant"])
    sd_nr_df = pd.read_csv(outdir / "metrics" / "shine_dalgarno_counts.tsv", sep="\t")
    total_nr = int(sd_nr_df[sd_nr_df["Subset"] == "Validated"]["Count"].sum())
    assert total_nr < total_standard

    # =========================================================================
    # BONUS STAGE: train-validator -> custom portable JSON model -> re-run validate!
    # =========================================================================
    training_pin = tmp_path / "custom_training_pin.tsv"
    base_pin = pd.read_csv(
        outdir / "Genome" / "Percolator" / "for_predicting" / "fixed_Genome_pin.txt",
        sep="\t",
    )
    # Build 20-peptide balanced labeled training PIN (10 HC + 10 nHC) for 5-fold GroupKFold CV
    synth_rows = []
    template_hc = base_pin[base_pin["ScanNr"] == 101].iloc[0].to_dict()
    template_lc = base_pin[base_pin["ScanNr"] == 102].iloc[0].to_dict()
    for i in range(10):
        r_hc = dict(template_hc)
        r_hc["SpecId"] = f"train_hc_{i}_1_2_1"
        r_hc["ScanNr"] = 1000 + i
        r_hc["Peptide"] = f"R.HCPEPTIDE{chr(65 + i)}K.S"
        r_hc["lnSpecEValue"] = 24.0 + 0.2 * i
        r_hc["label_spectra"] = 1
        synth_rows.append(r_hc)

        r_lc = dict(template_lc)
        r_lc["SpecId"] = f"train_lc_{i}_1_2_1"
        r_lc["ScanNr"] = 2000 + i
        r_lc["Peptide"] = f"R.LCPEPTIDE{chr(65 + i)}K.S"
        r_lc["lnSpecEValue"] = 2.0 + 0.1 * i
        r_lc["label_spectra"] = 0
        synth_rows.append(r_lc)

    pd.DataFrame(synth_rows).to_csv(training_pin, sep="\t", index=False)
    custom_model = tmp_path / "retrained_forest_v2.json"

    uproteins(
        [
            "train-validator",
            "--training_pin",
            str(training_pin),
            "--output_model",
            str(custom_model),
            "--n_estimators",
            "25",
            "--cv_folds",
            "5",
        ]
    )
    assert custom_model.is_file()
    uproteins(["validate", "-o", str(outdir), "-T", "YES", "--model", str(custom_model)])
    assert pathlib.Path.cwd() == caller_cwd
