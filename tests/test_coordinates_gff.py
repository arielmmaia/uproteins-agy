from argparse import Namespace
import pathlib
import pandas as pd
import pytest

from src.postprocess.resultwrapper import ResultsWrapper
from src.postprocess.specfilt import PostPercolator
from src.sequtils.locus import (
    GenomeCoordinates,
    GenomeCoordinatesRNA,
    RefSeqGFF,
    StringTieGFF,
    load_genome_contigs,
    project_transcript_interval_to_genome,
    read_gff_dataframe,
)
from src.sequtils.orflib import AltCodons
from src.sequtils.utilities import FastaConverter, PercolatorConverter


def test_project_transcript_interval_single_and_multi_exon_both_strands():
    # 1. Single-exon + transcript [100..200], local ORF [15..62] -> [114..161] (+)
    g_lo, g_hi, g_strand = project_transcript_interval_to_genome(
        exons=[(100, 200)],
        tx_strand="+",
        tx_orf_start=15,
        tx_orf_end=62,
        orf_strand="forward",
    )
    assert (g_lo, g_hi, g_strand) == (114, 161, "+")

    # 2. Single-exon - transcript [100..200], local ORF [1..30]:
    # 5' nt 1 is genomic 200; nt 30 is genomic 171 -> [171..200] (-), start <= end!
    g_lo_m, g_hi_m, g_strand_m = project_transcript_interval_to_genome(
        exons=[(100, 200)],
        tx_strand="-",
        tx_orf_start=1,
        tx_orf_end=30,
        orf_strand="forward",
    )
    assert (g_lo_m, g_hi_m, g_strand_m) == (171, 200, "-")
    assert g_lo_m <= g_hi_m

    # 3. Multi-exon + transcript: exon 1 [100..109] (10 nt), exon 2 [200..249] (50 nt)
    # Local ORF [5..25] spans nt 5 of exon 1 (104) to nt 15 of exon 2 (214)
    g_lo_sp, g_hi_sp, g_strand_sp = project_transcript_interval_to_genome(
        exons=[(100, 109), (200, 249)],
        tx_strand="+",
        tx_orf_start=5,
        tx_orf_end=25,
        orf_strand="forward",
    )
    assert (g_lo_sp, g_hi_sp, g_strand_sp) == (104, 214, "+")

    # 4. Multi-exon - transcript: exon A [100..149] (50 nt), exon B [200..209] (10 nt)
    # 5'->3' RNA starts at 209..200 (10 nt), then 149..100 (50 nt).
    # Local ORF [5..25]: nt 5 -> 209 - 4 = 205; nt 25 -> 149 - (15 - 1) = 135 -> [135..205] (-)
    g_lo_sm, g_hi_sm, g_strand_sm = project_transcript_interval_to_genome(
        exons=[(100, 149), (200, 209)],
        tx_strand="-",
        tx_orf_start=5,
        tx_orf_end=25,
        orf_strand="forward",
    )
    assert (g_lo_sm, g_hi_sm, g_strand_sm) == (135, 205, "-")

    # 5. Unstranded '.' transcript translated on reverse frame -> '-' genomic strand
    g_lo_u, g_hi_u, g_strand_u = project_transcript_interval_to_genome(
        exons=[(100, 200)],
        tx_strand=".",
        tx_orf_start=15,
        tx_orf_end=62,
        orf_strand="reverse",
    )
    assert (g_lo_u, g_hi_u, g_strand_u) == (114, 161, "-")


def test_multicontig_genome_and_altcodons_plasmid_lookup(tmp_path: pathlib.Path):
    genome_fa = tmp_path / "multicontig.fasta"
    # Chromosome 1 has NO ATG upstream, while plasmid pXO1 HAS an upstream ATG -> MKLPEPTIDER*
    chr1_seq = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    # On pXO1: pos 1..3 = ATG, pos 4..9 = AAACTG (KL), pos 10..39 = GTGCCCGAACCGACCATCGACGAACGCAAG (VPEPTIDERK), pos 40..42 = TGA
    pxo1_seq = "ATGAAACTGGTGCCCGAACCGACCATCGACGAACGCAAGTGACCCCCCCCCCCC"
    genome_fa.write_text(
        f">chr1 chromosome\n{chr1_seq}\n>pXO1 virulence_plasmid\n{pxo1_seq}\n"
    )

    contigs, default_contig = load_genome_contigs(genome_fa)
    assert default_contig == "chr1"
    assert set(contigs.keys()) == {"chr1", "pXO1"}

    # Verify GenomeCoordinates emits contig-qualified coordinates when >1 contig is present
    psm_tsv = tmp_path / "psm.tsv"
    pd.DataFrame(
        {
            "PSMId": ["run_SII_1_1_10_2_1", "run_SII_2_2_11_2_1"],
            "score": [3.0, 2.8],
            "q-value": [0.001, 0.002],
            "posterior_error_prob": [0.001, 0.002],
            "peptide": ["-.PEPTIDER.K", "-.MAAACCC.K"],
            "proteinIds": [
                "gORF_pXO1_1_10-42_forward",
                "gORF_chr1_1_1-30_forward",
            ],
        }
    ).to_csv(psm_tsv, sep="\t", index=False)

    gc = GenomeCoordinates(psm_tsv).get_coords()
    assert gc.coordinates == ["pXO1:10-42(+)", "chr1:1-30(+)"]

    # Verify AltCodons reads pXO1 sequence (GTG at 10..12, extending upstream to ATG at 1..3)
    # rather than truncating to genome_seq[0] (chr1 = all C's)!
    res02 = tmp_path / "genome_results_02.txt"
    pd.DataFrame(
        {
            "SpecFile": ["run.mzML"],
            "SpecID": ["index=1"],
            "ScanNum": [10],
            "FragMethod": ["HCD"],
            "Precursor": [500.0],
            "Genome Coordinates": ["pXO1:10-42(+)"],
            "ORF Sequence": ["VPEPTIDERK"],
            "IsotopeError": [0],
            "PrecursorError(ppm)": [0.5],
            "Charge": [2],
            "Peptide": ["-.PEPTIDER.K"],
            "Protein": ["gORF_pXO1_1_10-42_forward"],
            "DeNovoScore": [80],
            "MSGFScore": [90],
            "SpecEValue": [1e-10],
            "EValue": [1e-7],
        }
    ).to_csv(res02, sep="\t", index=False)

    alts = AltCodons(file=res02, genome=genome_fa, maxsize=300, subset="Genome")
    stop_key = "pXO1:42"
    assert stop_key in alts.alternatives
    initial_orf = list(alts.alternatives[stop_key])[0]
    assert initial_orf.chromosome == "pXO1"
    assert initial_orf.start_codon == "GTG"


def test_results_wrapper_preserves_genomic_coordinates_and_projects_transcriptome(
    tmp_path: pathlib.Path,
):
    # Minus-strand 2-exon transcript on NC_000962.3:
    # Exon 1: 100..149 (50 nt), Exon 2: 200..219 (20 nt) -> 5' starts at 219
    gtf_path = tmp_path / "assembled.gtf"
    gtf_content = (
        "# StringTie version 2.2.1\n"
        'NC_000962.3\tStringTie\ttranscript\t100\t219\t1000\t-\t.\tgene_id "uproteins.5"; transcript_id "uproteins.5.2";\n'
        'NC_000962.3\tStringTie\texon\t100\t149\t1000\t-\t.\tgene_id "uproteins.5"; transcript_id "uproteins.5.2"; exon_number "1";\n'
        'NC_000962.3\tStringTie\texon\t200\t219\t1000\t-\t.\tgene_id "uproteins.5"; transcript_id "uproteins.5.2"; exon_number "2";\n'
    )
    gtf_path.write_text(gtf_content)
    original_bytes = gtf_path.read_bytes()

    tx_folder = tmp_path / "Transcriptome"
    post_perc = tx_folder / "post_perc"
    post_perc.mkdir(parents=True)
    res04_path = post_perc / "transcriptome_results_04.txt"

    # Unextended tORF was local RNA [10..40]; AltCodons extended start upstream to local RNA [4..40]
    # On '-' 2-exon transcript:
    # local nt 4 -> 219 - 3 = 216 (in Exon 2)
    # local nt 40 -> 149 - (20 - 1) = 130 (in Exon 1)
    # True extended genomic coordinates MUST be 130-216 (NEVER local RNA 4-40!)
    pd.DataFrame(
        {
            "SpecFile": ["run1.mzML"],
            "SpecID": ["index=5"],
            "ScanNum": [55],
            "Extended ORF": ["tORF_extended_4-40_forward"],
            "Extended Sequence": ["MKLPEPTIDER"],
            "Free Energy": [-9.2],
            "Shine Dalgarno": ["Strong"],
            "Extended Coordinates": ["4-40"],
            "FragMethod": ["HCD"],
            "Precursor": [612.3],
            "Fixed Peptides": ["PEPTIDER"],
            "Genome Coordinates": ["130-210"],
            "ORF Sequence": ["LPEPTIDER"],
            "IsotopeError": [0],
            "PrecursorError(ppm)": [0.8],
            "Charge": [2],
            "Peptide": ["K.PEPTIDER.A"],
            "Protein": ["tORF_uproteins.5.2_1_10-40_forward"],
            "DeNovoScore": [85],
            "MSGFScore": [100],
            "SpecEValue": [1e-11],
            "EValue": [1e-8],
        }
    ).to_csv(res04_path, sep="\t", index=False)

    wrapper = ResultsWrapper(
        df=res04_path,
        folder=str(tx_folder),
        filetype="transcriptome",
        transcriptome_gff=gtf_path,
    )
    wrapper.reformat(pre_validation=True)

    out_file = tx_folder / "Results" / "transcriptome_pre_validation_results.txt"
    out_df = pd.read_csv(out_file, sep="\t")
    assert out_df.loc[0, "ORF name"] == "tORF_uproteins.5.2_1"
    assert out_df.loc[0, "Genome Coordinates"] == "130-216"
    assert out_df.loc[0, "Transcript Coordinates"] == "4-40"
    # Confirm GTF file on disk was NEVER modified in place!
    assert gtf_path.read_bytes() == original_bytes


def test_nondestructive_gff_readers_and_optional_gff_none(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    # 1. GFF3 with 7 header directives, RefSeq + Protein Homology + Prodigal + Prokka + Bakta sources,
    #    plus an embedded ##FASTA tail (as emitted by Prokka/Bakta)
    gff_path = tmp_path / "complex_bacterial.gff3"
    gff_text = (
        "##gff-version 3\n"
        "#!gff-spec-version 1.21\n"
        "#!processor NCBI annotwriter\n"
        "##sequence-region NC_000962.3 1 4411532\n"
        "##sequence-region pXO1 1 181677\n"
        "# comment line 6\n"
        "# comment line 7\n"
        "NC_000962.3\tRefSeq\tregion\t1\t4411532\t.\t+\t.\tID=NC_000962.3:1..4411532\n"
        "NC_000962.3\tProtein Homology\tCDS\t1524\t2500\t.\t+\t0\tID=cds-WP_003400001.1;Parent=gene-dnaA;locus_tag=Rv0001;Name=dnaA\n"
        "NC_000962.3\tProdigal:2.6\tCDS\t3000\t3300\t.\t-\t0\tID=1_2;locus_tag=PROKKA_00002\n"
        "pXO1\tBakta\tCDS\t500\t800\t.\t+\t0\tID=BAKTA_00010;Name=pagA\n"
        "##FASTA\n"
        ">NC_000962.3\n"
        "ATGCATGCATGCATGC\n"
    )
    gff_path.write_text(gff_text)
    before_bytes = gff_path.read_bytes()

    df_gff = read_gff_dataframe(gff_path)
    assert len(df_gff) == 4
    ref_dict = RefSeqGFF(gff_path).get_dict()
    assert gff_path.read_bytes() == before_bytes  # Strictly non-destructive!

    assert ref_dict["Rv0001"].start == 1524
    assert ref_dict["Rv0001_ANNO"].end == 2500
    assert ref_dict["WP_003400001.1"].start == 1524
    assert ref_dict["PROKKA_00002"].start == 3000
    assert ref_dict["PROKKA_00002"].strand == "-"
    assert ref_dict["BAKTA_00010"].chromosome == "pXO1"
    assert ref_dict["pagA_ANNO"].start == 500

    # 2. Omitted --gff (gff=None) MUST NOT crash RefSeqGFF, PercolatorConverter, FastaConverter, or PostPercolator!
    assert RefSeqGFF(None).get_dict() == {}

    monkeypatch.chdir(tmp_path)
    (tmp_path / "Genome" / "Percolator").mkdir(parents=True)
    (tmp_path / "Genome" / "post_perc").mkdir(parents=True)
    pout_path = tmp_path / "Genome" / "Percolator" / "genome_results_psm.txt"
    pout_path.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tproteinIds\n"
        "run1_SII_1_1_100_2_1\t2.5\t0.004\t0.001\t-.MPEPTIDER.K\tgORF_NC_000962.3_1_15-62_forward\n"
        "run1_SII_2_2_101_2_1\t3.1\t0.002\t0.0005\t-.MKLLEIR.K\tRv0001_ANNO\n"
    )

    conv = PercolatorConverter(pout=str(pout_path), handle="psm", gff=None)
    assert conv.gff.empty

    fa_in = tmp_path / "in.fasta"
    fa_in.write_text(">sp|P00001|DNAA_MYCTU\nMKLLEIRK\n")
    FastaConverter(fasta=str(fa_in), conversion_file=None, gff=None).convert_entries()
    assert (tmp_path / "converted_genome_database.fasta").is_file()

    args = Namespace(
        mode="postms",
        genome=tmp_path / "dummy.fasta",
        proteome=fa_in,
        gff=None,
        transcriptome=False,
    )
    pp = PostPercolator(args, folder="Genome", filetype="genome")
    pp.convert_output()
    pp.get_coordinates_genome()
    coords_df = pd.read_csv(
        tmp_path / "Genome" / "post_perc" / "genome_psm_coords.txt", sep="\t"
    )
    assert coords_df.loc[0, "Genome Coordinates"] == "15-62"
    assert coords_df.loc[1, "Genome Coordinates"] == "not found"
