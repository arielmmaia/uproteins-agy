# Copyright © 2025 Eduardo Vieira de Souza
# Copyright © 2025 Adriana Canedo
# Copyright © 2025 Cristiano Valim Bizarro
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


from argparse import Namespace
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.pipelines.pms import PostMSPipeline
from src.postprocess.specfilt import (
    AnnoFilter,
    Coordinator,
    PostPercolator,
    SubgroupFDR,
    classify_protein_group,
    compute_tdc_qvalues,
)


def test_classify_protein_group_mutually_exclusive():
    assert classify_protein_group("Rv0001_ANNO") == "canonical_target"
    assert classify_protein_group("gORF_c1_1_1-30_forward,Rv0001_ANNO") == "canonical_target"
    assert classify_protein_group("contaminant_TRYPSIN") == "canonical_target"
    assert classify_protein_group("gORF_c1_1_1-30_forward") == "novel_target"
    assert classify_protein_group("decoy_Rv0001_ANNO") == "canonical_decoy"
    assert classify_protein_group("decoy_contaminant_TRYPSIN") == "canonical_decoy"
    assert classify_protein_group("decoy_gORF_c1_1_1-30_forward") == "novel_decoy"


def test_subgroup_fdr_prevents_global_fdr_inflation_in_novel_smorfs():
    """Statistical regression test (Issue #25 / C3):
    Suppose 10,000 canonical `_ANNO` PSMs score >= 2.5, 100 true novel `gORF` PSMs
    score in [2.0, 3.0] with 0 novel decoys >= 2.0, and 50 false novel `gORF` PSMs
    + 49 `decoy_gORF_` PSMs score in [1.0, 1.9].
    - Global Percolator q-value at score 1.0 is (49+1)/(10,150) = 0.0049 <= 0.01,
      so global 1% FDR leaks ALL 50 false novel hits (true novel FDR = 33.3%!).
    - SubgroupFDR restricts TDC to `T_novel` vs `D_novel`:
      - The 100 true novel hits at score >= 2.0 have `novel_psm_q_value == 0.01` (retained).
      - All 50 false novel hits in [1.0, 1.9] have `novel_psm_q_value >= 0.33` (rejected!).
    """
    novel_rows = []
    # 100 true novel smORF PSMs (distinct peptides) with scores 3.00 down to 2.01
    for i in range(100):
        sc = 3.0 - i * 0.009
        pep = f"R.NOVELTRUE{i:03d}K.A"
        # Convert digits to valid AA tag so strip_psm_peptide keeps distinct sequences
        aa_tag = "".join("ACDEFGHIKLMNPQRSTVWY"[int(ch)] for ch in f"{i:03d}")
        pep = f"R.MTRUE{aa_tag}PEPK.A"
        novel_rows.append({
            "PSMId": f"run1_SII_{i+1}_1_{1000+i}_2_1",
            "score": sc,
            "q-value": 0.0002,  # tiny global q-value
            "posterior_error_prob": 0.001,
            "peptide": pep,
            "Genome Coordinates": f"{10*i+1}-{10*i+30}",
            "proteinIds": f"gORF_c1_{i+1}_{10*i+1}-{10*i+30}_forward",
        })

    # 50 false novel smORF PSMs in [1.0, 1.49] that leaked through global q <= 0.005!
    for j in range(50):
        sc = 1.49 - j * 0.009
        aa_tag = "".join("ACDEFGHIKLMNPQRSTVWY"[int(ch)] for ch in f"{j:03d}")
        novel_rows.append({
            "PSMId": f"run1_SII_{200+j}_1_{2000+j}_2_1",
            "score": sc,
            "q-value": 0.0049,  # leaked under global 1% FDR!
            "posterior_error_prob": 0.15,
            "peptide": f"R.MFALSE{aa_tag}PEPK.A",
            "Genome Coordinates": f"{5000+10*j}-{5030+10*j}",
            "proteinIds": f"gORF_c1_{200+j}_{5000+10*j}-{5030+10*j}_forward",
        })

    decoy_rows = []
    # 49 novel decoys interleaved in [1.01, 1.50] + 10 canonical decoys
    for k in range(49):
        sc = 1.50 - k * 0.009
        aa_tag = "".join("ACDEFGHIKLMNPQRSTVWY"[int(ch)] for ch in f"{k:03d}")
        decoy_rows.append({
            "PSMId": f"run1_SII_{400+k}_1_{4000+k}_2_1",
            "score": sc,
            "q-value": 0.0049,
            "posterior_error_prob": 0.20,
            "peptide": f"R.MDECOY{aa_tag}PEPK.A",
            "proteinIds": f"decoy_gORF_c1_{400+k}_1-30_forward",
        })
    for c in range(10):
        # High-ish canonical decoy scores must NOT penalize the novel subgroup!
        decoy_rows.append({
            "PSMId": f"run1_SII_{600+c}_1_{6000+c}_2_1",
            "score": 2.4,
            "q-value": 0.001,
            "posterior_error_prob": 0.01,
            "peptide": f"R.MCANNO{'ACDEFGHIKL'[c]}K.A",
            "proteinIds": f"decoy_Rv{c:04d}_ANNO",
        })

    novel_df = pd.DataFrame(novel_rows)
    decoy_df = pd.DataFrame(decoy_rows)

    scored = SubgroupFDR(pi0=1.0).score_novel_subgroup(novel_df, decoy_df)
    passing = scored[
        (scored["novel_psm_q_value"] <= 0.01)
        & (scored["novel_peptide_q_value"] <= 0.01)
    ]

    assert len(passing) == 100
    assert all(passing["score"] >= 2.0)
    rejected = scored[scored["score"] < 2.0]
    assert len(rejected) == 50
    assert all(rejected["global_q_value"] <= 0.005)
    assert all(rejected["novel_psm_q_value"] > 0.029)
    assert rejected["novel_psm_q_value"].iloc[-1] == pytest.approx(1.0 / 3.0)


def test_subgroup_peptide_fdr_catches_repeated_spectrum_psm_inflation():
    """Verify `novel_peptide_q_value` prevents a single abundant 200-PSM novel peptide
    from masking false single-spectrum novel peptides at the peptide level."""
    novel_rows = []
    # 1 abundant novel peptide observed across 200 spectra at score 3.5 -> 1 distinct peptide!
    for i in range(200):
        novel_rows.append({
            "PSMId": f"run1_SII_{i+1}_1_{100+i}_2_1",
            "score": 3.5,
            "q-value": 0.001,
            "posterior_error_prob": 0.0001,
            "peptide": "R.MABUNDANTPEPK.A",
            "Genome Coordinates": "1-39",
            "proteinIds": "gORF_c1_1_1-39_forward",
        })
    # 1 weak novel peptide at score 1.2 (below 1 novel decoy at score 1.5):
    # PSM-level q = (1+1)/201 = 0.00995 <= 0.01,
    # BUT peptide-level q = (1+1)/2 = 1.0 > 0.01 -> MUST BE REJECTED by novel_peptide_q_value!
    novel_rows.append({
        "PSMId": "run1_SII_999_1_999_2_1",
        "score": 1.2,
        "q-value": 0.005,
        "posterior_error_prob": 0.05,
        "peptide": "R.MWEAKSINGLEK.A",
        "Genome Coordinates": "200-239",
        "proteinIds": "gORF_c1_2_200-239_forward",
    })
    decoy_df = pd.DataFrame([{
        "PSMId": "run1_SII_888_1_888_2_1",
        "score": 1.5,
        "q-value": 0.005,
        "posterior_error_prob": 0.05,
        "peptide": "R.MDECOYSINGLEK.A",
        "proteinIds": "decoy_gORF_c1_9_500-539_forward",
    }])

    scored = SubgroupFDR(pi0=1.0).score_novel_subgroup(pd.DataFrame(novel_rows), decoy_df)
    weak_row = scored[scored["PSMId"] == "run1_SII_999_1_999_2_1"].iloc[0]
    assert weak_row["novel_psm_q_value"] < 0.01
    assert weak_row["novel_peptide_q_value"] == pytest.approx(1.0)


def test_coordinator_blocks_high_q_psm_leak_substring_collision_and_honors_pep(
    tmp_path: pathlib.Path,
):
    """Regression test (Issue #25 / C3):
    1. Scan 100 (`q=0.002`, `PEP=0.005`) maps to `gORF_1` -> KEPT.
    2. Scan 101 (`q=0.85`, `PEP=0.95`) ALSO maps to `gORF_1` in `proteined.tsv` -> MUST NOT LEAK!
    3. Scan 102 (`q=0.003`, `PEP=0.25`) maps to `gORF_2` -> REJECTED when `--pep 0.05` is enforced!
    4. Scan 100 in `proteined.tsv` also has a row for `gORF_10` at scan 999 -> `gORF_1` must NOT
       substring-match `gORF_10`!
    """
    utps = tmp_path / "genome_utps.txt"
    utps.write_text(
        "PSMId\tscore\tq-value\tposterior_error_prob\tpeptide\tGenome Coordinates\tproteinIds\n"
        "run1_SII_1_1_100_2_1\t2.8\t0.002\t0.005\t-.MPEPTIDER.K\t15-62\tgORF_1\n"
        "run1_SII_2_1_101_2_1\t-1.2\t0.850\t0.950\t-.MNOISEPEPK.R\t15-62\tgORF_1\n"
        "run1_SII_3_1_102_2_1\t2.1\t0.003\t0.250\t-.MHIGHPEPR.K\t100-160\tgORF_2\n"
    )

    proteined = tmp_path / "genome_proteined.tsv"
    proteined.write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tORF Sequence\tIsotopeError\t"
        "PrecursorError(ppm)\tCharge\tPeptide\tProtein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
        "run1.mzML\tindex=1\t100\tHCD\t500.2\tMPEPTIDERK\t0\t1.0\t2\t-.MPEPTIDER.K\tgORF_1(pre=-,post=K)\t80\t95\t1e-10\t1e-7\n"
        "run1.mzML\tindex=2\t101\tHCD\t510.2\tMPEPTIDERK\t0\t5.0\t2\t-.MNOISEPEPK.R\tgORF_1(pre=-,post=R)\t20\t15\t1e-2\t10.0\n"
        "run1.mzML\tindex=3\t102\tHCD\t520.2\tMHIGHPEPRK\t0\t1.5\t2\t-.MHIGHPEPR.K\tgORF_2(pre=-,post=K)\t75\t85\t1e-9\t1e-6\n"
        "run1.mzML\tindex=9\t100\tHCD\t500.2\tMDECOYCOLLISION\t0\t1.0\t2\t-.MPEPTIDER.K\tgORF_10(pre=-,post=K)\t80\t95\t1e-10\t1e-7\n"
    )

    out_res02 = tmp_path / "genome_results_02.txt"
    coord = Coordinator(utps=utps, proteined=proteined, qvalue=0.01, pep=0.05)
    coord.add_information(out_res02)

    df = pd.read_csv(out_res02, sep="\t")
    assert len(df) == 1
    assert int(df.iloc[0]["ScanNum"]) == 100
    assert df.iloc[0]["Protein"] == "gORF_1"
    assert df.iloc[0]["Genome Coordinates"] == "15-62"


def test_zero_hit_postms_writes_clean_empty_results_without_crash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #25 / C3):
    When 0 novel smORFs survive subgroup FDR, PostMSPipeline skips AltCodons/RBS
    cleanly and writes valid empty pre-validation result tables without KeyError.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Genome" / "post_perc").mkdir(parents=True)
    # Empty genome_results_02.txt (header only, 0 data rows)
    (tmp_path / "Genome" / "post_perc" / "genome_results_02.txt").write_text(
        "SpecFile\tSpecID\tScanNum\tFragMethod\tPrecursor\tGenome Coordinates\t"
        "ORF Sequence\tIsotopeError\tPrecursorError(ppm)\tCharge\tPeptide\t"
        "Protein\tDeNovoScore\tMSGFScore\tSpecEValue\tEValue\n"
    )

    args = Namespace(qvalue=0.01, pep=0.05, rrna=None, maxsize=300, genome=None)
    pipe = PostMSPipeline(args=args, filetype="genome", folder="Genome")
    monkeypatch.setattr(pipe, "_run_percolator", lambda: None)
    monkeypatch.setattr(pipe, "_process_percolator", lambda: None)
    pipe.run()

    pre_val = tmp_path / "Genome" / "Results" / "genome_pre_validation_results.txt"
    assert pre_val.is_file()
    out_df = pd.read_csv(pre_val, sep="\t")
    assert out_df.empty
    assert "ORF name" in out_df.columns
    assert "Protein sequence" in out_df.columns
    assert "Genome Coordinates" in out_df.columns
