# µProteInS v2.0.0 — Bacterial Proteogenomics Pipeline for smORF Microprotein Discovery

[![License: GPL v3 or later](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-10.1093%2Fbioinformatics%2Fbtac115-brightgreen.svg)](https://doi.org/10.1093/bioinformatics/btac115)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.0.0-informational.svg)](CHANGELOG.md)

**µProteInS** is an end-to-end bacterial proteogenomics pipeline for identifying, statistically validating, and characterizing novel microproteins encoded by small open reading frames (**smORFs**, typically $\le 100$ amino acids) from mass spectrometry (LC-MS/MS) and optional RNA-Seq data.

> **What's new in v2.0.0:** This release delivers a comprehensive 22-unit scientific, statistical, and software remediation of µProteInS v1.4.2 (`Issues #1–#43`, `PRs #2–#44` — see [`CHANGELOG.md`](CHANGELOG.md)), restoring compatibility with Python `>=3.10` / `pandas >=2.2` / `scikit-learn >=1.5` and resolving major proteogenomic methodology flaws across bacterial alignment, circular `oriC` translation, target-decoy contaminant design, class-specific subgroup FDR control, `I == L` peptide uniqueness, Shine-Dalgarno $\Delta G^\circ_{37}$ thermodynamics, alternative start-codon selection, and portable Spectrum Forest machine-learning validation.

![Workflow](images/figure_1_no_layers.png)

---

## Architecture & Scientific Methodology (v2.0.0)

µProteInS executes six sequential pipeline stages plus an optional classifier retraining command:

| Stage | Subcommand | Purpose & Key v2.0.0 Guarantees |
| :--- | :--- | :--- |
| **1** | `assembly` | **Bacterial RNA-Seq Alignment & Assembly:** Enforces un-spliced alignment (`--no-spliced-alignment`, opt-in `--allow_splicing`), streams HISAT2 SAM directly into `samtools sort` via OS pipes without uncompressed SAM disk/RAM bloat, and maps all library strandedness modes (`RF`, `FR`, `R`, `F`, and unstranded). |
| **2** | `database` | **Six-/Three-Frame smORF Translation & Proteome Matching:** Restores `--maxsize 300` (`100` aa) and `--minsize 18` (`6` aa) defaults, wraps circular bacterial chromosomes across `oriC` (`--circular` / `--linear`), supports NCBI genetic codes (`--genetic_code 11`, `4` for *Mycoplasma*, etc.) and IUPAC/soft-masked DNA, preserves contig and isoform IDs, excludes nested non-`ATG` canonical sub-ORF explosions and upstream readthroughs across canonical `GTG`/`TTG` starts, and places the 121 forward standard contaminants (`>contaminant_*`) into target FASTAs. |
| **3** | `ms` | **Target & Tryptic Pseudo-Reversed Decoy MS-GF+ Search:** Generates 1:1 peptide-level internal tryptic pseudo-reversed decoys (`decoy_*`, including `decoy_contaminant_*`) with $\ge 6$-aa (`I == L`) target collision masking, stages suffix-array index files inside `{Genome,Transcriptome}/db_index/` without polluting `--Mass_spec`, passes explicit `-o` paths on both target and decoy searches, and exposes `--java_mem`. |
| **4** | `postms` | **Per-Scan TDC, Class-Specific Subgroup FDR, Uniqueness, RBS & Start Selection:** Aligns enzyme specificity across MS-GF+, `msgf2pin`, and Percolator; performs top-1 per-scan Target-Decoy Competition (`compete_target_decoy_pin`); enforces **class-specific `SubgroupFDR`** (`novel_psm_q_value` and `novel_peptide_q_value`) separating novel smORFs from high-prior canonical `_ANNO` and contaminant hits; excludes `I == L`-equivalent and N-terminally Met-excised canonical/contaminant peptides (`CanonicalPeptideExclusionIndex`); preserves paralogous smORF loci (`Locus_Multiplicity`, `Paralog_Group_ID`); scores 16S rRNA–mRNA Shine-Dalgarno $\Delta G^\circ_{37}$ in pure Python (Xia/Turner 1998 nearest-neighbor rules, distinguishing `"Leaderless"` `<5`-nt 5' UTRs from `"None (SD absent)"`); projects exon-aware multi-contig coordinates (`start <= end` on `+` and `-`); and selects optimal start codons (`AltCodons`) retaining 100% of MS-detected peptides. |
| **5** | `validate` | **Spectrum Forest MS/MS Quality Validation & Sibling LC Rescue:** Evaluates the 200-tree Spectrum Forest via a pure-NumPy, version-independent `SpectralForestEngine` (`src/model/spectral_forest_v1_trees.json`) with frozen experiment-wide reference bounds, rescues sibling lower-confidence (`Rescued_LC`) scans supporting `HC`-validated microproteins, and writes `summarized_results.txt`. |
| **6** | `metrics` | **Summary Tables, Plots, Non-Redundant FASTAs & Overlap Venns:** Supports both dual (`Genome` + `Transcriptome`) and genome-only (`-T NO` or auto-detected) runs, normalizes amino-acid composition to `100.0%` per `(Database, Subset)`, tabulates Shine-Dalgarno / Leaderless classes without double-counting, honors `--non-redundant`, and exports collision-free `>{ORF_name}|{Genome_Coordinates}` FASTAs and Venn diagrams. |
| **Opt** | `train-validator` | **Custom Spectrum Forest Retraining:** Trains portable JSON Random Forest validators using database-size-invariant `lnSpecEValue`, unscaled tree splits, and `I == L` peptide-grouped `StratifiedGroupKFold` cross-validation. |

---

## Installation

### 1. Python Package (`Python >= 3.10`)

Using [`uv`](https://docs.astral.sh/uv/) or `pip`:

```bash
git clone https://github.com/arielmmaia/uproteins-agy.git
cd uproteins-agy

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[cli,test]"
```

### 2. External Bioinformatics Dependencies

Stage 2 (`database`), Stage 5 (`validate`), Stage 6 (`metrics`), and `train-validator` run in **100% pure Python** with zero external binaries required (including pure-Python Xia/Turner Shine-Dalgarno $\Delta G^\circ_{37}$ calculation in Stage 4 `--rbs_backend python`).

External C/Java tools are used by:
- **Stage 1 (`assembly`):** `hisat2`, `hisat2-build`, `samtools`, `stringtie`, `gffcompare`, `gffread`
- **Stage 3 (`ms`):** `java` (`>=11`) and `MSGFPlus.jar` (`msgf_plus`)
- **Stage 4 (`postms`):** `java` + `MSGFPlus.jar` (`MzIDToTsv`), `msgf2pin`, `percolator` (and optional Perl `free2bind` only if `--rbs_backend free2bind` is requested)

Install all external tools into a reproducible Conda/Mamba environment via [`environment.yml`](environment.yml) or helper script [`scripts/install_bio_tools.sh`](scripts/install_bio_tools.sh):

```bash
conda env create -f environment.yml
conda activate uproteins
pip install -e ".[cli,test]"
```

Tool paths are discovered automatically from `PATH` / Bioconda prefixes and can also be overridden via CLI flags (`--msgf_jar`, `--msgf2pin_path`, `--percolator_path`, `--gffcompare_path`, `--gffread_path`, `--free2bind_dir`) or environment variables:

| Environment Variable | Target Executable / Resource |
| :--- | :--- |
| `UPROTEINS_MSGF_JAR` | Path to `MSGFPlus.jar` |
| `UPROTEINS_JAVA` | Path to `java` executable |
| `UPROTEINS_MSGF2PIN` | Path to `msgf2pin` executable |
| `UPROTEINS_PERCOLATOR` | Path to `percolator` executable |
| `UPROTEINS_HISAT2` / `UPROTEINS_HISAT2_BUILD` | Paths to `hisat2` and `hisat2-build` |
| `UPROTEINS_SAMTOOLS` | Path to `samtools` |
| `UPROTEINS_STRINGTIE` | Path to `stringtie` |
| `UPROTEINS_GFFCOMPARE` / `UPROTEINS_GFFREAD` | Paths to `gffcompare` and `gffread` |
| `UPROTEINS_FREE2BIND_DIR` | Directory containing `free_align.pl` (optional) |

---

## Quickstart Workflow

### Step 1 — Transcriptome Assembly *(Optional; skip for Genome-only mode)*

```bash
uproteins assembly \
  -o results_run \
  -g reference/genome.fasta \
  -G reference/annotation.gtf \
  --reads1 rnaseq/sample1_R1.fastq.gz,rnaseq/sample2_R1.fastq.gz \
  --reads2 rnaseq/sample1_R2.fastq.gz,rnaseq/sample2_R2.fastq.gz \
  --strandness RF \
  --threads 8
```

### Step 2 — Custom Proteogenomic Database Generation

```bash
uproteins database \
  -o results_run \
  -g reference/genome.fasta \
  -p reference/uniprot_proteome.fasta \
  -T YES \
  --maxsize 300 \
  --minsize 18 \
  --circular \
  --genetic_code 11
```

### Step 3 — Target & Decoy Peptide Database Search (MS-GF+)

```bash
uproteins ms \
  -o results_run \
  -M mass_spec_mzml/ \
  -T YES \
  --inst 3 \
  --t 20ppm \
  --e 1 \
  --ntt 2 \
  --thread 8 \
  --java_mem 16G
```

### Step 4 — Subgroup FDR Filtering, Uniqueness, RBS Scoring & Start Selection

```bash
uproteins postms \
  -o results_run \
  -M mass_spec_mzml/ \
  -p reference/uniprot_proteome.fasta \
  -g reference/genome.fasta \
  --gff reference/annotation.gff3 \
  --rrna reference/16S_rrna.fasta \
  -T YES \
  --qvalue 0.01 \
  --pep 0.05 \
  --rbs_backend python
```

### Step 5 — Spectrum Forest Validation & Summary

```bash
uproteins validate \
  -o results_run \
  -T YES
```

### Step 6 — Compositional, Thermodynamic & Overlap Metrics

```bash
uproteins metrics \
  -o results_run \
  --non-redundant
```

### Optional — Retrain a Custom Spectrum Forest Model

```bash
uproteins train-validator \
  --training_pin labeled_psms.pin \
  --output_model custom_forest_v2.json \
  --label_column label_spectra \
  --n_estimators 200 \
  --cv_folds 5
```

---

## Running the Test Suite

The repository includes 236 deterministic unit, statistical, thermodynamic, ML parity, and full 6-stage end-to-end integration tests that run in ~10 seconds without requiring external C/Java bioinformatics binaries:

```bash
pytest -v
```

---

## Citations

If you use **µProteInS** in your research, please cite the original publications:

1. **Core Pipeline (*Bioinformatics*, 2022):**
   de Souza, E. V., Dalberto, P. F., Machado, V. P., Canedo Miranda, A., Saghatelian, A., Machado, P., Basso, L. A., & Bizarro, C. V. (2022). µProteInS — a proteogenomics pipeline for finding novel bacterial microproteins encoded by small ORFs. *Bioinformatics*, 38(9), 2612–2614. https://doi.org/10.1093/bioinformatics/btac115
2. **Protocol Chapter (*Methods in Molecular Biology*, 2024):**
   de Souza, E. V., & Bizarro, C. V. (2024). Identification of Novel Bacterial Microproteins Encoded by Small Open Reading Frames Using a Computational Proteogenomics Workflow. In *Protein Bioinformatics* (pp. 19–34). Springer US. https://doi.org/10.1007/978-1-0716-4007-4_2
3. **Large-Scale *M. tuberculosis* Application (*Scientific Reports*, 2024):**
   de Souza, E. V., Dalberto, P. F., Miranda, A. C., et al. (2024). Large-scale proteogenomics characterization of microproteins in *Mycobacterium tuberculosis*. *Scientific Reports*, 14, 31186. https://doi.org/10.1038/s41598-024-82465-w

---

## Changelog & Remediation Audit Trail

Every bug fix, statistical improvement, and architectural change between `v1.4.2` and `v2.0.0` is documented issue-by-issue and PR-by-PR in [`CHANGELOG.md`](CHANGELOG.md).

## License

µProteInS is free software distributed under the **GNU General Public License v3.0 or later** ([`LICENSE`](LICENSE)).
