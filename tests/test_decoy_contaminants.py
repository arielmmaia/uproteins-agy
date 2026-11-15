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


import pathlib

from Bio import SeqIO
import pytest

from src.database.database_generator import Database, load_forward_contaminants
from src.database.proteome_matcher import tryptic_peptides
from src.percolator import Decoy


def test_load_forward_contaminants_extracts_exact_121_forward_records():
    """Verify seqlib/contaminants.txt yields the exact 121 forward contaminant_* records
    and zero Reverse_contaminant_* records, preserving B/X/Z in headers and sequences."""
    records = load_forward_contaminants()
    assert len(records) == 121
    ids = [r.id for r in records]
    assert all(i.startswith("contaminant_") for i in ids)
    assert all(not i.startswith("Reverse_") for i in ids)

    # Check contaminant_UBIQUITIN07 ('B' in header!) and contaminant_INT-STD5 ('X' in sequence!)
    assert "contaminant_UBIQUITIN07" in ids
    std5 = next(r for r in records if r.id == "contaminant_INT-STD5")
    assert "X" in str(std5.seq)
    assert str(std5.seq) == "QSEWTYSEGVLDEKHWRLQYPDXGG"


def test_target_and_decoy_databases_symmetric_and_collision_free(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression test (Issue #19):
    1. Target genome_database.fasta includes all 121 forward contaminants (`>contaminant_...`).
    2. Decoy Genome_decoy.fasta has exact 1:1 count symmetry with target, EVERY entry starts with `decoy_`,
       ZERO forward `>contaminant_...` or `>Reverse_contaminant_...` entries appear in decoy FASTA,
       headers with B/X/Z are never corrupted, and sequences with X are not spliced.
    3. Zero decoy tryptic peptides (>=6 aa, I==L) collide with any target tryptic peptide,
       even for palindromic/homopolymeric target peptides (`MAAAAAAR`, `MELLEIR`).
    """
    monkeypatch.chdir(tmp_path)

    orfs_fa = tmp_path / "genome_ORFs.fasta"
    orfs_fa.write_text(
        ">gORF_NC_000962.3_1_1-24_forward\nMAAAAAAR\n"
        ">gORF_NC_000962.3_2_30-54_forward\nMELLEIRK\n"
        ">gORF_BXZ_TEST_3_60-90_forward\nMPEPTIXBZR\n"
    )
    proteome_fa = tmp_path / "proteome.fasta"
    proteome_fa.write_text(">Rv0001\nMTESTPEPTIDERKLLQ\n")

    db = Database(str(orfs_fa), str(proteome_fa), "genome")
    db.mark_annotated()
    db.unify()

    target_fa = tmp_path / "genome_database.fasta"
    target_records = list(SeqIO.parse(str(target_fa), "fasta"))
    target_ids = [r.id for r in target_records]

    # 3 ORFs + 1 _ANNO + 121 forward contaminants = 125 target records
    assert len(target_records) == 125
    contam_targets = [i for i in target_ids if i.startswith("contaminant_")]
    assert len(contam_targets) == 121
    assert not any(i.startswith("Reverse_") for i in target_ids)

    # Build Decoy FASTA
    decoy_builder = Decoy(db=str(target_fa), db_type="Genome")
    decoy_builder.reverse_sequences().to_fasta()

    decoy_fa = tmp_path / "Genome" / "Percolator" / "Genome_decoy.fasta"
    decoy_records = list(SeqIO.parse(str(decoy_fa), "fasta"))
    decoy_ids = [r.id for r in decoy_records]

    # Exact 1:1 symmetry & uniform `decoy_` prefix
    assert len(decoy_records) == len(target_records) == 125
    assert all(i.startswith("decoy_") for i in decoy_ids)
    assert "decoy_contaminant_UBIQUITIN07" in decoy_ids
    assert "decoy_gORF_BXZ_TEST_3_60-90_forward" in decoy_ids

    # Verify B/X/Z residues were preserved in sequence length (never deleted to splice flanking aa)
    bxz_target = next(r for r in target_records if r.id == "gORF_BXZ_TEST_3_60-90_forward")
    bxz_decoy = next(r for r in decoy_records if r.id == "decoy_gORF_BXZ_TEST_3_60-90_forward")
    assert len(bxz_decoy.seq) == len(bxz_target.seq)
    assert set("BXZ").issubset(set(str(bxz_decoy.seq)))

    # Verify ZERO >=6 aa (I==L) tryptic peptide collisions between target and decoy databases
    target_il = {
        pep.replace("I", "L")
        for rec in target_records
        for pep in tryptic_peptides(str(rec.seq), min_len=6, max_missed_cleavages=1, include_met_excision=True)
    }
    decoy_il = {
        pep.replace("I", "L")
        for rec in decoy_records
        for pep in tryptic_peptides(str(rec.seq), min_len=6, max_missed_cleavages=1, include_met_excision=True)
    }
    assert target_il.isdisjoint(decoy_il)
