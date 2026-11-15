# Copyright © 2021-2025 Eduardo Vieira de Souza
# Copyright © 2021-2025 Adriana Canedo
# Copyright © 2021-2025 Cristiano Valim Bizarro
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


import sqlite3
import typing as t


class DatabaseGenerator(object):
    """Generates a SQLite or FASTA store for predicted ORFs."""

    def __init__(self, name: str = "database", db_type: str = 'sql', reset: bool = True):
        self.db = None
        self.__check_db_type(db_type, name, reset=reset)

    def __check_db_type(self, db_type: str, name: str, reset: bool = True):
        if db_type == 'sql':
            self.db = SQLDatabase(name, reset=reset)
        elif db_type == 'fasta':
            self.db = FastaDatabase()
        else:
            raise ValueError(
                f"Invalid database type '{db_type}'. Expected 'sql' or 'fasta'."
            )

    def add_orfs(self, orfs):
        """Adds ORFs to the database."""
        self.db.add_orfs(orfs)
        return self

    def retrieve(self):
        return self.db.retrieve()

    def to_fasta(self, filename: str = "db_orfs.fasta", identifier: str = 'g'):
        """Writes entries and sequences inside the database to a FASTA file."""
        data = self.retrieve()
        to_write = [
            f">{identifier}{orf[1]}_{orf[4]}-{orf[5]}_{orf[6]}\n{orf[2]}\n"
            for orf in data
        ]
        with open(filename, 'w') as fa:
            fa.writelines(to_write)


class FastaDatabase(object):
    def __init__(self):
        self._rows: list[tuple] = []

    def add_orfs(self, orfs):
        start_id = len(self._rows) + 1
        for offset, orf in enumerate(orfs):
            seq_str = str(orf.seq)
            self._rows.append((
                start_id + offset,
                str(orf.name),
                seq_str,
                len(seq_str),
                int(orf.start),
                int(orf.end),
                str(orf.strand or ""),
            ))

    def retrieve(self):
        return list(self._rows)


class SQLDatabase(object):
    def __init__(self, name: str, reset: bool = True):
        self.name = f'{name}.db'
        self.__create_database(reset=reset)

    def __connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.name)

    def __create_database(self, reset: bool = True):
        conn = self.__connect()
        try:
            with conn:
                if reset:
                    conn.execute("DROP TABLE IF EXISTS ORFOME;")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS ORFOME(
                        ID     INT PRIMARY KEY NOT NULL,
                        NAME   TEXT            NOT NULL,
                        SEQ    TEXT            NOT NULL,
                        LENGTH INT             NOT NULL,
                        START  INT             NOT NULL,
                        END    INT             NOT NULL,
                        STRAND TEXT            NOT NULL
                    );"""
                )
        finally:
            conn.close()
        return self

    def add_orfs(self, orfs: t.Iterable):
        """Insert ORFs in a single parameterized transaction."""
        conn = self.__connect()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COALESCE(MAX(ID), 0) FROM ORFOME;")
                next_id = int(cursor.fetchone()[0]) + 1
                rows = [
                    (
                        next_id + offset,
                        str(orf.name),
                        str(orf.seq),
                        len(orf),
                        int(orf.start),
                        int(orf.end),
                        str(orf.strand or ""),
                    )
                    for offset, orf in enumerate(orfs)
                ]
                if rows:
                    cursor.executemany(
                        """INSERT INTO ORFOME(ID, NAME, SEQ, LENGTH, START, END, STRAND)
                           VALUES (?, ?, ?, ?, ?, ?, ?);""",
                        rows,
                    )
        finally:
            conn.close()

    def retrieve(self):
        conn = self.__connect()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ID, NAME, SEQ, LENGTH, START, END, STRAND FROM ORFOME ORDER BY ID;"
            )
            return cursor.fetchall()
        finally:
            conn.close()
