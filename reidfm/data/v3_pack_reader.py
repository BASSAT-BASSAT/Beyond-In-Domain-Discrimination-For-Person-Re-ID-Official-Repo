"""
REIDPACK3 random-access image reader.

Features:
- Reads images without full PACK extraction
- Verifies embedded SQLite index SHA256
- Verifies image payload SHA256
- Resolves Global manifest image paths
- Does not depend on FastReID

Important:
Create a separate reader inside each DataLoader worker.
"""

from __future__ import annotations

from pathlib import Path

import hashlib
import os
import posixpath
import sqlite3
import struct
import tempfile


HEADER = b"REIDPACK3\n"
FOOTER = struct.Struct("<8sQQ32s")

ALIASES = {
    "ORBench_RGB": "ORBench",
    "Occluded-PoseTrack-ReID":
        "Occluded_PoseTrack_ReID",
}


def _suffix_score(a: str, b: str) -> int:

    left = [
        part.casefold()
        for part in a.split("/")
        if part
    ]

    right = [
        part.casefold()
        for part in b.split("/")
        if part
    ]

    matched = 0

    for x, y in zip(
        reversed(left),
        reversed(right)
    ):
        if x != y:
            break
        matched += 1

    return matched


class V3PackReader:

    def __init__(
        self,
        pack_root: str | Path,
        dataset_root_id: str,
    ):

        name = ALIASES.get(
            dataset_root_id,
            dataset_root_id,
        )

        self.pack_path = (
            Path(pack_root) / f"{name}.pack"
        )

        if not self.pack_path.is_file():
            raise FileNotFoundError(
                self.pack_path
            )

        self._tmp = tempfile.TemporaryDirectory(
            prefix="reid_v3_index_"
        )

        try:
            self._index_path = (
                Path(self._tmp.name) /
                "index.sqlite"
            )

            self._verify_and_extract_index()

            uri = (
                self._index_path.as_uri() +
                "?mode=ro"
            )

            self.db = sqlite3.connect(
                uri,
                uri=True,
            )

            self.f = self.pack_path.open("rb")

        except BaseException:
            self._tmp.cleanup()
            raise


    def _verify_and_extract_index(self):

        with self.pack_path.open("rb") as f:

            assert (
                f.read(len(HEADER)) == HEADER
            ), "Invalid PACK header"

            f.seek(0, os.SEEK_END)
            total = f.tell()

            assert total > (
                len(HEADER) + FOOTER.size
            )

            f.seek(total - FOOTER.size)

            marker, start, size, digest = (
                FOOTER.unpack(
                    f.read(FOOTER.size)
                )
            )

            assert marker == b"RIDX3END"

            assert start >= len(HEADER)

            assert (
                start + size ==
                total - FOOTER.size
            )

            self._payload_limit = start

            f.seek(start)

            h = hashlib.sha256()

            with self._index_path.open(
                "wb"
            ) as out:

                remaining = size

                while remaining:

                    block = f.read(
                        min(
                            4 * 1024 * 1024,
                            remaining,
                        )
                    )

                    if not block:
                        raise EOFError(
                            "Incomplete PACK index"
                        )

                    out.write(block)
                    h.update(block)

                    remaining -= len(block)

            assert h.digest() == digest, (
                "PACK index SHA256 mismatch"
            )


    def read_bytes(
        self,
        relative_path: str,
    ) -> bytes:

        requested = relative_path.replace(
            "\\", "/"
        )

        if (
            requested.startswith("/")
            or ".." in requested.split("/")
        ):
            raise ValueError(
                "Unsafe relative image path"
            )

        requested = posixpath.normpath(
            requested
        )

        if requested in ("", "."):
            raise ValueError(
                "Empty image path"
            )

        # First, try exact path matching.
        for column in (
            "physical_path",
            "context_path",
            "leaf_path",
        ):

            rows = self.db.execute(
                f"""
                SELECT offset, bytes, sha256
                FROM images
                WHERE {column}=?
                """,
                (requested,),
            ).fetchall()

            if len(rows) == 1:
                return self._load(*rows[0])

            if len(rows) > 1:
                raise ValueError(
                    "Ambiguous image path: "
                    + requested
                )

        # Fall back to suffix matching.
        basename = requested.rsplit(
            "/", 1
        )[-1]

        matches = self.db.execute(
            """
            SELECT
                leaf_path,
                context_path,
                offset,
                bytes,
                sha256
            FROM images
            WHERE basename=?
            """,
            (basename,),
        ).fetchall()

        if not matches:
            raise FileNotFoundError(
                requested
            )

        ranked = []

        for (
            leaf,
            context,
            offset,
            size,
            digest,
        ) in matches:

            score = max(
                _suffix_score(
                    requested, leaf
                ),
                _suffix_score(
                    requested, context
                ),
            )

            ranked.append(
                (
                    score,
                    offset,
                    size,
                    digest,
                )
            )

        best = max(
            row[0] for row in ranked
        )

        winners = [
            row for row in ranked
            if row[0] == best
        ]

        if (
            len(winners) != 1
            or best < 1
        ):
            raise ValueError(
                "Ambiguous suffix mapping: "
                + requested
            )

        return self._load(
            *winners[0][1:]
        )


    def _load(
        self,
        offset: int,
        nbytes: int,
        digest: str,
    ) -> bytes:

        if (
            offset < len(HEADER)
            or nbytes <= 0
            or offset + nbytes >
                self._payload_limit
        ):
            raise ValueError(
                "Invalid PACK image bounds"
            )

        self.f.seek(offset)

        data = self.f.read(nbytes)

        if len(data) != nbytes:
            raise EOFError(
                "Incomplete image payload"
            )

        actual = hashlib.sha256(
            data
        ).hexdigest()

        if actual != digest:
            raise ValueError(
                "Image SHA256 mismatch"
            )

        return data


    def close(self):

        if hasattr(self, "f"):
            self.f.close()

        if hasattr(self, "db"):
            self.db.close()

        self._tmp.cleanup()


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()
