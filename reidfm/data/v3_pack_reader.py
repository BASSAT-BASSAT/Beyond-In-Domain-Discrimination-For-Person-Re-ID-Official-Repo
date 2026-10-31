"""
REIDPACK3 random-access image reader.

Features:
- Reads images without full PACK extraction
- Verifies embedded SQLite index SHA256
- Verifies image payload SHA256
- Resolves Global manifest image paths
- Does not depend on FastReID
- Reuses verified on-disk SQLite indices across DataLoader workers on Linux
- Caches resolved locations and pre-tokenized suffix candidates per worker

Important:
Create a separate reader inside each DataLoader worker.
"""

from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

import hashlib
import os
import posixpath
import sqlite3
import struct
import tempfile

try:
    import fcntl
except ImportError:  # Windows: preserve the original private-index behavior.
    fcntl = None


HEADER = b"REIDPACK3\n"
FOOTER = struct.Struct("<8sQQ32s")

ALIASES = {
    "ORBench_RGB": "ORBench",
    "Occluded-PoseTrack-ReID":
        "Occluded_PoseTrack_ReID",
}

_RESOLVED_PATH_CACHE_SIZE = 8192
_BASENAME_CACHE_SIZE = 32
_SUFFIX_INDEX_CANDIDATE_LIMIT = 2048
_MISSING_SUFFIX = object()


def _reversed_casefold_parts(path: str) -> tuple[str, ...]:
    return tuple(part.casefold() for part in reversed(path.split("/")) if part)


def _suffix_score_parts(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    matched = 0
    for left, right in zip(a, b):
        if left != right:
            break
        matched += 1
    return matched


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

        self._tmp = None

        try:
            # The PACK index is immutable within a run. Historically every
            # worker copied and hashed it from the mounted PACK independently.
            # Share a verified local copy instead; PACK image bytes are still
            # fetched and SHA256-checked individually on every read.
            cache_enabled = (
                fcntl is not None
                and os.environ.get("REIDFM_V3_INDEX_CACHE", "1") != "0"
            )
            if cache_enabled:
                self._index_path = self._get_cached_index()
            else:
                self._tmp = tempfile.TemporaryDirectory(
                    prefix="reid_v3_index_"
                )
                self._index_path = Path(self._tmp.name) / "index.sqlite"
                start, size, digest = self._index_metadata()
                self._extract_index(self._index_path, start, size, digest)

            uri = (
                self._index_path.as_uri() +
                "?mode=ro"
            )

            self.db = sqlite3.connect(
                uri,
                uri=True,
            )

            self.f = self.pack_path.open("rb")

            # Worker-local metadata only: image bytes are never cached here.
            self._resolved_paths = OrderedDict()
            self._basename_candidates = OrderedDict()

        except BaseException:
            if self._tmp is not None:
                self._tmp.cleanup()
            raise


    def _index_metadata(self):

        with self.pack_path.open("rb") as f:

            if f.read(len(HEADER)) != HEADER:
                raise ValueError("Invalid PACK header")

            f.seek(0, os.SEEK_END)
            total = f.tell()

            if total <= len(HEADER) + FOOTER.size:
                raise ValueError("PACK too small")

            f.seek(total - FOOTER.size)

            marker, start, size, digest = (
                FOOTER.unpack(
                    f.read(FOOTER.size)
                )
            )

            if marker != b"RIDX3END":
                raise ValueError("Invalid PACK index footer")
            if start < len(HEADER) or size <= 0 or start + size != total - FOOTER.size:
                raise ValueError("Invalid PACK index bounds")

            self._payload_limit = start

            return start, size, digest


    def _extract_index(self, destination, start, size, digest):
        """Copy an index and validate it against the PACK's footer."""
        with self.pack_path.open("rb") as f:

            f.seek(start)

            h = hashlib.sha256()

            with destination.open(
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

                out.flush()
                os.fsync(out.fileno())

            if h.digest() != digest:
                raise ValueError("PACK index SHA256 mismatch")


    @staticmethod
    def _valid_cached_index(path, size, digest):
        if not path.is_file() or path.stat().st_size != size:
            return False
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(block)
        return h.digest() == digest


    def _get_cached_index(self):
        start, size, digest = self._index_metadata()
        pack_stat = self.pack_path.stat()
        identity = "\0".join((
            str(self.pack_path.resolve()),
            str(pack_stat.st_dev),
            str(pack_stat.st_ino),
            str(pack_stat.st_size),
            str(pack_stat.st_mtime_ns),
            digest.hex(),
        ))
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()

        cache = Path(os.environ.get(
            "REIDFM_V3_INDEX_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "reidfm_v3_indices"),
        ))
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        index_path = cache / f"{key}.sqlite"
        lock_path = cache / f"{key}.lock"

        # A lock per PACK lets distinct domains initialize concurrently while
        # preventing several workers from extracting the same index at once.
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not self._valid_cached_index(index_path, size, digest):
                    # A prior aborted write or corrupt cache is never trusted.
                    with tempfile.NamedTemporaryFile(
                        prefix=f"{key}.", suffix=".tmp",
                        dir=cache, delete=False,
                    ) as temporary:
                        pending = Path(temporary.name)
                    try:
                        self._extract_index(pending, start, size, digest)
                        os.replace(pending, index_path)
                    finally:
                        pending.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        return index_path


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

        location = self._resolved_paths.get(requested)
        if location is not None:
            self._resolved_paths.move_to_end(requested)
        else:
            location = self._resolve_location(requested)
            self._resolved_paths[requested] = location
            if len(self._resolved_paths) > _RESOLVED_PATH_CACHE_SIZE:
                self._resolved_paths.popitem(last=False)

        # Even cache hits always read and re-hash the original image payload.
        return self._load(*location)


    def _candidates_for_basename(self, basename):
        cached = self._basename_candidates.get(basename)
        if cached is not None:
            self._basename_candidates.move_to_end(basename)
            return cached

        rows = self.db.execute(
            """
            SELECT leaf_path, context_path, offset, bytes, sha256
            FROM images WHERE basename=?
            """,
            (basename,),
        ).fetchall()
        prepared = tuple(
            (
                _reversed_casefold_parts(leaf),
                _reversed_casefold_parts(context),
                offset, size, digest,
            )
            for leaf, context, offset, size, digest in rows
        )

        # Index all distinct reversed suffixes. A suffix maps to exactly one
        # *row index* or None for ambiguous matches (even identical payload
        # offsets in two rows must remain ambiguous as in the original code).
        suffix_index = None
        if len(prepared) <= _SUFFIX_INDEX_CANDIDATE_LIMIT:
            suffix_index = {}
            for row_index, (leaf_parts, context_parts, *_unused) in enumerate(prepared):
                row_suffixes = {
                    parts[:depth]
                    for parts in (leaf_parts, context_parts)
                    for depth in range(1, len(parts) + 1)
                }
                for suffix in row_suffixes:
                    previous = suffix_index.get(suffix, _MISSING_SUFFIX)
                    if previous is _MISSING_SUFFIX:
                        suffix_index[suffix] = row_index
                    elif previous != row_index:
                        suffix_index[suffix] = None

            self._basename_candidates[basename] = (prepared, suffix_index)
            if len(self._basename_candidates) > _BASENAME_CACHE_SIZE:
                self._basename_candidates.popitem(last=False)
        return prepared, suffix_index


    def _resolve_location(self, requested):
        """Return (offset, size, sha256), preserving original tie behavior."""

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
                return tuple(rows[0])

            if len(rows) > 1:
                raise ValueError(
                    "Ambiguous image path: "
                    + requested
                )

        # Fall back to suffix matching.
        basename = requested.rsplit(
            "/", 1
        )[-1]

        matches, suffix_index = self._candidates_for_basename(basename)

        if not matches:
            raise FileNotFoundError(
                requested
            )

        requested_parts = _reversed_casefold_parts(requested)

        if suffix_index is not None:
            # Longest matching suffix wins. A suffix present in multiple rows
            # must raise the original ambiguity error, not pick a random row.
            for depth in range(len(requested_parts), 0, -1):
                match = suffix_index.get(requested_parts[:depth], _MISSING_SUFFIX)
                if match is _MISSING_SUFFIX:
                    continue
                if match is None:
                    raise ValueError("Ambiguous suffix mapping: " + requested)
                return matches[match][2:]
            raise ValueError("Ambiguous suffix mapping: " + requested)

        # Very large basename groups retain the bounded-memory legacy path.
        best = -1
        winner = None
        ties = 0

        for leaf_parts, context_parts, offset, size, digest in matches:
            score = max(
                _suffix_score_parts(requested_parts, leaf_parts),
                _suffix_score_parts(requested_parts, context_parts),
            )
            if score > best:
                best = score
                winner = (offset, size, digest)
                ties = 1
            elif score == best:
                ties += 1

        if ties != 1 or best < 1:
            raise ValueError(
                "Ambiguous suffix mapping: "
                + requested
            )

        return winner


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

        if hasattr(self, "_resolved_paths"):
            self._resolved_paths.clear()
        if hasattr(self, "_basename_candidates"):
            self._basename_candidates.clear()

        if hasattr(self, "f"):
            self.f.close()

        if hasattr(self, "db"):
            self.db.close()

        if self._tmp is not None:
            self._tmp.cleanup()


    def __enter__(self):
        return self


    def __exit__(self, *args):
        self.close()
