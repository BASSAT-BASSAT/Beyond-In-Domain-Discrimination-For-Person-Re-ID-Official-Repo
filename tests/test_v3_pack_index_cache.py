"""REIDPACK3 index-cache parity, integrity and concurrency tests (stdlib only)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from reidfm.data.v3_pack_reader import FOOTER, HEADER, V3PackReader, fcntl
from scripts.diagnose_v3_pack_io import run as diagnose_pack_io


PAYLOAD = b"synthetic image bytes for an index reader test"
IMAGE_PATH = "train/person001/frame00001.jpg"


def _write_pack(root: Path, corrupt_index=False, corrupt_payload=False):
    source = root / "source_index.sqlite"
    conn = sqlite3.connect(source)
    try:
        conn.execute(
            "CREATE TABLE images (physical_path TEXT, context_path TEXT, "
            "leaf_path TEXT, basename TEXT, offset INTEGER, bytes INTEGER, sha256 TEXT)"
        )
        conn.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
            (IMAGE_PATH, IMAGE_PATH, IMAGE_PATH, "frame00001.jpg",
             len(HEADER), len(PAYLOAD), sha256(PAYLOAD).hexdigest()),
        )
        conn.commit()
    finally:
        conn.close()

    embedded = source.read_bytes()
    start = len(HEADER) + len(PAYLOAD)
    digest = sha256(embedded).digest()
    if corrupt_index:
        digest = b"\x00" * 32
    image = bytes([PAYLOAD[0] ^ 1]) + PAYLOAD[1:] if corrupt_payload else PAYLOAD
    (root / "Sample.pack").write_bytes(
        HEADER + image + embedded
        + FOOTER.pack(b"RIDX3END", start, len(embedded), digest)
    )


def _worker_read(pack_dir):
    with V3PackReader(pack_dir, "Sample") as reader:
        return str(reader._index_path), reader.read_bytes(IMAGE_PATH)


def _write_duplicate_basename_pack(root: Path, n: int = 715):
    """PRID-like fixture: hundreds of indexed rows share one filename."""
    source = root / "source_index.sqlite"
    conn = sqlite3.connect(source)
    try:
        conn.execute(
            "CREATE TABLE images (physical_path TEXT, context_path TEXT, "
            "leaf_path TEXT, basename TEXT, offset INTEGER, bytes INTEGER, sha256 TEXT)"
        )
        chunks = []
        cursor = len(HEADER)
        for i in range(n):
            payload = f"payload-for-identity-{i:04d}".encode()
            chunks.append(payload)
            conn.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"archive/raw_{i:04d}/frame.jpg",
                    f"dataset/root/person_{i:04d}/frame.jpg",
                    f"raw_{i:04d}/frame.jpg",
                    "frame.jpg", cursor, len(payload), sha256(payload).hexdigest(),
                ),
            )
            cursor += len(payload)
        conn.execute("CREATE INDEX images_basename_idx ON images(basename)")
        conn.commit()
    finally:
        conn.close()

    embedded = source.read_bytes()
    (root / "PRID2011.pack").write_bytes(
        HEADER + b"".join(chunks) + embedded
        + FOOTER.pack(b"RIDX3END", cursor, len(embedded), sha256(embedded).digest())
    )


class V3IndexCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v3_pack_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environ = patch.dict(os.environ, {
            "REIDFM_V3_INDEX_CACHE_DIR": str(self.root / "cache"),
            "REIDFM_V3_INDEX_CACHE": "1",
        })
        self.environ.start()
        self.addCleanup(self.environ.stop)

    @unittest.skipIf(fcntl is None, "shared cache uses Linux flock")
    def test_cache_reused_across_independent_readers(self):
        _write_pack(self.root)
        with V3PackReader(self.root, "Sample") as first:
            cached = first._index_path
            self.assertEqual(first.read_bytes(IMAGE_PATH), PAYLOAD)
            self.assertEqual(first.read_bytes(IMAGE_PATH.replace("/", "\\")), PAYLOAD)
        with V3PackReader(self.root, "Sample") as second:
            self.assertEqual(second._index_path, cached)
            self.assertEqual(second.read_bytes(IMAGE_PATH), PAYLOAD)
        self.assertTrue(cached.is_file())
        self.assertEqual(len(list((self.root / "cache").glob("*.sqlite"))), 1)

    @unittest.skipIf(fcntl is None, "shared cache uses Linux flock")
    def test_corrupted_cached_index_is_replaced(self):
        _write_pack(self.root)
        with V3PackReader(self.root, "Sample") as reader:
            cached = reader._index_path
        cached.write_bytes(b"bad cache")
        with V3PackReader(self.root, "Sample") as reader:
            self.assertEqual(reader.read_bytes(IMAGE_PATH), PAYLOAD)
            self.assertEqual(reader._index_path, cached)

    @unittest.skipIf(fcntl is None, "shared cache uses Linux flock")
    def test_simultaneous_workers_share_one_index(self):
        _write_pack(self.root)
        with ProcessPoolExecutor(max_workers=4,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            readings = list(pool.map(_worker_read, [str(self.root)] * 4))
        self.assertEqual(len({path for path, _ in readings}), 1)
        self.assertTrue(all(data == PAYLOAD for _, data in readings))
        self.assertEqual(len(list((self.root / "cache").glob("*.sqlite"))), 1)

    def test_private_index_fallback_and_payload_integrity(self):
        _write_pack(self.root, corrupt_payload=True)
        os.environ["REIDFM_V3_INDEX_CACHE"] = "0"
        with V3PackReader(self.root, "Sample") as reader:
            with self.assertRaisesRegex(ValueError, "Image SHA256 mismatch"):
                reader.read_bytes(IMAGE_PATH)
        self.assertFalse(reader._index_path.exists())

    def test_cached_and_private_readers_return_identical_bytes(self):
        _write_pack(self.root)
        with V3PackReader(self.root, "Sample") as cached:
            cached_bytes = cached.read_bytes(IMAGE_PATH)
        os.environ["REIDFM_V3_INDEX_CACHE"] = "0"
        with V3PackReader(self.root, "Sample") as private:
            private_bytes = private.read_bytes(IMAGE_PATH)
        self.assertEqual(cached_bytes, private_bytes)
        self.assertEqual(private_bytes, PAYLOAD)

    def test_corrupted_pack_index_never_cached(self):
        _write_pack(self.root, corrupt_index=True)
        with self.assertRaisesRegex(ValueError, "PACK index SHA256 mismatch"):
            V3PackReader(self.root, "Sample")
        if fcntl is not None:
            self.assertFalse(list((self.root / "cache").glob("*.sqlite")))

    def test_io_diagnostic_reports_paths_without_training(self):
        _write_pack(self.root)
        manifest = self.root / "train_manifest.csv"
        manifest.write_text(
            "dataset_root_id,relative_path\nSample," + IMAGE_PATH + "\n",
            encoding="utf-8",
        )
        report = diagnose_pack_io(manifest, self.root, samples_per_domain=1)
        self.assertEqual(report["manifest_rows"], 1)
        self.assertEqual(report["domains"]["Sample"]["samples"], 1)
        self.assertIn("physical_path", report["domains"]["Sample"]["query_plans"])

    def test_prid_style_suffix_resolution_parity_and_cache(self):
        _write_duplicate_basename_pack(self.root)
        with V3PackReader(self.root, "PRID2011") as reader:
            paths = [
                f"root/person_{i:04d}/frame.jpg"
                for i in (0, 1, 77, 500, 714)
            ]
            for i, requested in zip((0, 1, 77, 500, 714), paths):
                expected = f"payload-for-identity-{i:04d}".encode()
                self.assertEqual(reader.read_bytes(requested), expected)
                self.assertEqual(reader.read_bytes(requested), expected)

            self.assertEqual(len(reader._basename_candidates), 1)
            candidates, suffix_index = reader._basename_candidates["frame.jpg"]
            self.assertEqual(len(candidates), 715)
            self.assertTrue(suffix_index)
            self.assertEqual(len(reader._resolved_paths), len(paths))

            # Exact matching still takes priority over suffix candidates.
            self.assertEqual(
                reader.read_bytes("archive/raw_0001/frame.jpg"),
                b"payload-for-identity-0001",
            )
            # Tied best-score candidates must never be resolved arbitrarily.
            with self.assertRaisesRegex(ValueError, "Ambiguous suffix mapping"):
                reader.read_bytes("frame.jpg")
            with self.assertRaises(FileNotFoundError):
                reader.read_bytes("unrelated.jpg")
            with self.assertRaisesRegex(ValueError, "Unsafe relative image path"):
                reader.read_bytes("../frame.jpg")

    def test_large_basename_group_preserves_legacy_fallback(self):
        _write_duplicate_basename_pack(self.root, n=2050)
        with V3PackReader(self.root, "PRID2011") as reader:
            self.assertEqual(
                reader.read_bytes("root/person_2049/frame.jpg"),
                b"payload-for-identity-2049",
            )
            self.assertFalse(reader._basename_candidates)  # size-limited cache
            with self.assertRaisesRegex(ValueError, "Ambiguous suffix mapping"):
                reader.read_bytes("frame.jpg")


if __name__ == "__main__":
    unittest.main()
