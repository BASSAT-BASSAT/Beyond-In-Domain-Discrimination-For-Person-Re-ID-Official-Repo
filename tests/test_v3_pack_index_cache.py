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


if __name__ == "__main__":
    unittest.main()
