from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_paths import DATA_ROOT_ENV
from data_paths import ensure_scenario_directories
from data_paths import get_scenario_paths
from data_paths import resolve_data_root


class DataPathsTests(unittest.TestCase):
    def test_external_root_must_exist(self) -> None:
        missing = Path(tempfile.gettempdir()) / "argus-test-missing-data-root"
        with patch.dict(os.environ, {DATA_ROOT_ENV: str(missing)}, clear=False):
            with self.assertRaises(FileNotFoundError):
                resolve_data_root()

    def test_scenario_layout_is_created_under_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(
                os.environ,
                {DATA_ROOT_ENV: temporary_directory},
                clear=False,
            ):
                paths = get_scenario_paths("cert_r4_2")
                ensure_scenario_directories(paths)

                self.assertEqual(paths.root.parent, Path(temporary_directory).resolve())
                self.assertTrue(paths.raw.is_dir())
                self.assertTrue(paths.interim.is_dir())
                self.assertTrue(paths.processed.is_dir())
                self.assertTrue(paths.manifests.is_dir())

    def test_unknown_scenario_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_scenario_paths("unknown")


if __name__ == "__main__":
    unittest.main()
