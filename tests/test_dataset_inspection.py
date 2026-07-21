from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from datasets.inspection import inspect_scenario
from datasets.loading import iter_scenario_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = PROJECT_ROOT / "tests" / "fixtures"


class DatasetInspectionTests(unittest.TestCase):
    def inspect_fixture(self, scenario: str) -> dict:
        return inspect_scenario(
            scenario,
            FIXTURES_ROOT / scenario / "raw",
            sample_rows=100,
        )

    def test_all_scenario_fixtures_match_contracts(self) -> None:
        for scenario in (
            "synthetic",
            "cert_r4_2",
            "cicids2017",
            "cse_cic_ids2018",
        ):
            with self.subTest(scenario=scenario):
                report = self.inspect_fixture(scenario)
                self.assertTrue(report["valid_schema"], report)

    def test_network_aliases_resolve_to_canonical_names(self) -> None:
        report = self.inspect_fixture("cse_cic_ids2018")
        canonical = report["sources"][0]["files"][0]["canonical_columns"]

        self.assertEqual(canonical["total_fwd_packets"], "Tot Fwd Pkts")
        self.assertEqual(canonical["flow_bytes_per_second"], "Flow Byts/s")

    def test_sample_quality_signals_are_reported(self) -> None:
        cse_report = self.inspect_fixture("cse_cic_ids2018")
        cse_file = cse_report["sources"][0]["files"][0]
        self.assertEqual(cse_file["sample_repeated_header_rows"], 1)
        self.assertEqual(cse_file["sample_non_finite_tokens"], 1)

        cic_report = self.inspect_fixture("cicids2017")
        cic_file = cic_report["sources"][0]["files"][0]
        self.assertEqual(cic_file["sample_duplicate_rows"], 1)

    def test_missing_required_column_invalidates_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            (raw_dir / "synthetic_logs.csv").write_text(
                "timestamp,user_id,label\n2025-01-01,user_001,0\n",
                encoding="utf-8",
            )
            report = inspect_scenario("synthetic", raw_dir, sample_rows=10)

        self.assertFalse(report["valid_schema"])
        missing = report["sources"][0]["files"][0]["missing_required_columns"]
        self.assertIn("source_ip", missing)

    def test_missing_source_group_invalidates_cert_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            (raw_dir / "logon.csv").write_text(
                "id,date,user,pc,activity\nL1,01/01/2010,U1,PC1,Logon\n",
                encoding="utf-8",
            )
            report = inspect_scenario("cert_r4_2", raw_dir, sample_rows=10)

        self.assertFalse(report["valid_schema"])
        self.assertIn("email", report["missing_source_groups"])

    def test_chunked_loader_adds_provenance_and_canonical_columns(self) -> None:
        raw_dir = FIXTURES_ROOT / "cse_cic_ids2018" / "raw"
        chunks = list(
            iter_scenario_chunks(
                "cse_cic_ids2018",
                raw_dir,
                chunk_size=1,
            )
        )

        self.assertEqual(len(chunks), 2)
        self.assertIn("total_fwd_packets", chunks[0].columns)
        self.assertIn("flow_bytes_per_second", chunks[0].columns)
        self.assertEqual(chunks[0]["__source_row"].tolist(), [0])
        self.assertEqual(chunks[1]["__source_row"].tolist(), [1])
        self.assertEqual(chunks[0]["__source_group"].iloc[0], "flows")


if __name__ == "__main__":
    unittest.main()
