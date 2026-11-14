from pathlib import Path
import tempfile
import unittest

from audit_cert import audit, count_duplicate_ids, load_answers


class CertAuditTests(unittest.TestCase):
    def test_full_audit_reports_duplicates_without_changing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, answers = root / "raw", root / "answers"
            raw.mkdir()
            answers.mkdir()
            (raw / "LDAP").mkdir()
            (raw / "LDAP/2010-01.csv").write_text("user_id\nU1\n")
            (raw / "psychometric.csv").write_text("user_id\nU1\n")
            common = "id,date,user,pc"
            date = "01/01/2010 01:00:00"
            sources = {
                "logon": (common + ",activity", f"L1,{date},U1,P1,Logon"),
                "device": (common + ",activity", f"D1,{date},U1,P1,Connect"),
                "file": (common + ",filename,content", f"F1,{date},U1,P1,a.txt,example"),
                "http": (common + ",url,content", f"H1,{date},U1,P1,https://example.test,example"),
                "email": (common + ",to,from,size,attachments", f"E1,{date},U1,P1,to,from,inf,0"),
            }
            for name, (header, row) in sources.items():
                (raw / f"{name}.csv").write_text(header + "\n" + row + "\n")
            with (raw / "logon.csv").open("a") as stream:
                stream.write(sources["logon"][1] + "\n")
            before = (raw / "logon.csv").read_bytes()
            (answers / "insiders.csv").write_text(
                "dataset,scenario,details,user,start,end\n"
                "4.2,1,incident.csv,U1,01/01/2010 00:00:00,01/02/2010 00:00:00\n")
            (answers / "incident.csv").write_text("logon," + sources["logon"][1] + "\n")
            report = audit(raw, answers, root / "audit.json", root, chunk_size=1)
            self.assertEqual(report["total_events"], 6)
            self.assertEqual(report["sources"]["logon"]["duplicate_ids_within_source"], 1)
            self.assertEqual(report["sources"]["logon"]["unmatched_observables"], 0)
            self.assertEqual(report["sources"]["email"]["invalid_numeric_cells"], 1)
            self.assertEqual((raw / "logon.csv").read_bytes(), before)
            self.assertEqual(list(root.glob("cert-audit-*")), [])

    def test_variable_width_answers_and_context_users(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "insiders.csv").write_text(
                "dataset,scenario,details,user,start,end\n"
                "4.2,3,incident.csv,U1,01/01/2010 00:00:00,01/02/2010 00:00:00\n"
                "6.2,1,not-downloaded.csv,U2,01/01/2010 00:00:00,01/02/2010 00:00:00\n")
            (root / "incident.csv").write_text(
                "logon,shared,01/01/2010 01:00:00,U1,P1,Logon\n"
                "email,shared,01/01/2010 02:00:00,U2,P2,to,cc,bcc,from,12,0,content\n")
            incidents, expected = load_answers(root)
            self.assertEqual(len(incidents), 1)
            self.assertEqual(incidents[0]["other_user_observable_rows"], 1)
            self.assertEqual(incidents[0]["outside_master_interval"], 0)
            self.assertEqual(len(expected["logon"]["shared"]), 1)
            self.assertEqual(len(expected["email"]["shared"]), 1)

    def test_external_duplicate_sort_is_exact_and_cleans_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "ids.txt"
            path.write_text("C\nA\nB\nA\nA\n")
            self.assertEqual(count_duplicate_ids(path, root), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_incident_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "insiders.csv").write_text(
                "dataset,scenario,details,user,start,end\n"
                "4.2,1,missing.csv,U1,01/01/2010 00:00:00,01/02/2010 00:00:00\n")
            with self.assertRaisesRegex(ValueError, "Missing/ambiguous"):
                load_answers(root)
