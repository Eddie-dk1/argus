import csv
from pathlib import Path
import tempfile
import unittest

from datasets.cert_labels import build_label_index


class CertLabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.master = [
            ["dataset", "scenario", "details", "user", "start", "end"],
            ["4.2", "3", "a.csv", "U1", "01/01/2010 00:00:00", "01/03/2010 23:59:59"],
        ]
        self.logon = ("same", "01/02/2010 01:00:00", "U2", "P2", "Logon")
        self.email = ("same", "01/02/2010 02:00:00", "U2", "P2", "to", "", "", "from", "10", "0", "line1\nline2")
        self.write("insiders.csv", self.master)
        self.write("a.csv", [["logon", *self.logon], ["email", *self.email]])

    def write(self, name, rows):
        with (self.root / name).open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)

    def test_cross_account_days_exact_membership_and_unlisted_background(self):
        index = build_label_index(self.root, accounts={"U1", "U2", "U3"})
        self.assertEqual(index.event_label("email", self.email), 1)
        self.assertEqual(index.event_label("logon", self.logon), 1)
        unlisted = ("other", *self.logon[1:])
        self.assertEqual(index.event_label("logon", unlisted), 0)
        self.assertEqual(set(index.positive_days()), {("U2", "2010-01-02")})
        self.assertEqual(len(index.positive_days()[("U2", "2010-01-02")]), 2)
        self.assertEqual(index.attributions[("logon", "same")][0].incident_user, "U1")
        self.assertEqual(index.account_groups["U1"], index.account_groups["U2"])
        self.assertNotEqual(index.account_groups["U1"], index.account_groups["U3"])
        with self.assertRaisesRegex(ValueError, "complete raw row"):
            index.event_label("logon", (*self.logon[:-1], "Logoff"))

    def test_transitive_groups_and_order_invariance(self):
        self.master.append(["4.2", "3", "b.csv", "U2", "01/01/2010 00:00:00", "01/03/2010 23:59:59"])
        self.write("insiders.csv", self.master)
        self.write("b.csv", [["logon", "L2", self.logon[1], "U3", "P3", "Logon"]])
        first = build_label_index(self.root, accounts={"U1", "U2", "U3", "U4"})
        self.write("insiders.csv", [self.master[0], *reversed(self.master[1:])])
        second = build_label_index(self.root, accounts={"U1", "U2", "U3", "U4"})
        self.assertEqual(first.account_groups, second.account_groups)
        self.assertEqual(len(set(first.account_groups.values())), 2)
        self.assertNotEqual(first.input_hashes, second.input_hashes)

    def test_duplicate_observables_deduplicated_without_losing_attribution(self):
        self.master.append(["4.2", "3", "b.csv", "U3", "01/01/2010 00:00:00", "01/03/2010 23:59:59"])
        self.write("insiders.csv", self.master)
        self.write("b.csv", [["logon", *self.logon], ["logon", *self.logon]])
        index = build_label_index(self.root)
        self.assertEqual(len(index.rows), 2)
        self.assertEqual(len(index.attributions[("logon", "same")]), 2)

    def test_reject_unknown_account_and_outside_interval(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*account"):
            build_label_index(self.root, accounts={"U1"})
        self.write("a.csv", [["logon", "L", "01/04/2010 00:00:00", "U1", "P1", "Logon"]])
        with self.assertRaisesRegex(ValueError, "outside incident interval"):
            build_label_index(self.root)

    def test_reject_conflicting_source_id_and_missing_detail(self):
        self.write("a.csv", [["logon", *self.logon], ["logon", *self.logon[:-1], "Logoff"]])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            build_label_index(self.root)
        (self.root / "a.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/ambiguous"):
            build_label_index(self.root)


if __name__ == "__main__":
    unittest.main()
