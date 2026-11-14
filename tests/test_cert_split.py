import unittest
from datasets.cert_labels import Attribution, LabelIndex
from datasets.cert_split import allocate_groups


class CertSplitTests(unittest.TestCase):
    def fixture(self):
        groups = {f"U{i:03}": f"U{i:03}" for i in range(1000)}
        # Multi-account groups must remain indivisible, including background accounts.
        groups["U090"] = groups["U091"] = groups["U000"]
        rows, attrs = {}, {}
        for i in range(70):
            account = f"U{i:03}"
            scenario = "1" if i < 30 else "2" if i < 60 else "3"
            key = ("logon", str(i))
            rows[key] = (str(i), "01/01/2010 01:00:00", account, "PC", "Logon")
            attrs[key] = (Attribution(f"incident{i}", account, scenario),)
        return LabelIndex(rows, attrs, groups, {"fixture": "hash"})

    def test_deterministic_complete_disjoint_and_supported(self):
        index = self.fixture()
        result = allocate_groups(index)
        index.account_groups = dict(reversed(list(index.account_groups.items())))
        index.rows = dict(reversed(list(index.rows.items())))
        self.assertEqual(result, allocate_groups(index))
        self.assertEqual(set(result["accounts"]), set(index.account_groups))
        for gid in set(index.account_groups.values()):
            self.assertEqual(len({result["accounts"][a]["split"] for a, g in index.account_groups.items() if g == gid}), 1)
        for split, target in zip(("train", "validation", "test"), (600, 200, 200)):
            stats = result["summary"][split]
            self.assertLessEqual(abs(stats["accounts"] - target), 3)
            self.assertEqual(set(stats["incidents_by_scenario"]), {"1", "2", "3"})
        for key, attrs in index.attributions.items():
            for attr in attrs:
                self.assertEqual(result["incidents"][attr.incident_id], result["accounts"][index.rows[key][2]]["split"])
        self.assertNotEqual(result["accounts"], allocate_groups(index, seed=43)["accounts"])

    def test_reject_invalid_fractions_and_insufficient_support(self):
        for fractions in ((1, 0, 0), (.5, .2, .2), (float("nan"), .2, .2)):
            with self.assertRaises(ValueError):
                allocate_groups(self.fixture(), fractions)
        index = self.fixture()
        for key in list(index.rows):
            if int(key[1]) >= 61:
                del index.rows[key]
                del index.attributions[key]
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            allocate_groups(index)

    def test_reject_split_incident(self):
        index = self.fixture()
        index.attributions[("logon", "0")] = (Attribution("bad", "U999", "1"),)
        with self.assertRaisesRegex(ValueError, "crosses"):
            allocate_groups(index)

    def test_two_groups_preserve_train_test_and_report_validation_gap(self):
        index = self.fixture()
        for key in list(index.rows):
            if int(key[1]) >= 62:
                del index.rows[key]
                del index.attributions[key]
        result = allocate_groups(index)
        self.assertEqual(result["required_scenario_splits"]["3"], ["train", "test"])
        self.assertNotIn("3", result["summary"]["validation"]["incidents_by_scenario"])
        for split in ("train", "test"):
            self.assertEqual(result["summary"][split]["incidents_by_scenario"]["3"], 1)
