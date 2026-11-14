"""Deterministic, model-independent allocation of whole CERT account groups."""
from collections import Counter
import hashlib
import math

from datasets.cert_labels import LabelIndex, POLICY_ID

ALGORITHM = "cert_group_greedy_v1"
SPLITS = ("train", "validation", "test")


def allocate_groups(index: LabelIndex, fractions=(0.6, 0.2, 0.2), seed=42):
    if (len(fractions) != 3 or any(not math.isfinite(f) or f <= 0 for f in fractions)
            or not math.isclose(sum(fractions), 1)):
        raise ValueError("Expected three positive fractions summing to one")
    groups = {}
    for account, group in sorted(index.account_groups.items()):
        groups.setdefault(group, {"accounts": [], "incidents": {}, "days": set(), "positive": set()})["accounts"].append(account)
    for (account, day) in index.positive_days():
        group = groups[index.account_groups[account]]
        group["days"].add((account, day))
        group["positive"].add(account)
    for key, attrs in index.attributions.items():
        account = index.rows[key][2]
        group_id = index.account_groups[account]
        for attr in attrs:
            if index.account_groups[attr.incident_user] != group_id:
                raise ValueError("Incident crosses account groups")
            groups[group_id]["incidents"][attr.incident_id] = attr.scenario
    scenarios = sorted({s for g in groups.values() for s in g["incidents"].values()})
    support = Counter(s for g in groups.values() for s in set(g["incidents"].values()))
    if len(groups) < 3 or any(support[s] < 2 for s in scenarios):
        raise ValueError("Insufficient independent groups for train/test scenario support")
    required = {s: ({0, 2} if support[s] == 2 else {0, 1, 2}) for s in scenarios}
    vectors = {}
    for gid, g in groups.items():
        counts = Counter(g["incidents"].values())
        vectors[gid] = [len(g["accounts"]), len(g["positive"]), len(g["days"]),
                        *[counts[s] for s in scenarios]]
    totals = [sum(v[j] for v in vectors.values()) for j in range(3 + len(scenarios))]
    assigned = [[0] * len(totals) for _ in SPLITS]
    covered = [set() for _ in SPLITS]
    remaining = support.copy()
    allocation = {}
    def tie(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    # Rare scenarios first; large groups next. Seed only breaks stable ties.
    order = sorted(groups, key=lambda gid: (
        min((support[s] for s in groups[gid]["incidents"].values()), default=float("inf")),
        -len(groups[gid]["accounts"]), -len(groups[gid]["days"]), tie(gid)))
    for gid in order:
        present = set(groups[gid]["incidents"].values())
        remaining.subtract(present)
        choices = []
        for i, fraction in enumerate(fractions):
            # Reserve enough unassigned supporting groups for uncovered splits.
            if any(sum(s not in (covered[k] | (present if k == i else set()))
                       for k in required[s]) > remaining[s] for s in scenarios):
                continue
            delta = sum(((assigned[i][j] + v - fraction * totals[j]) ** 2
                         - (assigned[i][j] - fraction * totals[j]) ** 2)
                        / max(totals[j], 1) ** 2 for j, v in enumerate(vectors[gid]))
            choices.append((delta, tie(f"{gid}:{i}"), i))
        if not choices:
            raise ValueError("Greedy allocation cannot preserve scenario support; review protocol")
        i = min(choices)[2]
        allocation[gid] = SPLITS[i]
        covered[i].update(present)
        assigned[i] = [a + b for a, b in zip(assigned[i], vectors[gid])]
    summary = {}
    for split in SPLITS:
        selected = [g for gid, g in groups.items() if allocation[gid] == split]
        if not selected:
            raise ValueError("Empty split")
        summary[split] = {
            "groups": len(selected), "accounts": sum(len(g["accounts"]) for g in selected),
            "positive_accounts": sum(len(g["positive"]) for g in selected),
            "positive_account_days": sum(len(g["days"]) for g in selected),
            "incidents_by_scenario": dict(sorted(Counter(s for g in selected for s in g["incidents"].values()).items())),
        }
    return {"algorithm": ALGORITHM, "policy_id": POLICY_ID, "seed": seed,
            "fractions": dict(zip(SPLITS, fractions)), "summary": summary,
            "scenario_group_support": dict(sorted(support.items())),
            "required_scenario_splits": {s: [SPLITS[i] for i in sorted(v)] for s, v in required.items()},
            "accounts": {a: {"group": g, "split": allocation[g]} for a, g in sorted(index.account_groups.items())},
            "incidents": {incident: allocation[gid] for gid in sorted(groups)
                          for incident in sorted(groups[gid]["incidents"])},
            "answer_sha256": index.input_hashes,
            "scope": "account assignment only; full day population and raw event checks pending"}
