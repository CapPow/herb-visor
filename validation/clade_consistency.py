#!/usr/bin/env python3
"""
clade_consistency.py  (validation/)

Label-free reliability check over the FULL test set. For every prediction,
asks: did the model assert a reproductive trait that is botanically IMPOSSIBLE
for that specimen's clade (per the GBIF backbone)? Needs no human annotation,
so it scales to the entire corpus.

Run from validation/:
    python clade_consistency.py [--preds preds.jsonl]
                                                                   [--clade clade_map.csv]

Reports, per source (student / teacher):
  - per-trait IMPOSSIBLE RATE = P(trait predicted true | trait impossible here),
    with the gateable denominator shown (so a 0% is read in context)
  - specimen-level rate (>=1 impossible positive)
  - on-clade positive rates (context: how often the trait is used where it IS
    possible; correctness of these is NOT adjudicated by taxonomy)
Outputs outputs/impossible_report.csv.
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

POSSIBLE = {
    "angiosperm":  {"flower", "fruit"},
    "gymnosperm":  {"pollen_cone", "seed_cone"},
    "spore_plant": {"sporulating"},
    "unknown":     {"flower", "fruit", "pollen_cone", "seed_cone", "sporulating"},
}
TRAIT_PATH = {"flower": "structures.phenology.flower",
              "fruit": "structures.phenology.fruit",
              "pollen_cone": "structures.phenology.pollen_cone",
              "seed_cone": "structures.phenology.seed_cone",
              "sporulating": "structures.phenology.sporulating"}
SOURCES = {"student": "student_pred", "teacher": "teacher_gt"}


def dotget(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="preds.jsonl")
    ap.add_argument("--clade", default="clade_map.csv")
    ap.add_argument("--out", default="outputs/impossible_report.csv")
    ap.add_argument("--list-violations", type=int, default=20,
                    help="max example violations to print per source")
    args = ap.parse_args()

    preds = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]
    clade = {}
    cp = Path(args.clade)
    if not cp.exists():
        raise SystemExit(f"{args.clade} missing — run enrich_clade.py on the full preds first")
    for row in csv.DictReader(cp.open()):
        clade[row["gbifID"]] = row.get("repro_group", "unknown")

    # coverage
    have = [p for p in preds if p["gbifID"] in clade]
    missing = len(preds) - len(have)
    dist = defaultdict(int)
    for p in have:
        dist[clade[p["gbifID"]]] += 1

    print(f"predictions: {len(preds)}   clade-resolved: {len(have)}   "
          f"unresolved: {missing}")
    if missing:
        print(f"  !! {missing} predictions have no clade row — re-run enrich_clade.py "
              f"on the FULL preds.jsonl so clade_map.csv covers every gbifID.")
    print("clade distribution (resolved):")
    for g in ("angiosperm", "gymnosperm", "spore_plant", "unknown"):
        print(f"  {g:12s} {dist.get(g,0)}")
    # gateable opportunities per trait (clade-determined, source-independent)
    gateable = {t: 0 for t in TRAIT_PATH}
    onclade_opp = {t: 0 for t in TRAIT_PATH}
    for p in have:
        grp = clade[p["gbifID"]]
        if grp == "unknown":
            continue
        for t in TRAIT_PATH:
            if t in POSSIBLE[grp]:
                onclade_opp[t] += 1
            else:
                gateable[t] += 1

    out_rows = []
    for src, key in SOURCES.items():
        viol = defaultdict(int)         # impossible positives
        onclade_pos = defaultdict(int)  # positives where trait IS possible
        spec_violators = set()
        examples = []
        for p in have:
            grp = clade[p["gbifID"]]
            if grp == "unknown":
                continue
            pred = p[key]
            for t, path in TRAIT_PATH.items():
                if dotget(pred, path) is True:
                    if t in POSSIBLE[grp]:
                        onclade_pos[t] += 1
                    else:
                        viol[t] += 1
                        spec_violators.add(p["gbifID"])
                        if len(examples) < args.list_violations:
                            examples.append((p["gbifID"], p["taxon"], t, grp))

        n_resolved = sum(1 for p in have if clade[p["gbifID"]] != "unknown")
        total_viol = sum(viol.values())
        print(f"\n================  {src.upper()}  ================")
        print(f"{'trait':14s} {'impossible':>10} {'gateable_n':>11} {'rate':>8}   "
              f"{'on-clade_pos':>12} {'on-clade_n':>11}")
        for t in TRAIT_PATH:
            r = viol[t] / gateable[t] if gateable[t] else float("nan")
            print(f"{t:14s} {viol[t]:>10} {gateable[t]:>11} {r:>8.4f}   "
                  f"{onclade_pos[t]:>12} {onclade_opp[t]:>11}")
            out_rows.append([src, t, viol[t], gateable[t],
                             f"{r:.5f}" if gateable[t] else "",
                             onclade_pos[t], onclade_opp[t]])
        spec_rate = len(spec_violators) / n_resolved if n_resolved else float("nan")
        print(f"specimen-level impossible rate: {len(spec_violators)}/{n_resolved} "
              f"= {spec_rate:.4f}   (total impossible positives: {total_viol})")
        out_rows.append([src, "SPECIMEN_LEVEL", len(spec_violators), n_resolved,
                         f"{spec_rate:.5f}" if n_resolved else "", "", ""])
        if examples:
            print(f"examples (up to {args.list_violations}):")
            for gid, tax, t, grp in examples:
                print(f"   {gid}  {tax}  predicted {t}  [{grp}]")

    with Path(args.out).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "trait", "impossible", "gateable_n", "impossible_rate",
                    "onclade_positives", "onclade_n"])
        w.writerows(out_rows)
    print(f"\nwrote {args.out}")
    print("\nNote: 'impossible' positives are confirmed hallucinations (taxonomy "
          "forbids them). On-clade positives are NOT validated here — taxonomy "
          "can confirm a trait is impossible, not that a present one is correct.")


if __name__ == "__main__":
    main()
