#!/usr/bin/env python3
"""
adjudicate.py  (validation/)

Final analysis. Joins human gold (review_verdicts.csv) with LLM predictions
(preds.jsonl) and the clade gate (clade_map.csv) on gbifID, keeps reviewed=true.
Produces parallel scorecards for STUDENT-vs-human and TEACHER-vs-human.

Run from validation/:
    python adjudicate.py

Key design points:
- Human phenology was collapsed to a single boolean `repro_visible`. The LLM
  side is the disjunction any_repro = OR(flower, fruit, pollen_cone, seed_cone,
  sporulating, reproductive_unknown). This measures CATEGORY-level FP/FN only.
- The clade gate is SEPARATE and complementary: it catches botanically-impossible
  positives (e.g. pollen_cone on an angiosperm) that the category check can mask
  when a real repro structure is also present. Both are reported.
- foliage_type is NOT human-validated; reported as a clade SOFT-flag only.
- Per field, human values of uncertain / cannot_determine / blank are dropped
  from THAT field's denominator (counts reported).
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

ENUMS = ["type", "structures.foliage", "structures.stem"]
DIRECT_BOOLS = ["attached_photo",
                "refs.label", "refs.barcode", "refs.stamp", "refs.crc", "refs.scale_bar"]
REPRO_LLM = ["structures.phenology.flower", "structures.phenology.fruit",
             "structures.phenology.pollen_cone", "structures.phenology.seed_cone",
             "structures.phenology.sporulating",
             "structures.phenology.reproductive_unknown"]
SCORED = ENUMS + DIRECT_BOOLS + ["repro_visible"]          # 10 human fields
DROP = {"uncertain", "cannot_determine", ""}
# trait short-name -> dotted path, for the clade gate
GATE_PATH = {"flower": "structures.phenology.flower",
             "fruit": "structures.phenology.fruit",
             "pollen_cone": "structures.phenology.pollen_cone",
             "seed_cone": "structures.phenology.seed_cone",
             "sporulating": "structures.phenology.sporulating"}
POSSIBLE = {
    "angiosperm":  {"flower", "fruit"},
    "gymnosperm":  {"pollen_cone", "seed_cone"},
    "spore_plant": {"sporulating"},
    "unknown":     set(GATE_PATH),
}


def dotget(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def b2s(v):
    return "true" if v is True else "false" if v is False else str(v)


def llm_field(pred, field):
    """LLM string value for a scored human field (repro_visible = OR of 6)."""
    if field == "repro_visible":
        return "true" if any(dotget(pred, p) is True for p in REPRO_LLM) else "false"
    return b2s(dotget(pred, field))


# ---- metric helpers --------------------------------------------------------
def enum_acc(pairs):                       # pairs: (human, llm)
    n = len(pairs)
    return (sum(h == p for h, p in pairs), n)


def bool_counts(pairs):                     # positive class = "true"
    tp = fp = tn = fn = 0
    for h, p in pairs:
        ph, pp = h == "true", p == "true"
        tp += ph and pp; fp += (not ph) and pp
        tn += (not ph) and (not pp); fn += ph and (not pp)
    return tp, fp, tn, fn


def prf(tp, fp, tn, fn):
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    return prec, rec, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="review_verdicts.csv")
    ap.add_argument("--preds", default="preds.jsonl")
    ap.add_argument("--clade", default="clade_map.csv")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    preds = {}
    for l in Path(args.preds).read_text().splitlines():
        if l.strip():
            r = json.loads(l); preds[r["gbifID"]] = r

    clade = {}
    cp = Path(args.clade)
    if cp.exists():
        for row in csv.DictReader(cp.open()):
            clade[row["gbifID"]] = row.get("repro_group", "unknown")
    else:
        print(f"WARN: {args.clade} missing — clade gate skipped (run enrich_clade.py)")

    rows = [r for r in csv.DictReader(Path(args.verdicts).open())
            if r.get("reviewed") == "true"]
    rows = [r for r in rows if r["gbifID"] in preds]
    print(f"adjudicating {len(rows)} reviewed specimens with predictions\n")

    # collect (human, llm) pairs per field per source --------------------------
    pairs = {"student": defaultdict(list), "teacher": defaultdict(list)}
    drop_counts = defaultdict(int)
    joined = []                       # per-specimen side-by-side
    for r in rows:
        gid = r["gbifID"]; pr = preds[gid]
        jr = {"gbifID": gid, "taxon": pr["taxon"],
              "repro_group": clade.get(gid, "")}
        for fld in SCORED:
            hv = (r.get(fld) or "").strip()
            sv = llm_field(pr["student_pred"], fld)
            tv = llm_field(pr["teacher_gt"], fld)
            jr[f"H:{fld}"] = hv; jr[f"S:{fld}"] = sv; jr[f"T:{fld}"] = tv
            if hv in DROP:
                drop_counts[fld] += 1
                continue
            pairs["student"][fld].append((hv, sv))
            pairs["teacher"][fld].append((hv, tv))
        joined.append(jr)

    # ---- scorecards ---------------------------------------------------------
    metric_rows = []
    for src in ("student", "teacher"):
        print(f"================  {src.upper()} vs HUMAN  ================")
        print("ENUMS (accuracy):")
        for f in ENUMS:
            c, n = enum_acc(pairs[src][f])
            print(f"  {f:24s} {c}/{n} = {c/n:.3f}" if n else f"  {f:24s} (no data)")
            metric_rows.append([src, f, "enum_acc", c, n, f"{c/n:.4f}" if n else ""])
        print("BOOLEANS (tp/fp/tn/fn  P/R/acc):")
        for f in DIRECT_BOOLS + ["repro_visible"]:
            tp, fp, tn, fn = bool_counts(pairs[src][f])
            p, rc, ac = prf(tp, fp, tn, fn)
            print(f"  {f:34s} {tp:>3}/{fp:>3}/{tn:>3}/{fn:>3}   "
                  f"P={p:.2f} R={rc:.2f} A={ac:.2f}")
            metric_rows.append([src, f, "bool", f"{tp},{fp},{tn},{fn}",
                                tp + fp + tn + fn,
                                f"P={p:.3f};R={rc:.3f};A={ac:.3f}"])
        # overall exact-match over specimens with NO dropped field
        em_n = em_ok = 0
        for r in rows:
            if any((r.get(f) or "").strip() in DROP for f in SCORED):
                continue
            em_n += 1
            pr = preds[r["gbifID"]]["student_pred" if src == "student" else "teacher_gt"]
            if all((r.get(f) or "").strip() == llm_field(pr, f) for f in SCORED):
                em_ok += 1
        print(f"OVERALL exact-match (all {len(SCORED)} fields, clean specimens): "
              f"{em_ok}/{em_n} = {em_ok/em_n:.3f}\n" if em_n else "(none)\n")
        metric_rows.append([src, "OVERALL", "exact_match", em_ok, em_n,
                            f"{em_ok/em_n:.4f}" if em_n else ""])

    # ---- student-vs-teacher improvement breakdown ---------------------------
    print("============  STUDENT vs TEACHER (relative to human truth)  ============")
    print(f"{'field':34s} fixed  regress  both_ok  both_bad   (drop={'uncertain'})")
    imp_rows = []
    for f in SCORED:
        fixed = regress = both_ok = both_bad = 0
        for r in rows:
            hv = (r.get(f) or "").strip()
            if hv in DROP:
                continue
            pr = preds[r["gbifID"]]
            s_ok = llm_field(pr["student_pred"], f) == hv
            t_ok = llm_field(pr["teacher_gt"], f) == hv
            fixed += s_ok and not t_ok
            regress += t_ok and not s_ok
            both_ok += s_ok and t_ok
            both_bad += not s_ok and not t_ok
        print(f"{f:34s} {fixed:>5}  {regress:>7}  {both_ok:>7}  {both_bad:>8}")
        imp_rows.append([f, fixed, regress, both_ok, both_bad])

    # ---- clade gate (confirmed hallucinations) ------------------------------
    print("\n============  CLADE GATE — botanically-impossible positives  ============")
    print("(LLM predicted true for a trait impossible in that clade = hallucination)")
    gate_rows = []
    for src in ("student", "teacher"):
        key = "student_pred" if src == "student" else "teacher_gt"
        hard = defaultdict(list); onclade = defaultdict(int)
        for r in rows:
            gid = r["gbifID"]; grp = clade.get(gid)
            if not grp:
                continue
            pred = preds[gid][key]
            for trait, path in GATE_PATH.items():
                if dotget(pred, path) is True:
                    if trait in POSSIBLE.get(grp, set()):
                        onclade[trait] += 1
                    elif grp != "unknown":
                        hard[trait].append((gid, preds[gid]["taxon"], grp))
        n_hard = sum(len(v) for v in hard.values())
        print(f"\n{src.upper()}: {n_hard} confirmed-impossible positives")
        for trait in GATE_PATH:
            if hard[trait]:
                print(f"  {trait}: {len(hard[trait])} "
                      f"(on-clade plausible: {onclade.get(trait,0)})")
                for gid, tax, grp in hard[trait]:
                    print(f"      {gid}  {tax}  [{grp}]")
                    gate_rows.append([src, trait, gid, tax, grp])

    # ---- rare positives, loud & raw -----------------------------------------
    print("\n============  RARE POSITIVES (raw counts — n tiny, % unstable)  ============")
    for src in ("student", "teacher"):
        tp, fp, tn, fn = bool_counts(pairs[src]["attached_photo"])
        print(f"  attached_photo {src:8s}: tp={tp} fp={fp} fn={fn} tn={tn}")
    print("  cone/spore hallucinations: see CLADE GATE section above.")

    # ---- write outputs ------------------------------------------------------
    out = Path(args.outdir)
    with (out / "adjudication_results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "field", "metric", "value", "n", "detail"])
        w.writerows(metric_rows)
        w.writerow([])
        w.writerow(["improvement", "field", "fixed", "regress", "both_ok", "both_bad"])
        for ir in imp_rows:
            w.writerow(["improvement"] + ir)
        w.writerow([])
        w.writerow(["clade_gate", "source", "trait", "gbifID", "taxon", "group"])
        for gr in gate_rows:
            w.writerow(["clade_gate"] + gr)

    if joined:
        cols = ["gbifID", "taxon", "repro_group"] + \
               [f"{p}:{f}" for f in SCORED for p in ("H", "S", "T")]
        with (out / "joined_review.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(joined)

    print(f"\ndropped-from-denominator (human uncertain/cannot_determine/blank):")
    for f in SCORED:
        if drop_counts[f]:
            print(f"  {f}: {drop_counts[f]}")
    print(f"\nwrote {out/'adjudication_results.csv'} and {out/'joined_review.csv'}")


if __name__ == "__main__":
    main()
