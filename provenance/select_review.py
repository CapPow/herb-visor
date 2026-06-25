#!/usr/bin/env python3
"""
select_review.py  (pkg_v2/human_gt/)
Build a BLIND human-review manifest from preds.jsonl.

Run from pkg_v2/:
    python human_gt/select_review.py [--target 100] ...

Strata fill order (dedup: a specimen counts once, earliest stratum wins):
  A) rare-positive census  — EVERY specimen where student OR teacher is true
                             for any of {reproductive_unknown, pollen_cone,
                             seed_cone, attached_photo}. NEVER truncated.
  B) disagreement sample   — up to --b-per-field each from
                             {flower, fruit, stem, foliage_type, stamp}.
  C) agreement control     — up to --c-size random exact_match specimens.
  top-up) control_random   — random remaining specimens to reach --target.

Cap = --target (HARD), except A may overflow it (reported). When A+B+C would
exceed the cap, C yields slots first, then B (B has priority over C).

Output: human_gt/review_manifest.json  — predictions STRIPPED. Each entry:
  {gbifID, taxon, image_path, reason_tags, fields:{field: [allowed values]}}
reason_tags are for OUR analysis only; the GUI must not surface field names
or use them to order presentation. Manifest is emitted in random order.
"""
import argparse, json, random, shutil, sys
from collections import defaultdict
from pathlib import Path

# --- schema -----------------------------------------------------------------
ENUM_VALUES = {
    "type": ["PH", "PI"],
    "structures.foliage": ["present", "absent"],
    "structures.foliage_type": ["leaf", "needle", "scale", "frond",
                                "spine", "mixed", "unknown", "none"],
    "structures.stem": ["woody", "herbaceous", "unknown"],
}
BOOL_FIELDS = [
    "attached_photo",
    "structures.phenology.flower", "structures.phenology.fruit",
    "structures.phenology.pollen_cone", "structures.phenology.seed_cone",
    "structures.phenology.sporulating", "structures.phenology.reproductive_unknown",
    "refs.label", "refs.barcode", "refs.stamp", "refs.crc", "refs.scale_bar",
]
ALL_FIELDS = list(ENUM_VALUES) + BOOL_FIELDS                      # 16
# "uncertain" is model-emitted -> offered in manifest. (GUI adds cannot_determine.)
ALLOWED = {f: v + ["uncertain"] for f, v in ENUM_VALUES.items()}
ALLOWED.update({f: ["true", "false", "uncertain"] for f in BOOL_FIELDS})

RARE = ["structures.phenology.reproductive_unknown",
        "structures.phenology.pollen_cone",
        "structures.phenology.seed_cone",
        "attached_photo"]
DISAGREE = ["structures.phenology.flower", "structures.phenology.fruit",
            "structures.stem", "structures.foliage_type", "refs.stamp"]


def dotget(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="human_gt/preds.jsonl")
    ap.add_argument("--out", default="human_gt/review_manifest.json")
    ap.add_argument("--target", type=int, default=100, help="hard cap (A may overflow)")
    ap.add_argument("--b-per-field", type=int, default=13)
    ap.add_argument("--c-size", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-images", action="store_true",
                    help="copy selected images into human_gt/images/")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]
    by_id = {r["gbifID"]: r for r in rows}
    if len(by_id) != len(rows):
        print(f"WARN: {len(rows)-len(by_id)} duplicate gbifID(s) in preds.jsonl", file=sys.stderr)

    selected = {}          # gbifID -> {strata:set, fields:set}
    def tag(gid, stratum, field=None):
        e = selected.setdefault(gid, {"strata": set(), "fields": set()})
        e["strata"].add(stratum)
        if field:
            e["fields"].add(field)

    def true_either(r, f):
        return dotget(r["student_pred"], f) is True or dotget(r["teacher_gt"], f) is True

    # --- A: rare-positive census (never truncated) --------------------------
    rare_capture = {f: 0 for f in RARE}
    for r in rows:
        hit = [f for f in RARE if true_either(r, f)]
        for f in hit:
            rare_capture[f] += 1
        if hit:
            tag(r["gbifID"], "rare_positive")
            for f in hit:
                selected[r["gbifID"]]["fields"].add(f)
    n_A = len(selected)
    cap = max(args.target, n_A)          # A may overflow the target
    overflow = n_A > args.target

    # --- B: disagreement (B priority over C) --------------------------------
    for f in DISAGREE:
        if len(selected) >= cap:
            break
        cands = [r["gbifID"] for r in rows
                 if r["agree"].get(f) is False and r["gbifID"] not in selected]
        rng.shuffle(cands)
        for gid in cands[:args.b_per_field]:
            if len(selected) >= cap:
                break
            tag(gid, "disagree", f)

    # --- C: agreement control -----------------------------------------------
    c_cands = [r["gbifID"] for r in rows
               if r.get("exact_match") and r["gbifID"] not in selected]
    rng.shuffle(c_cands)
    for gid in c_cands[:args.c_size]:
        if len(selected) >= cap:
            break
        tag(gid, "agree_control")

    # --- top-up: control_random ---------------------------------------------
    rest = [r["gbifID"] for r in rows if r["gbifID"] not in selected]
    rng.shuffle(rest)
    for gid in rest:
        if len(selected) >= cap:
            break
        tag(gid, "control_random")

    # --- emit (predictions stripped, random order) --------------------------
    manifest = []
    for gid in selected:
        r = by_id[gid]
        e = selected[gid]
        manifest.append({
            "gbifID": gid,
            "taxon": r["taxon"],
            "image_path": r["image_path"],
            "reason_tags": sorted(e["strata"]) + sorted(e["fields"]),  # analysis-only
            "fields": {f: ALLOWED[f] for f in ALL_FIELDS},
        })
    rng.shuffle(manifest)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    if args.copy_images:
        dst = Path("human_gt/images"); dst.mkdir(parents=True, exist_ok=True)
        for m in manifest:
            src = Path(m["image_path"])
            if src.exists():
                shutil.copy2(src, dst / src.name)

    # --- coverage summary ----------------------------------------------------
    strata_count = defaultdict(int)
    for e in selected.values():
        for s in e["strata"]:
            strata_count[s] += 1
    print(f"manifest: {len(manifest)} specimens  (target {args.target}, cap {cap})")
    if overflow:
        print(f"  !! A-census OVERFLOW: {n_A} rare specimens > target {args.target}; kept all of A")
    print("strata (a specimen may carry >1 tag; counted once in manifest):")
    for s in ("rare_positive", "disagree", "agree_control", "control_random"):
        print(f"  {s:15s} {strata_count[s]}")
    print("rare-positive capture (student OR teacher true):")
    for f in RARE:
        print(f"  {f:42s} {rare_capture[f]}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
