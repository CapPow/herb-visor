#!/usr/bin/env python3
"""
migrate_verdicts.py  (pkg_v2/human_gt/)

One-shot, non-destructive patch of review_verdicts.csv to the post-fork schema.

Run from pkg_v2/:
    python human_gt/migrate_verdicts.py [--csv human_gt/review_verdicts.csv]

What it does:
  - Backs up the source CSV to review_verdicts.<timestamp>.bak (never overwrites).
  - DROPS columns: structures.foliage_type and all 6 individual phenology fields
    (flower/fruit/pollen_cone/seed_cone/sporulating/reproductive_unknown).
  - BLANKS structures.foliage  (you are re-annotating leaf_visible).
  - ADDS repro_visible (blank — to be filled in the GUI).
  - KEEPS untouched: refs.*, type, structures.stem, attached_photo, notes,
    review_seconds, gbifID.
  - RECOMPUTES `reviewed` against the new required field set.

Idempotent: safe to re-run (already-migrated files just get re-normalized).
Source is read, backed up, then atomically replaced with the new-schema file.
"""
import argparse, csv, shutil, sys, time
from pathlib import Path

KEEP = [
    "type", "structures.foliage", "structures.stem", "attached_photo",
    "refs.label", "refs.barcode", "refs.stamp", "refs.crc", "refs.scale_bar",
    "repro_visible",
]
BLANK_ON_MIGRATE = {"structures.foliage"}          # re-annotate
NEW_COLUMNS = ["gbifID", "reviewed"] + KEEP + ["notes", "review_seconds"]
DROPPED = [
    "structures.foliage_type",
    "structures.phenology.flower", "structures.phenology.fruit",
    "structures.phenology.pollen_cone", "structures.phenology.seed_cone",
    "structures.phenology.sporulating", "structures.phenology.reproductive_unknown",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="human_gt/review_verdicts.csv")
    args = ap.parse_args()

    src = Path(args.csv)
    if not src.exists():
        sys.exit(f"not found: {src}")

    with src.open() as f:
        rows = list(csv.DictReader(f))
    old_cols = rows[0].keys() if rows else []

    bak = src.with_name(f"{src.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(src, bak)

    dropped_present = [c for c in DROPPED if c in old_cols]
    out_rows = []
    fill = {c: 0 for c in KEEP}
    for r in rows:
        nr = {"gbifID": r.get("gbifID", "")}
        for c in KEEP:
            if c in BLANK_ON_MIGRATE:
                nr[c] = ""                          # cleared for re-annotation
            else:
                nr[c] = r.get(c, "")                # repro_visible -> "" if absent
            if nr[c] != "":
                fill[c] += 1
        nr["notes"] = r.get("notes", "")
        nr["review_seconds"] = r.get("review_seconds", "")
        nr["reviewed"] = "true" if all(nr[c] != "" for c in KEEP) else "false"
        out_rows.append(nr)

    tmp = src.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NEW_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)
    tmp.replace(src)

    n_done = sum(1 for r in out_rows if r["reviewed"] == "true")
    print(f"backup     : {bak}")
    print(f"rows       : {len(out_rows)}")
    print(f"dropped    : {dropped_present or '(none present)'}")
    print(f"blanked    : {sorted(BLANK_ON_MIGRATE)}")
    print(f"added      : repro_visible (blank)")
    print(f"reviewed   : {n_done}/{len(out_rows)} complete under new schema")
    print("per-field fill (non-blank):")
    for c in KEEP:
        print(f"  {c:24s} {fill[c]}/{len(out_rows)}")
    print(f"written    : {src}")
    print("\nremaining manual sweep: structures.foliage, repro_visible, attached_photo")


if __name__ == "__main__":
    main()
