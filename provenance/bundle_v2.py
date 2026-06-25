#!/usr/bin/env python3
"""
bundle_v2.py — Local -> HPC dataset bundler for the v2 herbarium captioner.

Walks an `output_simplified/<taxon>/` tree of (jpg, json) pairs, extracts the
top-level `caption` object from each sidecar, drops `caption_failed` and
`refs.fragment_packet`, then writes a self-contained, relative-pathed package
(HF arrow datasets + images + manifest) ready to zip and paste to scratch.

Label schema = schema.json MINUS refs.fragment_packet. Split = grouped
(taxon) + stratified (reproductive vs vegetative), seeded, deterministic.

Output layout (all paths inside the package are relative):
    <out>/
        images/<taxon>/<gbifID>.jpg
        train/                      # datasets.load_from_disk target
        val/
        test/                       # only if --test-frac > 0
        manifest.csv                # taxonID, gbifID, repro, split, image_path
        dataset_summary.json
    <out>.zip                       # if --zip

HPC side reads each row as: Image.open(DATA_ROOT / row["image_path"]).
"""

from __future__ import annotations
import argparse
import json
import random
import shutil
import sys
from collections import defaultdict, Counter
from pathlib import Path

# ---- canonical label construction --------------------------------------------
# Pulling each key explicitly enforces the schema (KeyError on a malformed
# sidecar) and structurally drops caption_failed + refs.fragment_packet.

PHENOLOGY_FLAGS = (
    "flower", "fruit", "pollen_cone", "seed_cone",
    "sporulating", "reproductive_unknown",
)


def build_label(cap: dict) -> dict:
    s = cap["structures"]
    ph = s["phenology"]
    r = cap["refs"]
    return {
        "type": cap["type"],
        "attached_photo": cap["attached_photo"],
        "structures": {
            "foliage": s["foliage"],
            "foliage_type": s["foliage_type"],
            "stem": s["stem"],
            "phenology": {k: bool(ph[k]) for k in PHENOLOGY_FLAGS},
        },
        "refs": {  # fragment_packet intentionally omitted
            "label": r["label"],
            "barcode": r["barcode"],
            "stamp": r["stamp"],
            "crc": r["crc"],
            "scale_bar": r["scale_bar"],
        },
    }


def is_reproductive(label: dict) -> bool:
    return any(label["structures"]["phenology"].values())


# ---- record collection -------------------------------------------------------

def collect_records(src: Path, keep_failed: bool):
    """Yield dicts per usable specimen; track rejections with reasons."""
    records, rejects = [], []
    taxon_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    if not taxon_dirs:
        sys.exit(f"No taxon subdirectories under {src}")

    for tdir in taxon_dirs:
        taxon = tdir.name
        for jpath in sorted(tdir.glob("*.json")):
            if jpath.name == "_summary.json":
                continue
            gbif = jpath.stem
            ipath = jpath.with_suffix(".jpg")
            try:
                meta = json.loads(jpath.read_text())
                cap = meta["caption"]
                if not keep_failed and (
                    cap.get("caption_failed")
                    or meta.get("recaption", {}).get("caption_failed")
                ):
                    rejects.append((taxon, gbif, "caption_failed"))
                    continue
                if not ipath.exists():
                    rejects.append((taxon, gbif, "missing_image"))
                    continue
                label = build_label(cap)
            except (KeyError, json.JSONDecodeError) as e:
                rejects.append((taxon, gbif, f"bad_sidecar:{type(e).__name__}"))
                continue

            records.append({
                "taxon": taxon,
                "taxonID": meta.get("target_taxonID"),
                "gbifID": gbif,
                "src_image": ipath,
                "rel_image": f"images/{taxon}/{gbif}.jpg",
                "caption_json": json.dumps(label, separators=(",", ":")),
                "repro": is_reproductive(label),
            })
    return records, rejects


# ---- split -------------------------------------------------------------------

def assign_splits(records, val_frac, test_frac, seed):
    """Flat random split over all records (no stratification, no floors).

    One seeded global shuffle, sliced by cumulative fractions:
      first n_val -> val, next n_test -> test, remainder -> train.
    Fractions are exact in expectation; a taxon is held out wholly only by
    chance (a 4-record taxon has ~0.997 odds of appearing in train at 75%).
    """
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    n = len(order)
    n_val = round(val_frac * n)
    n_test = round(test_frac * n)
    split_of = {}
    for i, r in enumerate(order):
        if i < n_val:
            split_of[r["gbifID"]] = "val"
        elif i < n_val + n_test:
            split_of[r["gbifID"]] = "test"
        else:
            split_of[r["gbifID"]] = "train"
    return split_of


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="output_simplified/ root (taxon subdirs of jpg+json)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output package dir (created; must not exist)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.0,
                    help="0 = train/val only (matches current notebook)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-failed", action="store_true",
                    help="do NOT drop caption_failed rows")
    ap.add_argument("--zip", action="store_true",
                    help="also produce <out>.zip for paste transfer")
    args = ap.parse_args()

    from datasets import Dataset  # imported here so --help works without the dep

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists():
        sys.exit(f"Refusing to overwrite existing {out}")
    if not src.is_dir():
        sys.exit(f"src not a directory: {src}")

    print(f"Scanning {src} ...")
    records, rejects = collect_records(src, args.keep_failed)
    if not records:
        sys.exit("No usable records collected.")
    split_of = assign_splits(records, args.val_frac, args.test_frac, args.seed)

    # materialize package
    (out / "images").mkdir(parents=True)
    buckets = defaultdict(list)
    for r in records:
        sp = split_of[r["gbifID"]]
        dst = out / r["rel_image"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["src_image"], dst)
        buckets[sp].append({"image_path": r["rel_image"],
                            "caption_json": r["caption_json"]})

    for sp, rows in buckets.items():
        Dataset.from_list(rows).save_to_disk(str(out / sp))

    # manifest
    import csv
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["taxonID", "taxon", "gbifID", "repro", "split", "image_path"])
        for r in sorted(records, key=lambda x: (x["taxon"], x["gbifID"])):
            w.writerow([r["taxonID"], r["taxon"], r["gbifID"],
                        int(r["repro"]), split_of[r["gbifID"]], r["rel_image"]])

    # summary
    split_counts = Counter(split_of.values())
    repro_counts = Counter((split_of[r["gbifID"]],
                            "repro" if r["repro"] else "veg") for r in records)
    n_taxa = len({r["taxon"] for r in records})

    summary = {
        "seed": args.seed,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_records": len(records),
        "n_taxa": n_taxa,
        "n_rejected": len(rejects),
        "splits": dict(split_counts),
        "split_x_repro": {f"{a}/{b}": c for (a, b), c in repro_counts.items()},
        "reject_reasons": dict(Counter(r[2] for r in rejects)),
        "taxa_absent_from_train": sum(
            1 for t in {r["taxon"] for r in records}
            if not any(split_of[r["gbifID"]] == "train"
                       for r in records if r["taxon"] == t)),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    if rejects:
        print(f"\n{len(rejects)} rejected (see reasons above; not packaged).")

    if args.zip:
        print("Zipping ...")
        shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"Wrote {out}.zip")
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
