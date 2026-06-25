#!/usr/bin/env python3
"""
eval_server.py — evaluate the local llama.cpp herbarium captioning server.

Two modes (single script):
  accuracy : run the whole held-out test split, strict-JSON parse only,
             report raw-valid %, exact-match %, and a per-field table
             (booleans -> tp/fp/tn/fn + P/R/acc; enums -> accuracy).
             Also reports images/sec as a freebie.
  speed    : sweep concurrency over the first --n images, report
             images/sec + p50/p95 latency per concurrency level.

Place this file in the pkg_v2 dir (alongside images/, test/, manifest.csv).
Image paths in the test arrow are treated as relative to this dir.

Requires: aiohttp, datasets  (pip install aiohttp datasets)
"""

import argparse
import asyncio
import base64
import csv
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import datasets

# ── CONFIG (edit here) ────────────────────────────────────────────────────────
ENDPOINT   = "http://192.168.1.112:8000/v1/chat/completions"
MODEL      = "qwenHerbarium"
MAX_TOKENS = 300

DATA_ROOT  = Path(__file__).resolve().parent          # pkg_v2/
TEST_DIR   = DATA_ROOT / "test"                        # held-out arrow
OUT_CSV    = DATA_ROOT / "eval_test_results.csv"

GT_COL_CANDIDATES   = ("caption_json", "caption")      # ground-truth JSON column
PATH_COL_CANDIDATES = ("image_path",)                  # relative image path column

# 16 evaluated leaf fields (fragment_packet intentionally excluded)
BOOL_FIELDS = [
    "attached_photo",
    "structures.phenology.flower",
    "structures.phenology.fruit",
    "structures.phenology.pollen_cone",
    "structures.phenology.seed_cone",
    "structures.phenology.sporulating",
    "structures.phenology.reproductive_unknown",
    "refs.label",
    "refs.barcode",
    "refs.stamp",
    "refs.crc",
    "refs.scale_bar",
]
ENUM_FIELDS = [
    "type",
    "structures.foliage",
    "structures.foliage_type",
    "structures.stem",
]
FIELDS = ENUM_FIELDS[:1] + BOOL_FIELDS[:1] + [  # display order ~ schema; not critical
    "structures.foliage", "structures.foliage_type", "structures.stem",
    *BOOL_FIELDS[1:],
]
# de-dup while preserving order
FIELDS = list(dict.fromkeys(ENUM_FIELDS + BOOL_FIELDS))


# ── helpers ───────────────────────────────────────────────────────────────────
def get_nested(d, key_path):
    for k in key_path.split("."):
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d


def taxon_from_path(rel_path: str) -> str:
    """images/Abelia_chinensis/123.jpg -> 'Abelia chinensis' (bare, natural language)."""
    return Path(rel_path).parent.name.replace("_", " ")


def parse_strict(text):
    """Strip only. Returns (dict|None, raw_valid_bool)."""
    s = text.strip()
    raw_valid = s.startswith("{") and s.endswith("}")
    try:
        return json.loads(s), raw_valid
    except json.JSONDecodeError:
        return None, raw_valid


def encode_image(abs_path: Path) -> str:
    with open(abs_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def build_payload(taxon: str, b64: str, temp: float) -> dict:
    return {
        "model": MODEL,
        "temperature": temp,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": taxon},
            ],
        }],
    }


def detect_col(ds, candidates, label):
    for c in candidates:
        if c in ds.column_names:
            return c
    sys.exit(f"ERROR: no {label} column found in {candidates}; "
             f"available: {ds.column_names}")


# ── data loading ──────────────────────────────────────────────────────────────
def load_samples():
    """Returns list of dicts: {rel, abs, taxon, gt_str}. Skips missing images."""
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: test dir not found: {TEST_DIR}")
    ds = datasets.load_from_disk(str(TEST_DIR))
    gt_col   = detect_col(ds, GT_COL_CANDIDATES, "ground-truth")
    path_col = detect_col(ds, PATH_COL_CANDIDATES, "image-path")

    out, missing = [], 0
    for row in ds:
        rel = row[path_col]
        ap  = DATA_ROOT / rel
        if not ap.exists():
            missing += 1
            continue
        out.append({
            "rel": rel,
            "abs": ap,
            "taxon": taxon_from_path(rel),
            "gt_str": row[gt_col].strip(),
        })
    if missing:
        print(f"  WARNING: {missing} image(s) not found on disk; skipped.")
    return out


def preencode(samples):
    """base64 all images up front (threaded) so timing isolates the server."""
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        b64s = list(ex.map(lambda s: encode_image(s["abs"]), samples))
    for s, b in zip(samples, b64s):
        s["b64"] = b
    dt = time.perf_counter() - t0
    print(f"  pre-encoded {len(samples)} images in {dt:.1f}s "
          f"({len(samples)/dt:.1f} img/s, client-side, not counted in server timing)")
    return samples


# ── async request core ────────────────────────────────────────────────────────
async def post_one(session, sem, payload):
    """Returns (text|None, latency_s, ok_bool)."""
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(ENDPOINT, json=payload) as r:
                data = await r.json()
                dt = time.perf_counter() - t0
                if r.status != 200:
                    return (None, dt, False)
                text = data["choices"][0]["message"]["content"]
                return (text, dt, True)
        except Exception:
            return (None, time.perf_counter() - t0, False)


async def run_batch(samples, concurrency, temp):
    """Fire one request per sample, bounded by `concurrency`. Order preserved.
    Returns (responses, latencies, wall_s)."""
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [post_one(session, sem, build_payload(s["taxon"], s["b64"], temp))
                 for s in samples]
        results = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    responses  = [r[0] for r in results]
    latencies  = [r[1] for r in results]
    return responses, latencies, wall


# ── accuracy mode ─────────────────────────────────────────────────────────────
def evaluate(samples, responses):
    n = len(samples)
    n_raw_valid = n_parsed = n_exact = 0
    # bool tallies
    bt = {f: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for f in BOOL_FIELDS}
    # enum tallies
    ec = {f: {"correct": 0, "n": 0} for f in ENUM_FIELDS}
    dump_rows = []  # per-sample record for the manual-review pipeline

    for s, resp in zip(samples, responses):
        try:
            gt = json.loads(s["gt_str"])
        except json.JSONDecodeError:
            continue
        if resp is None:
            continue
        pred, raw_valid = parse_strict(resp)
        if raw_valid:
            n_raw_valid += 1
        if pred is None:
            continue
        n_parsed += 1
        exact = json.dumps(pred, sort_keys=True) == json.dumps(gt, sort_keys=True)
        if exact:
            n_exact += 1

        # per-sample dump: student pred + teacher gt + per-field agreement (all 16)
        agree = {f: (get_nested(pred, f) == get_nested(gt, f)) for f in FIELDS}
        dump_rows.append({
            "gbifID": Path(s["rel"]).stem,
            "taxon": s["taxon"],
            "image_path": s["rel"],
            "student_pred": pred,
            "teacher_gt": gt,
            "agree": agree,
            "exact_match": exact,
        })

        for f in BOOL_FIELDS:
            gt_pos   = get_nested(gt, f) is True
            pred_pos = get_nested(pred, f) is True
            if   gt_pos and pred_pos:       bt[f]["tp"] += 1
            elif gt_pos and not pred_pos:   bt[f]["fn"] += 1
            elif not gt_pos and pred_pos:   bt[f]["fp"] += 1
            else:                           bt[f]["tn"] += 1

        for f in ENUM_FIELDS:
            ec[f]["n"] += 1
            if get_nested(pred, f) == get_nested(gt, f):
                ec[f]["correct"] += 1

    return {
        "n": n, "raw_valid": n_raw_valid, "parsed": n_parsed,
        "exact": n_exact, "bool": bt, "enum": ec,
    }, dump_rows


def pr(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return p, r


def report_accuracy(res, wall):
    n = res["n"]
    print("\n" + "=" * 78)
    print("HELD-OUT TEST EVALUATION")
    print("=" * 78)
    print(f"  Samples            : {n}")
    print(f"  Raw JSON valid     : {res['raw_valid']:4d}  ({100*res['raw_valid']/n:.1f}%)")
    print(f"  Strict-parsed      : {res['parsed']:4d}  ({100*res['parsed']/n:.1f}%)")
    print(f"  Exact match        : {res['exact']:4d}  ({100*res['exact']/n:.1f}%)")
    print(f"  Throughput         : {n/wall:.2f} img/s  ({wall:.1f}s wall)")
    d = res["parsed"] or 1

    print("\n  ENUM FIELDS" + " " * 29 + "acc%")
    print("  " + "-" * 44)
    for f in ENUM_FIELDS:
        e = res["enum"][f]
        acc = 100 * e["correct"] / e["n"] if e["n"] else float("nan")
        print(f"  {f:<40} {acc:6.1f}")

    print("\n  BOOLEAN FIELDS (positive = true)")
    print(f"  {'field':<42}{'tp':>4}{'fp':>4}{'fn':>4}{'tn':>5}"
          f"{'prec%':>7}{'rec%':>7}{'acc%':>7}")
    print("  " + "-" * 76)
    for f in BOOL_FIELDS:
        c = res["bool"][f]
        p, r = pr(c["tp"], c["fp"], c["fn"])
        acc = 100 * (c["tp"] + c["tn"]) / d
        print(f"  {f:<42}{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}{c['tn']:>5}"
              f"{100*p:>7.1f}{100*r:>7.1f}{acc:>7.1f}")
    print("=" * 78)


def save_csv(res, wall):
    n, d = res["n"], (res["parsed"] or 1)
    rows = [{
        "field": "SUMMARY", "n": n,
        "raw_valid_%":  round(100 * res["raw_valid"] / n, 1),
        "parsed_%":     round(100 * res["parsed"]   / n, 1),
        "exact_%":      round(100 * res["exact"]    / n, 1),
        "img_per_s":    round(n / wall, 2),
        "kind": "", "tp": "", "fp": "", "fn": "", "tn": "",
        "prec_%": "", "rec_%": "", "acc_%": "",
    }]
    for f in ENUM_FIELDS:
        e = res["enum"][f]
        rows.append({"field": f, "kind": "enum",
                     "acc_%": round(100 * e["correct"] / e["n"], 1) if e["n"] else None,
                     "n": "", "raw_valid_%": "", "parsed_%": "", "exact_%": "",
                     "img_per_s": "", "tp": "", "fp": "", "fn": "", "tn": "",
                     "prec_%": "", "rec_%": ""})
    for f in BOOL_FIELDS:
        c = res["bool"][f]
        p, r = pr(c["tp"], c["fp"], c["fn"])
        rows.append({"field": f, "kind": "bool",
                     "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
                     "prec_%": round(100 * p, 1), "rec_%": round(100 * r, 1),
                     "acc_%": round(100 * (c["tp"] + c["tn"]) / d, 1),
                     "n": "", "raw_valid_%": "", "parsed_%": "", "exact_%": "",
                     "img_per_s": ""})
    cols = ["field", "kind", "n", "raw_valid_%", "parsed_%", "exact_%",
            "img_per_s", "tp", "fp", "fn", "tn", "prec_%", "rec_%", "acc_%"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {OUT_CSV}")


def write_dump(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"Dumped {len(rows)} per-sample records: {path}")


# ── speed mode ────────────────────────────────────────────────────────────────
def report_speed(concurrency, latencies, wall, n):
    lat = sorted(latencies)
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
    print(f"  conc={concurrency:<3}  {n/wall:7.2f} img/s   "
          f"p50={p50*1000:7.0f}ms  p95={p95*1000:7.0f}ms  wall={wall:5.1f}s")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["accuracy", "speed"], default="accuracy")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="request-side temperature (0 = greedy; for accuracy runs)")
    ap.add_argument("--accuracy-concurrency", type=int, default=8)
    ap.add_argument("--n", type=int, default=100,
                    help="speed mode: images per sweep")
    ap.add_argument("--concurrency", type=str, default="1,4,8",
                    help="speed mode: comma-separated concurrency levels")
    ap.add_argument("--dump", type=str,
                    default=str(DATA_ROOT / "human_gt" / "preds.jsonl"),
                    help="accuracy mode: per-sample JSONL for the review pipeline "
                         "(empty string to disable)")
    args = ap.parse_args()

    print(f"Loading test split: {TEST_DIR}")
    samples = load_samples()
    print(f"  {len(samples)} samples ready.")

    if args.mode == "accuracy":
        preencode(samples)
        print(f"Running accuracy: {len(samples)} imgs, "
              f"concurrency={args.accuracy_concurrency}, temp={args.temp}")
        responses, _, wall = asyncio.run(
            run_batch(samples, args.accuracy_concurrency, args.temp))
        res, dump_rows = evaluate(samples, responses)
        report_accuracy(res, wall)
        save_csv(res, wall)
        if args.dump:
            write_dump(args.dump, dump_rows)

    else:  # speed
        sweep = [int(x) for x in args.concurrency.split(",")]
        subset = samples[:args.n]
        preencode(subset)
        print(f"Running speed: n={len(subset)}, sweep={sweep}, temp={args.temp}\n")
        for c in sweep:
            _, latencies, wall = asyncio.run(run_batch(subset, c, args.temp))
            ok = [l for l, ok in zip(latencies, [True] * len(latencies))]  # all returned
            report_speed(c, latencies, wall, len(subset))


if __name__ == "__main__":
    main()
