#!/usr/bin/env python3
"""
recaption.py — Re-caption an EXISTING harvest with the simplified boolean
phenology schema. No GBIF, no downloads, no classifier, no screener.

What it does
------------
Walks a source output tree (produced by broad_harvest_v3.py), finds the kept
samples in each taxon folder, copies each image into a NEW output tree, and
writes a fresh sidecar whose `caption` block is produced by a clean VLM pass
against schema.json / caption_prompt.txt in THIS directory. The source tree is
never modified.

Strict sample discovery
-----------------------
A kept sample sidecar is matched ONLY by ^\\d+\\.json$ at the top level of a
taxon folder. Excludes _summary.json (leading underscore), <gbifID>_*.json
side files (e.g. screener sidecars), and anything under <taxon>/rejected/
(never recursed into). Sidecar JSON is parsed strictly; a malformed sidecar is
recorded as an error and skipped, never salvaged.

Validation + escalation ladder
-------------------------------
Validation HARD-FAILS (no auto-repair). On a content/validation failure the
caption is retried up an escalation ladder; on a transport failure it retries
in place and does NOT climb (escalation can't fix a dead endpoint, and more
reasoning only lengthens a timeout):

    attempt 1:  reasoning=<base>, temp=0.0, timeout=<base>   (matches bulk corpus)
    attempt 2:  reasoning=medium, temp=0.0, timeout=120      (more compute)
    attempt 3:  reasoning=medium, temp=0.3, timeout=120      (last resort: new draw)

The accepted attempt's params are written to the sidecar's `recaption` block.
On terminal failure the full ladder is stored, tagged by error_class:
    "transport"            - endpoint/network death (NOT a specimen problem)
    "validation_exhausted" - schema-valid caption unreachable across the ladder
                             (your ambiguous / cross-clade specimen pile)

Cross-clade phenology exclusivity is relaxed for families listed in
relaxed_families.txt (Gnetophyta terminology blur); the relaxation is keyed on
the sidecar's family_name and fails closed when that is null/missing.

Parallelism
-----------
One worker thread per --vlm-model, each pinned to one llama.cpp GPU instance,
pulling whole taxon folders off a shared queue.

Resume
------
A destination <gbifID>.json that parses, validates (with the same family /
relaxed context), and is not caption_failed is left untouched. Everything else
is (re)captioned. Copying the image is idempotent.

Usage
-----
    python recaption.py --source-dir ../../output --output-dir ../../output_simplified \\
        --vlm-models qwen27b1,qwen27b2 --reasoning-effort off
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import shutil
import sys
import threading
import time
from pathlib import Path

# Self-contained local import (avoids colliding with the original src/captioner.py
# regardless of how the script is invoked).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simple_captioner import (  # noqa: E402
    caption as vlm_caption,
    validate,
    load_relaxed_families,
    CaptionError,
    CaptionValidationError,
)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (CLI-overridable)
# ──────────────────────────────────────────────────────────────────────────────
VLM_ENDPOINT          = "http://192.168.1.112:8000/v1/chat/completions"
VLM_MODELS            = ("qwen27b1", "qwen27b2")
VLM_REASONING_EFFORT  = "medium"          # base rung reasoning effort
VLM_BASE_TIMEOUT      = 190             # base rung timeout (seconds)
VLM_ESCALATED_TIMEOUT = 300           # rungs 2-3 timeout
VLM_TRANSPORT_RETRIES = 3             # transport retries WITHIN a rung
VLM_RETRY_BACKOFF     = 2.0           # seconds, * attempt number

SCHEMA_VERSION = "simplified_phenology_v2"

# Strict kept-sidecar matcher: digits-only stem, .json extension.
KEPT_SIDECAR_RE = re.compile(r"^\d+\.json$")

# Old caption-derived keys dropped from the carried-over sidecar.
_DROP_KEYS = ("caption", "dwc", "ppo_terms", "repro_agreement")

# Phenology flags in the simplified schema (for the summary tally).
PHENO_FLAGS = ("flower", "fruit", "pollen_cone", "seed_cone",
               "sporulating", "reproductive_unknown")

log = logging.getLogger("recaption")
SHUTDOWN = threading.Event()


# ──────────────────────────────────────────────────────────────────────────────
# Escalation ladder
# ──────────────────────────────────────────────────────────────────────────────
def build_ladder(base_effort: str, base_timeout: int) -> list[dict]:
    """Three-rung ladder. Rung 1 matches the bulk corpus (CLI base); rungs 2-3
    add reasoning, then a non-zero temperature as a last resort."""
    return [
        {"attempt": 1, "reasoning_effort": base_effort, "temperature": 0.6, "timeout": base_timeout},
        {"attempt": 2, "reasoning_effort": "high",    "temperature": 0.6, "timeout": VLM_ESCALATED_TIMEOUT},
        {"attempt": 3, "reasoning_effort": "high",    "temperature": 1.0, "timeout": VLM_ESCALATED_TIMEOUT},
    ]


def _rung_meta(rung: dict) -> dict:
    return {k: rung[k] for k in ("attempt", "reasoning_effort", "temperature", "timeout")}


def run_caption(img_path: Path, model: str, endpoint: str, base_effort: str,
                context: str | None, family: str | None,
                relaxed_families: frozenset) -> dict:
    """Caption one image, climbing the ladder on validation failures only.

    Returns:
        {
          "cap":         validated caption dict (caption_failed=False) OR a
                         failure stub (caption_failed=True),
          "accepted":    accepted rung meta | None,
          "attempts":    list of attempt records (ladder tried so far),
          "error_class": None | "transport" | "validation_exhausted",
        }
    """
    ladder = build_ladder(base_effort, VLM_BASE_TIMEOUT)
    attempts: list[dict] = []
    # print("=================================")
    # print(ladder)
    # print("=================================")
    for rung in ladder:
        for t in range(1, VLM_TRANSPORT_RETRIES + 1):
            try:
                #print(f"params: {int(rung['timeout'])}, {rung['reasoning_effort']}, {rung["temperature"]}")
                cap = vlm_caption(
                    str(img_path), endpoint,
                    reasoning_effort=rung["reasoning_effort"],
                    timeout=int(rung["timeout"]),
                    context=context,
                    model=model,
                    temperature=rung["temperature"],
                    family=family,
                    relaxed_families=relaxed_families,
                )
                cap["caption_failed"] = False
                return {"cap": cap, "accepted": _rung_meta(rung),
                        "attempts": attempts, "error_class": None}

            except CaptionValidationError as e:
                # Deterministic-ish content failure at temp 0 → don't burn the
                # transport retries; climb to the next rung.
                attempts.append({**_rung_meta(rung), "transport_try": t,
                                 "error_type": "validation", "error": str(e)})
                break

            except CaptionError as e:  # transport / protocol
                attempts.append({**_rung_meta(rung), "transport_try": t,
                                 "error_type": "transport", "error": str(e)})
                if t < VLM_TRANSPORT_RETRIES:
                    time.sleep(VLM_RETRY_BACKOFF * t)
                    continue
                # Transport exhausted at this rung → terminal; escalation won't help.
                stub = {"caption_failed": True, "error_class": "transport", "error": str(e)}
                return {"cap": stub, "accepted": None,
                        "attempts": attempts, "error_class": "transport"}

    # Ladder exhausted on content/validation failures → ambiguous-or-invalid.
    last = attempts[-1]["error"] if attempts else "unknown"
    stub = {"caption_failed": True, "error_class": "validation_exhausted", "error": last}
    return {"cap": stub, "accepted": None,
            "attempts": attempts, "error_class": "validation_exhausted"}


# ──────────────────────────────────────────────────────────────────────────────
# Sidecar discovery + I/O
# ──────────────────────────────────────────────────────────────────────────────
def iter_kept_sidecars(taxon_dir: Path):
    """Yield top-level kept-sample sidecar paths (strict ^\\d+\\.json$ match)."""
    for p in sorted(taxon_dir.glob("*.json")):
        if KEPT_SIDECAR_RE.match(p.name):
            yield p


def load_sidecar_strict(path: Path) -> dict:
    """Parse a sidecar strictly. Raises on malformed JSON or non-object."""
    obj = json.loads(path.read_text(encoding="utf-8"))  # may raise JSONDecodeError
    if not isinstance(obj, dict):
        raise ValueError(f"sidecar root is {type(obj).__name__}, expected object")
    return obj


def dest_is_complete(dest_json: Path, relaxed_families: frozenset) -> bool:
    """True if the destination sidecar already holds a valid, non-failed caption.

    Validated with the SAME family / relaxed context, so a legitimately-relaxed
    cross-clade caption is not re-flagged on resume."""
    if not dest_json.exists():
        return False
    try:
        obj = json.loads(dest_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    cap = obj.get("caption")
    if not isinstance(cap, dict) or cap.get("caption_failed"):
        return False
    # `caption_failed` is bookkeeping we inject onto the caption dict; it is not
    # a schema field, so strip it before re-validating against the schema's
    # additionalProperties:false root.
    cap_check = {k: v for k, v in cap.items() if k != "caption_failed"}
    ok, _ = validate(cap_check, family=obj.get("family_name"),
                     relaxed_families=relaxed_families)
    return ok


def build_new_sidecar(old: dict, result: dict, src_sidecar: Path,
                      model: str, base_effort: str,
                      image_size: list | None) -> dict:
    """Carry provenance from the old sidecar, swap in the fresh caption +
    recaption metadata (accepted params on success; full ladder on failure)."""
    cap = result["cap"]
    new = {k: v for k, v in old.items() if k not in _DROP_KEYS}
    if image_size is not None and not new.get("image_size"):
        new["image_size"] = image_size
    failed = bool(cap.get("caption_failed"))
    new["caption"] = cap

    rc = {
        "schema":                SCHEMA_VERSION,
        "model":                 model,
        "base_reasoning_effort": base_effort,
        "source_sidecar":        str(src_sidecar),
        "generated_at":          time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caption_failed":        failed,
    }
    if failed:
        rc["error_class"] = result["error_class"]   # transport | validation_exhausted
        rc["attempts"]    = result["attempts"]       # full ladder tried
    else:
        rc["params"] = result["accepted"]            # accepted attempt only
    new["recaption"] = rc
    return new


# ──────────────────────────────────────────────────────────────────────────────
# Per-taxon processing
# ──────────────────────────────────────────────────────────────────────────────
def process_taxon(taxon_dir: Path, dest_root: Path, model: str, endpoint: str,
                  base_effort: str, relaxed_families: frozenset) -> dict:
    dest_dir = dest_root / taxon_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    counters = dict(total=0, captioned=0, skipped_done=0,
                    failed_transport=0, failed_validation=0,
                    missing_image=0, bad_sidecar=0)
    pheno_tally = {f: 0 for f in PHENO_FLAGS}
    n_vegetative = 0

    for src_json in iter_kept_sidecars(taxon_dir):
        if SHUTDOWN.is_set():
            break
        counters["total"] += 1
        stem = src_json.stem  # gbifID (digits, guaranteed by the regex)
        dest_json = dest_dir / f"{stem}.json"
        dest_jpg = dest_dir / f"{stem}.jpg"

        if dest_is_complete(dest_json, relaxed_families):
            counters["skipped_done"] += 1
            continue

        try:
            old = load_sidecar_strict(src_json)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            log.error("[%s] bad sidecar %s: %s", taxon_dir.name, src_json.name, e)
            counters["bad_sidecar"] += 1
            continue

        src_jpg = taxon_dir / f"{stem}.jpg"
        if not src_jpg.exists():
            log.error("[%s] missing image for %s", taxon_dir.name, src_json.name)
            counters["missing_image"] += 1
            continue

        if not dest_jpg.exists():
            shutil.copy2(src_jpg, dest_jpg)  # idempotent, preserves mtime

        context = old.get("scientificName") or old.get("target_taxon_name")
        family = old.get("family_name")
        result = run_caption(dest_jpg, model, endpoint, base_effort,
                             context, family, relaxed_families)

        new = build_new_sidecar(old, result, src_json, model, base_effort,
                                old.get("image_size"))
        dest_json.write_text(
            json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")

        cap = result["cap"]
        if cap.get("caption_failed"):
            if result["error_class"] == "transport":
                counters["failed_transport"] += 1
            else:
                counters["failed_validation"] += 1
        else:
            counters["captioned"] += 1
            ph = (cap.get("structures") or {}).get("phenology") or {}
            any_repro = False
            for f in PHENO_FLAGS:
                if ph.get(f):
                    pheno_tally[f] += 1
                    any_repro = True
            if not any_repro:
                n_vegetative += 1

    summary = {
        "target_taxon_name":    taxon_dir.name,
        "schema":               SCHEMA_VERSION,
        "model":                model,
        "base_reasoning_effort": base_effort,
        "generated_at":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "counts":               counters,
        "phenology_flag_tally": pheno_tally,
        "n_vegetative":         n_vegetative,
    }
    (dest_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("[%s] total=%d captioned=%d skip_done=%d fail_transport=%d "
             "fail_validation=%d missing_img=%d bad_sidecar=%d",
             taxon_dir.name, counters["total"], counters["captioned"],
             counters["skipped_done"], counters["failed_transport"],
             counters["failed_validation"], counters["missing_image"],
             counters["bad_sidecar"])
    return counters


# ──────────────────────────────────────────────────────────────────────────────
# Worker pool
# ──────────────────────────────────────────────────────────────────────────────
def _worker_loop(model: str, task_q: "queue.Queue", dest_root: Path,
                 endpoint: str, base_effort: str,
                 relaxed_families: frozenset) -> None:
    log.info("worker[model=%s] ready", model)
    while True:
        if SHUTDOWN.is_set():
            break
        item = task_q.get()
        try:
            if item is None or SHUTDOWN.is_set():
                break
            idx, total, taxon_dir = item
            log.info("=== [%d/%d] %s → %s ===", idx, total, taxon_dir.name, model)
            try:
                process_taxon(taxon_dir, dest_root, model, endpoint,
                              base_effort, relaxed_families)
            except Exception as e:
                log.exception("[%s] failed: %s", taxon_dir.name, e)
        finally:
            task_q.task_done()


def discover_taxa(source_dir: Path) -> list[Path]:
    """Taxon folders = immediate subdirectories holding ≥1 kept sidecar."""
    taxa = []
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        if next(iter_kept_sidecars(d), None) is not None:
            taxa.append(d)
    return taxa


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path, required=True,
                    help="Existing harvest output root (read-only; untouched).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="New output root for simplified captions.")
    ap.add_argument("--vlm-endpoint", default=VLM_ENDPOINT)
    ap.add_argument("--vlm-models", default=",".join(VLM_MODELS),
                    help="Comma-separated caption model names, one worker/GPU per name.")
    ap.add_argument("--reasoning-effort", default=VLM_REASONING_EFFORT,
                    choices=["off", "low", "medium", "high"],
                    help="Base (rung-1) reasoning effort; rungs 2-3 escalate to medium.")
    ap.add_argument("--relaxed-families", type=Path, default=None,
                    help="Override path to the relaxed-families allowlist "
                         "(default: relaxed_families.txt beside this script).")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s")

    source_dir = args.source_dir.resolve()
    dest_root = args.output_dir.resolve()
    if not source_dir.is_dir():
        sys.exit(f"Source dir not found: {source_dir}")
    if dest_root == source_dir:
        sys.exit("Refusing to run: --output-dir must differ from --source-dir.")

    models = tuple(m.strip() for m in args.vlm_models.split(",") if m.strip())
    if not models:
        sys.exit("No VLM models specified.")

    relaxed_families = load_relaxed_families(args.relaxed_families)

    taxa = discover_taxa(source_dir)
    if not taxa:
        sys.exit(f"No taxon folders with kept samples under {source_dir}")

    dest_root.mkdir(parents=True, exist_ok=True)
    log.info("source=%s dest=%s taxa=%d models=%s endpoint=%s base_reasoning=%s "
             "relaxed_families=%d",
             source_dir, dest_root, len(taxa), models, args.vlm_endpoint,
             args.reasoning_effort, len(relaxed_families))

    task_q: "queue.Queue" = queue.Queue()
    for i, d in enumerate(taxa, 1):
        task_q.put((i, len(taxa), d))
    for _ in models:
        task_q.put(None)

    t0 = time.time()
    threads = []
    for model in models:
        t = threading.Thread(
            target=_worker_loop, name=f"w-{model}",
            args=(model, task_q, dest_root, args.vlm_endpoint,
                  args.reasoning_effort, relaxed_families),
            daemon=True)
        t.start()
        threads.append(t)

    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        log.warning("Interrupt — halting new taxa/captions. An in-flight caption "
                    "on a GPU will finish; model stays loaded. Exiting…")
        SHUTDOWN.set()
        for t in threads:
            t.join(timeout=max(5, VLM_ESCALATED_TIMEOUT // 4))

    log.info("All done. Elapsed: %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
