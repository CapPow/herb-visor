#!/usr/bin/env python3
"""
enrich_clade.py  (pkg_v2/human_gt/)

Map each specimen's taxon -> coarse reproductive-capability group via the GBIF
backbone (pygbif name_backbone), so adjudicate.py can flag botanically-impossible
LLM predictions (e.g. pollen_cone on an angiosperm = confirmed hallucination).

Run from pkg_v2/:
    python human_gt/enrich_clade.py [--preds human_gt/preds.jsonl]

Outputs:
    human_gt/clade_cache.json   taxon -> full GBIF backbone match (cache; re-used)
    human_gt/clade_map.csv      gbifID, taxon, repro_group, + GBIF ranks + flags

Network: needs api.gbif.org. If offline, see --offline note at bottom.

------------------------------------------------------------------------------
GATE MAPPING — the botanical heart. Edit freely; logic reads only these sets.
Group -> which reproductive traits are POSSIBLE. Anything not possible, if the
LLM predicts true, is a confirmed hallucination.
------------------------------------------------------------------------------
"""
import argparse, csv, json, sys, time
from pathlib import Path

GYMNO_CLASSES = {"Pinopsida", "Cycadopsida", "Ginkgoopsida", "Gnetopsida",
                 "Coniferopsida"}
ANGIO_CLASSES = {"Magnoliopsida", "Liliopsida"}
SPORE_CLASSES = {"Polypodiopsida", "Filicopsida", "Pteridopsida",
                 "Marattiopsida", "Psilotopsida", "Equisetopsida",
                 "Lycopodiopsida"}
BRYO_PHYLA = {"Bryophyta", "Marchantiophyta", "Anthocerotophyta"}

# group -> set of traits that can legitimately be present
POSSIBLE = {
    "angiosperm":  {"flower", "fruit"},
    "gymnosperm":  {"pollen_cone", "seed_cone"},
    "spore_plant": {"sporulating"},
    "unknown":     {"flower", "fruit", "pollen_cone", "seed_cone", "sporulating"},
}
GATED_TRAITS = ["flower", "fruit", "pollen_cone", "seed_cone", "sporulating"]
# reproductive_unknown is a catch-all and is never gated.


def parse_backbone(r):
    """Flatten the GBIF v2 name_backbone response into flat keys the rest of
    the code expects. v2 shape: ranks in r['classification'] (list of
    {name,rank}); match info in r['diagnostics']; accepted name in r['usage'].
    Older flat responses pass through unchanged."""
    if not isinstance(r, dict):
        return {"matchType": "NONE"}
    if "classification" not in r and "diagnostics" not in r:
        return r                                    # already-flat (old schema)
    ranks = {c.get("rank", "").upper(): c.get("name")
             for c in r.get("classification", [])}
    diag = r.get("diagnostics", {})
    usage = r.get("usage", {})
    return {
        "matchType": diag.get("matchType", "NONE"),
        "confidence": diag.get("confidence", 0),
        "matched_name": usage.get("name", ""),
        "rank": usage.get("rank", ""),
        "kingdom": ranks.get("KINGDOM"),
        "phylum": ranks.get("PHYLUM"),
        "class": ranks.get("CLASS"),
        "order": ranks.get("ORDER"),
        "family": ranks.get("FAMILY"),
        "synonym": r.get("synonym"),
    }


def classify(bb):
    cls = bb.get("class"); phy = bb.get("phylum")
    if cls in GYMNO_CLASSES:
        return "gymnosperm"
    if cls in ANGIO_CLASSES:
        return "angiosperm"
    if cls in SPORE_CLASSES or phy in BRYO_PHYLA:
        return "spore_plant"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="human_gt/preds.jsonl")
    ap.add_argument("--cache", default="human_gt/clade_cache.json")
    ap.add_argument("--out", default="human_gt/clade_map.csv")
    ap.add_argument("--min-confidence", type=int, default=90,
                    help="flag matches below this GBIF confidence for review")
    args = ap.parse_args()

    try:
        from pygbif import species
    except ImportError:
        sys.exit("pygbif not installed: pip install pygbif  "
                 "(needs network to api.gbif.org)")

    rows = [json.loads(l) for l in Path(args.preds).read_text().splitlines() if l.strip()]
    taxa_by_id = {r["gbifID"]: r["taxon"] for r in rows}

    cache_path = Path(args.cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    uniq = sorted(set(taxa_by_id.values()))
    for i, t in enumerate(uniq, 1):
        # re-query if absent OR if a prior run cached a failure
        if t in cache and cache[t].get("matchType") not in ("ERROR", "NONE", None):
            continue
        try:
            raw = species.name_backbone(t, kingdom="Plantae", strict=False)
            bb = parse_backbone(raw)
            if bb.get("kingdom") and bb["kingdom"] != "Plantae":
                bb["matchType"] = "NONE"            # non-plant homonym, reject
            cache[t] = bb
        except Exception as e:
            cache[t] = {"matchType": "ERROR", "error": str(e)}
        if i % 25 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))
            print(f"  ...{i}/{len(uniq)} taxa resolved")
        time.sleep(0.05)
    cache_path.write_text(json.dumps(cache, indent=2))

    fields = ["gbifID", "taxon", "repro_group", "matchType", "confidence",
              "matched_name", "rank", "kingdom", "phylum", "class", "order",
              "family", "flag"]
    group_counts, flagged = {}, []
    with Path(args.out).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for gid, taxon in taxa_by_id.items():
            bb = cache.get(taxon, {})
            mt = bb.get("matchType", "NONE")
            conf = bb.get("confidence", 0)
            grp = classify(bb) if mt not in ("NONE", "ERROR") else "unknown"
            flag = ""
            if mt in ("NONE", "ERROR"):
                flag = "no_match"
            elif mt == "FUZZY":
                flag = "fuzzy"
            elif conf < args.min_confidence:
                flag = "low_confidence"
            elif grp == "unknown":
                flag = "unmapped_class"
            if flag:
                flagged.append((taxon, grp, mt, conf, bb.get("class"), flag))
            group_counts[grp] = group_counts.get(grp, 0) + 1
            w.writerow({
                "gbifID": gid, "taxon": taxon, "repro_group": grp,
                "matchType": mt, "confidence": conf,
                "matched_name": bb.get("scientificName", ""),
                "rank": bb.get("rank", ""), "kingdom": bb.get("kingdom", ""),
                "phylum": bb.get("phylum", ""), "class": bb.get("class", ""),
                "order": bb.get("order", ""), "family": bb.get("family", ""),
                "flag": flag,
            })

    print(f"\nwrote {args.out}  ({len(taxa_by_id)} specimens, {len(uniq)} unique taxa)")
    print("repro_group distribution:")
    for g in ("angiosperm", "gymnosperm", "spore_plant", "unknown"):
        print(f"  {g:12s} {group_counts.get(g, 0)}")
    if flagged:
        print(f"\n!! {len(flagged)} taxa need a human glance (gate trusts these):")
        for taxon, grp, mt, conf, cls, flag in flagged:
            print(f"  [{flag:14s}] {taxon:32s} -> {grp:11s} "
                  f"(class={cls}, {mt} conf={conf})")
    else:
        print("\nall taxa matched confidently.")


if __name__ == "__main__":
    main()

# OFFLINE FALLBACK: if api.gbif.org is unreachable, download the GBIF Backbone
# (Catalogue of Life-derived) Taxon.tsv and resolve class/phylum locally, or
# build clade_cache.json by hand for the (~unique) taxa — same schema:
#   {"<taxon>": {"matchType":"EXACT","confidence":99,"class":"Magnoliopsida",
#                "phylum":"Tracheophyta", ...}}
