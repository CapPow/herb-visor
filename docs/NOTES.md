# Notes

Background, lineage, and interpretation caveats for Herb-VISOR. The README covers usage and results; this file covers how the artifact was produced and how to read its numbers honestly.

## Project lineage

The public repository contains the validation pipeline and the model. The upstream stages that produced the training data and weights are documented here for transparency but are not packaged for re-execution (they depend on local paths, harvested images, and HPC resources).

```
Teacher captioning        A larger reasoning model (qwen27b, Qwen3.6-27B-UD-Q5_K_XL)
(provenance/)             captioned harvested specimen images into structured JSON
                          against the schema in schema/schema.json.
        |
        v
Frozen training input     Images, HF arrow splits, and manifest were frozen as a
                          canonical snapshot. gbifID is the universal join key
                          across images, splits, manifest, and predictions.
        |
        v
Distillation training     Herb-VISOR (Qwen3-VL-4B) was full-weight fine-tuned on the
                          teacher's captions. Two phases: phase 1 with full schema
                          instructions in the prompt, phase 2 with image and taxon
                          name only. Phase 2 bakes the schema into the weights, so
                          end users need no prompt beyond the binomial.
        |
        v
Validation (validation/)  The frozen predictions were scored against a human gold
                          standard and checked for taxonomic consistency. This is
                          the only stage meant to be re-run.
```

## Data triage with Malon

Training images were harvested from GBIF aggregator records, which contain a fraction of material unsuitable for specimen captioning (in-situ field photos, habitat shots, unrelated media). Images were triaged with [Malon](https://github.com/CapPow/Malon), a lightweight three-class classifier for herbarium specimen images: not-useful, atypical, and standard.

Malon's default use is data-pollution filtering: discard not-useful, review atypical, keep standard. This pipeline used it in two directions. Not-useful images (including in-situ non-specimen photos) were excluded. Atypical images, rather than being discarded, were deliberately oversampled so the model would see hard and edge-case specimens during training rather than only clean sheets. The dataset card visualizes the resulting class balance.

## Images and re-harvesting

Specimen images are not redistributed in this repository. They carry mixed licenses inherited from their source institutions, and isolating a uniformly licensed subset is out of scope for this release.

The images are recoverable. Every record's `gbifID` is the image filename stem and the join key used throughout the pipeline (`manifest.csv`, `preds.jsonl`, the clade map, the verdicts). A `gbifID` resolves to its GBIF occurrence and associated media through the GBIF API or occurrence portal, so the training corpus can be reconstructed from the identifiers shipped here. The validation pipeline does not need the images: it runs entirely from the frozen prediction and label artifacts.

## Interpretation caveats

Carry these into any use of the model output or the reported numbers.

**Reproductive presence is category-level.** The human-validated field `repro_visible` records only that some reproductive structure is visible. Within-category discrimination (flower vs fruit vs cone type) was not human-validated, because a single non-specialist annotator cannot reliably make those calls across all vascular plants. The model emits fine-grained phenology fields, but their accuracy beyond the category level is unmeasured.

**The clade-consistency rate is an upper bound.** `clade_consistency.py` flags predictions asserting a trait botanically impossible for the specimen's clade. Several flagged cases are morphological mimics, not true errors: Casuarina and Allocasuarina bear woody cone-like infructescences, and Cycas and Gnetum have fruit-like structures. The shipped code does not apply any taxonomic relaxation, so the reported rate (8/643 for Herb-VISOR, 6/643 for the teacher) overcounts real errors. Read it as a ceiling.

**The clade gate sees only false positives.** Taxonomy can confirm a trait is impossible for a clade; it cannot confirm that a present structure was read correctly, and it cannot detect a real structure the model missed. The check is a constraint-adherence measure, not an accuracy measure. A zero impossible rate does not imply correctness.

**Impossible-rate denominators are asymmetric.** Cone and spore traits are well-powered because most of the corpus is angiosperm. Flower- and fruit-impossible cases rest on the small gymnosperm and spore-plant tail. The clade distribution (574 angiosperm, 10 gymnosperm, 59 spore-plant) should be read alongside the rates.

**Ground truth is soft.** Validation is a single non-specialist annotator over 100 specimens; 36 of 100 carry at least one uncertain field. Some apparent model errors are likely annotator-limited rather than model errors (for example, bract vs leaf, or the label vs stamp boundary). The reported accuracies are a conservative floor, and the model's true accuracy is plausibly higher on fields where annotation was hardest. Model output is a curator-assist candidate, not authoritative write-back.

**Herb-VISOR tracks its teacher, including its errors.** Distillation preserved teacher behavior; the student did not exceed the teacher on either accuracy or clade consistency. The `type` field is always `PH` on herbarium input and is not a discriminative result.

## Deployment tuning

The llama.cpp server used to produce the speed and accuracy numbers in this
repository set `image-min-tokens 2048`, raising the vision-tower's per-image
token floor. In a same-image spot check (an *Acer pseudoplatanus* sheet, q8
weights), this changed the image from ~1,292 to ~2,088 prompt tokens but
produced byte-identical JSON. It may help on sparse images where a small
specimen occupies little of the sheet. The model produces valid output without it.

## Dataset card

The interactive dataset card (`docs/dataset-card.html`, viewable through GitHub Pages) was generated from aggregates over the training corpus by a local build pipeline. That pipeline depends on local harvest paths and is not shipped. Each section of the card is tagged by data source: GBIF-verified metadata, Malon classifier output, or teacher-inferred (model output, unverified).
