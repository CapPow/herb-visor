#!/usr/bin/env python3
"""
review_gui.py  (pkg_v2/human_gt/)  — BLIND herbarium review tool.

Run from pkg_v2/:
    python human_gt/review_gui.py [--manifest human_gt/review_manifest.json]
                                  [--out human_gt/review_verdicts.csv]

NEVER shows student/teacher predictions. You determine all 16 fields cold.

WORKFLOW (column-wise): one field at a time, swept across all specimens in
the manifest's (already-random) order. Answer -> auto-advance to next specimen;
at the end of a field, jumps to the next field. Resumable: progress reloads on
launch and you restart at the first unanswered cell.

CONTROLS
  Booleans:  y = true   n = false   u = uncertain   c = cannot_determine
  Enums:     1..N pick value (legend on right)   u = uncertain   c = cannot_determine
  Image:     mouse wheel = zoom (at cursor)   left-drag = pan   r = reset view
  Navigate:  ] next spec   [ prev spec   . next field   , prev field
  Notes:     TAB toggles per-specimen note entry (Enter/TAB commit, ESC cancel)
  Quit:      ESC (saves)
Saves after every answer.

VALUE SEMANTICS
  uncertain         = image is genuinely ambiguous on this field (model-emitted class)
  cannot_determine  = field not visible / you can't make the call (GUI escape hatch)
adjudicate.py drops both from that field's denominator but counts them separately.
"""
import argparse, csv, json, sys, time
from pathlib import Path
import pygame

# --- post-patch schema. Individual phenology fields + foliage_type are no
# longer human-adjudicated: 6 phenology -> single repro_visible super-field;
# foliage_type dropped (low annotator confidence, handled by clade soft-flag).
# Human-judged fields only. Join key gbifID + these go to adjudicate.py.
MANUAL_FIELDS = [
    "type", "structures.foliage", "structures.stem", "attached_photo",
    "refs.label", "refs.barcode", "refs.stamp", "refs.crc", "refs.scale_bar",
    "repro_visible",
]
ENUM_FIELDS = {"type", "structures.foliage", "structures.stem"}
# repro_visible: boolean — was any reproductive structure visible to the human?
# adjudicate.py maps it against LLM OR(flower,fruit,pollen_cone,seed_cone,
# sporulating,reproductive_unknown).
COLUMNS = ["gbifID", "reviewed"] + MANUAL_FIELDS + ["notes", "review_seconds"]

# review sweep order (visual-locus grouped; refs done, type/stem done,
# foliage re-annotate, repro_visible + attached_photo new)
SWEEP_ORDER = [
    "refs.label", "refs.barcode", "refs.stamp", "refs.crc", "refs.scale_bar",
    "type", "structures.foliage", "structures.stem",
    "repro_visible", "attached_photo",
]
assert set(SWEEP_ORDER) == set(MANUAL_FIELDS) and len(SWEEP_ORDER) == 10

# friendlier bool legend labels per field
BOOL_LABELS = {
    "repro_visible": ("repro structure visible", "none visible"),
}

BG = (24, 26, 30); PANEL = (34, 37, 43); FG = (228, 230, 234)
DIM = (140, 146, 156); ACCENT = (120, 200, 160); WARN = (235, 180, 90)
W, H = 1320, 840
PANEL_W = 360


def short(f):  # display label
    return f.split(".")[-1] if "." in f else f


class Reviewer:
    def __init__(self, manifest, out_path, copy_dir):
        self.specs = manifest                      # presentation order = manifest order
        self.out_path = out_path
        self.copy_dir = copy_dir
        self.v = {s["gbifID"]: {} for s in self.specs}   # gid -> {field: value}
        self.notes = {s["gbifID"]: "" for s in self.specs}
        self.sec = {s["gbifID"]: 0.0 for s in self.specs}
        self._load()
        self.field_i, self.spec_i = self._resume()
        # view state
        self.scale = 1.0; self.ox = 0; self.oy = 0
        self.img_cache = {}; self.cur_img = None
        self.cell_start = time.time()
        self.note_mode = False; self.note_buf = ""
        self.history = []          # (field_i, spec_i, gid, field, prev_value, added_sec)
        self.field_banner = True   # loud cue on entering a field

    # ---- persistence -------------------------------------------------------
    def _load(self):
        p = Path(self.out_path)
        if not p.exists():
            return
        with p.open() as f:
            for row in csv.DictReader(f):
                gid = row["gbifID"]
                if gid not in self.v:
                    continue
                for fld in MANUAL_FIELDS:
                    val = row.get(fld, "")
                    if val:
                        self.v[gid][fld] = val
                self.notes[gid] = row.get("notes", "") or ""
                try:
                    self.sec[gid] = float(row.get("review_seconds") or 0)
                except ValueError:
                    self.sec[gid] = 0.0

    def save(self):
        tmp = Path(self.out_path).with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for s in self.specs:
                gid = s["gbifID"]
                vals = self.v[gid]
                reviewed = all(vals.get(fld, "") != "" for fld in MANUAL_FIELDS)
                w.writerow(
                    [gid, "true" if reviewed else "false"]
                    + [vals.get(fld, "") for fld in MANUAL_FIELDS]
                    + [self.notes[gid], round(self.sec[gid], 1)]
                )
        tmp.replace(self.out_path)

    def _resume(self):
        for fi, fld in enumerate(SWEEP_ORDER):
            for si, s in enumerate(self.specs):
                if self.v[s["gbifID"]].get(fld, "") == "":
                    return fi, si
        return 0, 0

    # ---- current cell ------------------------------------------------------
    @property
    def field(self):
        return SWEEP_ORDER[self.field_i]

    @property
    def spec(self):
        return self.specs[self.spec_i]

    def allowed(self, field):
        # manifest carries allowed values (enum vals + 'uncertain'); strip 'uncertain'
        vals = [x for x in self.spec["fields"][field] if x != "uncertain"]
        return vals

    def _touch(self):
        self.cell_start = time.time()
        self.scale = 0.0  # force refit on draw
        self.cur_img = None

    def record(self, value):
        gid = self.spec["gbifID"]
        added = time.time() - self.cell_start
        prev = self.v[gid].get(self.field, "")
        self.history.append((self.field_i, self.spec_i, gid, self.field, prev, added))
        self.sec[gid] += added
        self.v[gid][self.field] = value
        self.field_banner = False
        self.save()
        self.advance()

    def undo(self):
        if not self.history:
            return
        fi, si, gid, field, prev, added = self.history.pop()
        self.field_i, self.spec_i = fi, si
        if prev == "":
            self.v[gid].pop(field, None)
        else:
            self.v[gid][field] = prev
        self.sec[gid] = max(0.0, self.sec[gid] - added)
        self.field_banner = False
        self.save()
        self._touch()

    def advance(self):
        prev_fi = self.field_i
        self.spec_i += 1
        if self.spec_i >= len(self.specs):
            self.spec_i = 0
            self.field_i = (self.field_i + 1) % len(SWEEP_ORDER)
        if self.field_i != prev_fi:
            self.field_banner = True
        self._touch()

    def nav(self, dspec=0, dfield=0):
        if dfield:
            self.field_i = (self.field_i + dfield) % len(SWEEP_ORDER)
            self.spec_i = 0
            self.field_banner = True
        if dspec:
            self.spec_i = (self.spec_i + dspec) % len(self.specs)
        self._touch()

    # ---- image -------------------------------------------------------------
    def load_image(self):
        if self.cur_img is not None:
            return self.cur_img
        ip = self.spec["image_path"]
        surf = self.img_cache.get(ip)
        if surf is None:
            cand = [Path(ip)]
            if self.copy_dir:
                cand.append(Path(self.copy_dir) / Path(ip).name)
            cand.append(Path("human_gt/images") / Path(ip).name)
            surf = None
            for c in cand:
                if c.exists():
                    try:
                        surf = pygame.image.load(str(c)).convert()
                        break
                    except pygame.error:
                        pass
            self.img_cache[ip] = surf
        self.cur_img = surf
        return surf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="human_gt/review_manifest.json")
    ap.add_argument("--out", default="human_gt/review_verdicts.csv")
    ap.add_argument("--images", default="human_gt/images",
                    help="optional copied-image dir fallback")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest:
        sys.exit("empty manifest")

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Blind herbarium review")
    font = pygame.font.SysFont("dejavusansmono", 18)
    big = pygame.font.SysFont("dejavusans", 26)
    small = pygame.font.SysFont("dejavusansmono", 15)
    clock = pygame.time.Clock()

    R = Reviewer(manifest, args.out, args.images)
    pane = pygame.Rect(0, 0, W - PANEL_W, H)
    dragging = False

    def fit(surf):
        if not surf:
            return
        sw, sh = surf.get_size()
        R.scale = min((pane.w - 40) / sw, (pane.h - 40) / sh)
        R.ox = (pane.w - sw * R.scale) / 2
        R.oy = (pane.h - sh * R.scale) / 2

    running = True
    while running:
        surf = R.load_image()
        if R.scale == 0.0:
            fit(surf)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif R.note_mode:
                if e.type == pygame.KEYDOWN:
                    if e.key in (pygame.K_RETURN, pygame.K_TAB):
                        R.notes[R.spec["gbifID"]] = R.note_buf; R.save(); R.note_mode = False
                    elif e.key == pygame.K_ESCAPE:
                        R.note_mode = False
                    elif e.key == pygame.K_BACKSPACE:
                        R.note_buf = R.note_buf[:-1]
                    elif e.unicode and e.unicode.isprintable():
                        R.note_buf += e.unicode
                continue

            elif e.type == pygame.KEYDOWN:
                k = e.unicode.lower()
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_TAB:
                    R.note_mode = True; R.note_buf = R.notes[R.spec["gbifID"]]
                elif k == "r":
                    fit(surf)
                elif k == "z":
                    R.undo()
                elif e.key == pygame.K_RIGHTBRACKET:
                    R.nav(dspec=+1)
                elif e.key == pygame.K_LEFTBRACKET:
                    R.nav(dspec=-1)
                elif k == ".":
                    R.nav(dfield=+1)
                elif k == ",":
                    R.nav(dfield=-1)
                elif k == "c":
                    R.record("cannot_determine")
                elif k == "u":
                    R.record("uncertain")
                elif R.field not in ENUM_FIELDS and k in ("y", "n"):
                    R.record("true" if k == "y" else "false")
                elif R.field in ENUM_FIELDS and k.isdigit():
                    opts = R.allowed(R.field)
                    idx = int(k) - 1
                    if 0 <= idx < len(opts):
                        R.record(opts[idx])

            elif e.type == pygame.MOUSEWHEEL and pane.collidepoint(pygame.mouse.get_pos()):
                mx, my = pygame.mouse.get_pos()
                factor = 1.1 if e.y > 0 else 1 / 1.1
                R.ox = mx - (mx - R.ox) * factor
                R.oy = my - (my - R.oy) * factor
                R.scale *= factor
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and pane.collidepoint(e.pos):
                dragging = True
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                R.ox += e.rel[0]; R.oy += e.rel[1]

        # ---- draw ----------------------------------------------------------
        screen.fill(BG)
        if surf:
            sw, sh = surf.get_size()
            scaled = pygame.transform.smoothscale(
                surf, (max(1, int(sw * R.scale)), max(1, int(sh * R.scale))))
            screen.set_clip(pane)
            screen.blit(scaled, (R.ox, R.oy))
            screen.set_clip(None)
        else:
            screen.blit(big.render("image not found", True, WARN), (40, 40))

        # panel
        px = W - PANEL_W
        pygame.draw.rect(screen, PANEL, (px, 0, PANEL_W, H))
        y = 24
        done = sum(1 for s in R.specs if R.v[s["gbifID"]].get(R.field, ""))
        screen.blit(small.render(
            f"field {R.field_i+1}/{len(SWEEP_ORDER)}   spec {R.spec_i+1}/{len(R.specs)}   "
            f"[{done}/{len(R.specs)} this field]", True, DIM), (px + 20, y)); y += 30
        screen.blit(big.render(short(R.field), True, ACCENT), (px + 20, y)); y += 44
        screen.blit(font.render(R.spec["taxon"], True, FG), (px + 20, y)); y += 28
        screen.blit(small.render(R.spec["gbifID"], True, DIM), (px + 20, y)); y += 40

        cur = R.v[R.spec["gbifID"]].get(R.field, "")
        screen.blit(font.render("current: " + (cur or "—"),
                                True, ACCENT if cur else DIM), (px + 20, y)); y += 40

        if R.field in ENUM_FIELDS:
            for i, opt in enumerate(R.allowed(R.field)):
                screen.blit(font.render(f"[{i+1}] {opt}", True, FG), (px + 20, y)); y += 26
            y += 8
            for key, lbl in (("u", "uncertain"), ("c", "cannot_determine")):
                screen.blit(font.render(f"[{key}] {lbl}", True, DIM), (px + 20, y)); y += 26
        else:
            yl, nl = BOOL_LABELS.get(R.field, ("true", "false"))
            for key, lbl in (("y", yl), ("n", nl),
                             ("u", "uncertain"), ("c", "cannot_determine")):
                screen.blit(font.render(f"[{key}] {lbl}", True, FG), (px + 20, y)); y += 26

        y = H - 120
        for line in ("] [ next/prev spec   . , next/prev field",
                     "z undo   wheel zoom · drag pan · r reset",
                     "TAB note   ESC save+quit"):
            screen.blit(small.render(line, True, DIM), (px + 20, y)); y += 22

        if R.note_mode:
            box = pygame.Rect(px + 14, 200, PANEL_W - 28, 80)
            pygame.draw.rect(screen, (50, 54, 62), box)
            pygame.draw.rect(screen, ACCENT, box, 2)
            screen.blit(small.render("note:", True, DIM), (box.x + 8, box.y + 6))
            screen.blit(font.render(R.note_buf[-34:] + "_", True, FG),
                        (box.x + 8, box.y + 30))

        # loud field-change banner (clears on first answer in the field)
        if R.field_banner:
            is_enum = R.field in ENUM_FIELDS
            scheme = "ENUM — keys 1..N" if is_enum else "BOOLEAN — y / n"
            bcol = WARN if is_enum else ACCENT
            bh = 110
            band = pygame.Surface((pane.w, bh), pygame.SRCALPHA)
            band.fill((20, 22, 26, 235))
            screen.blit(band, (0, pane.centery - bh // 2))
            pygame.draw.rect(screen, bcol,
                             (0, pane.centery - bh // 2, pane.w, bh), 3)
            t1 = big.render(f"\u25b6 NEW FIELD:  {short(R.field)}", True, bcol)
            t2 = font.render(scheme + "    (u uncertain · c cannot_determine)", True, FG)
            screen.blit(t1, (pane.centerx - t1.get_width() // 2, pane.centery - 40))
            screen.blit(t2, (pane.centerx - t2.get_width() // 2, pane.centery + 4))

        pygame.display.flip()
        clock.tick(60)

    R.save()
    pygame.quit()
    n_done = sum(1 for s in R.specs
                 if all(R.v[s["gbifID"]].get(f, "") for f in MANUAL_FIELDS))
    print(f"saved {args.out} — {n_done}/{len(R.specs)} specimens fully reviewed")


if __name__ == "__main__":
    main()
