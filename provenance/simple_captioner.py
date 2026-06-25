"""
Simplified herbarium captioner (recaption package).

A trimmed clone of the original captioner.py. Keeps only what the re-caption
pass needs:
    caption()  - Send an image to a VLM endpoint and return a validated dict.
    validate() - Validate a dict against the simplified JSON schema, including
                 phenology mutual-exclusivity rules.
    load_relaxed_families() - Load the family allowlist for relaxed cross-clade
                 checking.

Removed relative to the original: to_dwc(), get_ppo_lookup(), and all the
PO/PPO/DwC mapping tables (deterministic; re-derivable later from the stored
phenology booleans).

Exception split
---------------
    CaptionError           - transport / protocol failures (network, non-200,
                             unparseable response envelope). Escalating the VLM
                             will not fix these.
    CaptionValidationError - the model produced content that is not valid JSON
                             or fails schema / consistency checks. A subclass of
                             CaptionError, so existing `except CaptionError`
                             still catches it; callers that want to escalate
                             sampling can catch the subclass specifically.

Prompt, schema, and the relaxed-families list are loaded relative to THIS
module's directory, so the recaption folder is self-contained.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

_MODULE_DIR = Path(__file__).resolve().parent
_PROMPT_PATH = _MODULE_DIR / "caption_prompt.txt"
_SCHEMA_PATH = _MODULE_DIR / "schema.json"
_RELAXED_FAMILIES_PATH = _MODULE_DIR / "relaxed_families.txt"


class CaptionError(ValueError):
    """Transport / protocol error (network, non-200, bad response envelope)."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class CaptionValidationError(CaptionError):
    """Model-content error: VLM output is not valid JSON, or fails the schema /
    consistency checks. Subclass of CaptionError so broad handlers still catch
    it; escalation logic catches this type specifically."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def load_relaxed_families(path: Path | None = None) -> frozenset[str]:
    """Load the cross-clade-relaxed family allowlist (lowercased, stripped).

    Returns an empty set if the file is absent. Blank lines and '#' comments
    are ignored.
    """
    p = path or _RELAXED_FAMILIES_PATH
    if not p.exists():
        return frozenset()
    fams: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        fams.add(s.lower())
    return frozenset(fams)


# ---------------------------------------------------------------------------
# Lightweight JSON Schema validator (subset of Draft-07; no external dep)
# ---------------------------------------------------------------------------

def _check_required_keys(obj: dict[str, Any], required: list[str], path: str = "") -> list[str]:
    errors: list[str] = []
    for key in required:
        if key not in obj:
            label = f"{path}.{key}" if path else key
            errors.append(f"Missing required field: {label}")
    return errors


def _check_enum(value: Any, allowed: list[str], path: str) -> list[str]:
    if value not in allowed:
        return [f"Validation failed: {path} value {value!r} not in allowed enums: {allowed}"]
    return []


def _check_type(value: Any, expected_type: type, path: str) -> list[str]:
    if not isinstance(value, expected_type):
        return [f"Type error: {path} expected {expected_type.__name__}, got {type(value).__name__}"]
    return []


def _validate_object(obj: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    """Recursively validate obj against schema (object/required/properties/
    additionalProperties + string enums + boolean types). Nested objects (the
    phenology boolean block) validate recursively."""
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type == "boolean":
        if not isinstance(obj, bool):
            errors.extend(_check_type(obj, bool, path))
        return errors

    if expected_type == "string":
        if not isinstance(obj, str):
            errors.extend(_check_type(obj, str, path))
        if isinstance(obj, str) and "enum" in schema:
            errors.extend(_check_enum(obj, schema["enum"], path))
        return errors

    if expected_type == "object":
        if not isinstance(obj, dict):
            return [f"Expected object at {path or 'root'}, got {type(obj).__name__}"]
        if "required" in schema:
            errors.extend(_check_required_keys(obj, schema["required"], path))
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in obj:
                child_path = f"{path}.{prop_name}" if path else prop_name
                errors.extend(_validate_object(obj[prop_name], prop_schema, child_path))
        if schema.get("additionalProperties") is False:
            extra = set(obj.keys()) - set(properties.keys())
            for key in sorted(extra):
                errors.append(f"Unknown field: {path}.{key}" if path else f"Unknown field: {key}")

    return errors


# ---------------------------------------------------------------------------
# Cross-field consistency (not expressible in the per-field schema)
# ---------------------------------------------------------------------------
# Reproductive "modes" are mutually exclusive at the specimen level across
# clades; WITHIN a mode, co-occurrence is real and allowed (flower+fruit;
# pollen_cone+seed_cone on monoecious conifers).
_PHENO_MODES: dict[str, tuple[str, ...]] = {
    "angiosperm": ("flower", "fruit"),
    "gymnosperm": ("pollen_cone", "seed_cone"),
    "cryptogam":  ("sporulating",),
}
_PHENO_SPECIFIC: tuple[str, ...] = (
    "flower", "fruit", "pollen_cone", "seed_cone", "sporulating",
)
_PHENO_ALL: tuple[str, ...] = _PHENO_SPECIFIC + ("reproductive_unknown",)


def _check_cross_field_consistency(
    caption_dict: dict[str, Any],
    family: str | None = None,
    relaxed_families: frozenset[str] = frozenset(),
) -> list[str]:
    """Couplings the per-field schema cannot express.

    1. foliage / foliage_type coupling.
    2. reproductive_unknown is exclusive with every specific phenology flag
       (definitional — NEVER relaxed by family).
    3. cross-clade phenology exclusivity: flags may not span more than one of
       {angiosperm, gymnosperm, cryptogam}. RELAXED when `family` (normalized)
       is in `relaxed_families`. Records with null/missing family fail closed.
    """
    errors: list[str] = []
    structures = caption_dict.get("structures")
    if not isinstance(structures, dict):
        return errors  # schema pass reports the structural problem

    # --- 1. foliage coupling ---
    foliage = structures.get("foliage")
    foliage_type = structures.get("foliage_type")
    if foliage == "absent" and foliage_type not in (None, "none"):
        errors.append(
            "Cross-field consistency: structures.foliage_type must be 'none' "
            f"when structures.foliage='absent' (got {foliage_type!r})"
        )
    if foliage == "present" and foliage_type == "none":
        errors.append(
            "Cross-field consistency: structures.foliage_type must not be 'none' "
            "when structures.foliage='present'"
        )

    # --- phenology rules (only when the block is well-formed; else schema pass) ---
    ph = structures.get("phenology")
    if not isinstance(ph, dict):
        return errors
    flags = {k: ph.get(k) for k in _PHENO_ALL}
    if any(not isinstance(v, bool) for v in flags.values()):
        return errors  # missing/typed-wrong flag -> schema pass handles it

    # --- 2. reproductive_unknown exclusivity (definitional, never relaxed) ---
    specific_true = [k for k in _PHENO_SPECIFIC if flags[k]]
    if flags["reproductive_unknown"] and specific_true:
        errors.append(
            "Cross-field consistency: phenology.reproductive_unknown must not "
            f"co-occur with specific flags {specific_true} (set the specific "
            "flag(s) and leave reproductive_unknown=false)"
        )

    # --- 3. cross-clade exclusivity (relaxable by family allowlist) ---
    active_modes = [m for m, fs in _PHENO_MODES.items() if any(flags[f] for f in fs)]
    if len(active_modes) > 1:
        fam_norm = (family or "").strip().lower()
        if fam_norm not in relaxed_families:
            errors.append(
                "Cross-field consistency: phenology flags span multiple clades "
                f"{active_modes} (specimen-level mutually exclusive; "
                f"family={family!r} not in relaxed allowlist)"
            )

    return errors


def validate(
    caption_dict: dict[str, Any],
    family: str | None = None,
    relaxed_families: frozenset[str] = frozenset(),
) -> tuple[bool, list[str]]:
    """Validate a caption dict against the simplified schema + consistency rules.

    `family` + `relaxed_families` govern only the cross-clade phenology check;
    all other checks are unconditional. Returns (ok, errors)."""
    schema = _load_schema()
    errors = _validate_object(caption_dict, schema)
    errors.extend(_check_cross_field_consistency(caption_dict, family, relaxed_families))
    return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# caption()
# ---------------------------------------------------------------------------

def caption(
    image_path: str,
    endpoint: str,
    api_key: str | None = None,
    *,
    reasoning_effort: str = "off",
    timeout: int = 60,
    context: str | None = None,
    family: str | None = None,
    relaxed_families: frozenset[str] = frozenset(),
    **kwargs: Any,
) -> dict[str, Any]:
    """Send an image to a VLM endpoint and return a validated caption dict.

    Raises:
        CaptionValidationError - VLM output not valid JSON, or fails schema /
            consistency checks (content problem; caller may escalate sampling).
        CaptionError - transport / protocol failure (caller should not escalate).
    """
    try:
        prompt_text = _load_prompt()
    except OSError as exc:
        raise CaptionError(f"Cannot load caption prompt: {exc}") from exc

    try:
        raw = Path(image_path).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        ext = Path(image_path).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime = mime_map.get(ext)
        if mime is None:
            supported = ", ".join(sorted(mime_map.keys()))
            raise CaptionError(f"Unsupported image format '{ext}'; supported: {supported}")
    except OSError as exc:
        raise CaptionError(f"Cannot read image file {image_path!r}: {exc}") from exc

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    user_content: list[dict[str, Any]] = []
    if context:
        user_content.append({"type": "text", "text": f"Specimen context: {context}"})
    user_content.append(
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    )

    payload: dict[str, Any] = {
        "model": kwargs.get("model", "gpt-4o"),
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 13928,
        "temperature": kwargs.get("temperature", 0.0),
    }

    payload["reasoning_format"] = "auto"
    _THINK_BUDGET = {"off": 0, "low": 512, "medium": 2048, "high": 8192}
    if reasoning_effort == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["thinking_budget_tokens"] = 0
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": True, "preserve_thinking": True}
        payload["thinking_budget_tokens"] = _THINK_BUDGET.get(reasoning_effort, 2048)
        payload["reasoning_control"] = True
        
    #print(payload)  ## WARNING Prints entire encoded image. Useful to know what you expect is going to the llm.
    # --- transport / protocol failures -> CaptionError ---
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise CaptionError(f"API request failed: {exc}") from exc

    if resp.status_code != 200:
        raise CaptionError(f"API returned status {resp.status_code}: {resp.text.strip()[:500]}")

    try:
        body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise CaptionError(f"Cannot parse VLM response envelope: {exc}") from exc

    # --- model-content failures -> CaptionValidationError ---
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    text = stripped.strip()

    try:
        caption_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CaptionValidationError(f"VLM output is not valid JSON: {exc}") from exc

    is_valid, errors = validate(caption_dict, family=family, relaxed_families=relaxed_families)
    if not is_valid:
        raise CaptionValidationError("Validation failed: " + "; ".join(errors))

    return caption_dict
