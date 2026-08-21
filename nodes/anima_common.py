"""
anima_common.py

Shared logic for the Anima-2.9B remap/merge node package:
  - block-key pattern matching (main net.blocks vs. llm_adapter.blocks)
  - expand_manifest.json loading and base->target block-index mapping
  - connected-MODEL block-count detection

Used by both lora_remap_anima.py and checkpoint_merge_anima.py so the two
nodes can never disagree about how blocks are detected or mapped.
"""

import json
import logging
import os
import re

logger = logging.getLogger("AnimaCommon")

# ---------------------------------------------------------------------------
# Block-key pattern handling
# ---------------------------------------------------------------------------

BLOCK_PATTERNS = [
    re.compile(r"(\.blocks\.)(\d+)(\.)"),   # e.g. net.blocks.12.self_attn...
    re.compile(r"(_blocks_)(\d+)(_)"),       # e.g. lora_unet_net_blocks_12_self_attn...
]


def _is_llm_adapter_key(key):
    """
    llm_adapter has its own separate, small block structure (6 blocks,
    unaffected by the 28->40 expansion of the main net.blocks). Any key
    containing "llm_adapter" must never be touched by the main-block
    remapping logic below, even though it also matches ".blocks.<N>.".
    """
    return "llm_adapter" in key


def find_block_indices(keys):
    """Return the set of MAIN transformer block indices referenced by any key in `keys`."""
    indices = set()
    for key in keys:
        if _is_llm_adapter_key(key):
            continue
        for pat in BLOCK_PATTERNS:
            m = pat.search(key)
            if m:
                indices.add(int(m.group(2)))
                break
    return indices


def remap_key(key, base_to_target):
    """
    Rewrite the MAIN block index in `key` using base_to_target (base_idx -> target_idx).
    Returns:
        (new_key, base_idx)  -- successfully remapped
        (None, base_idx)     -- a block index was found but has no mapping entry
                                 (i.e. it's a newly-inserted layer with no old
                                 counterpart) -> caller should drop this tensor
        (key, None)          -- no MAIN block index found (llm_adapter keys, or
                                 non-block keys like embedders/final_layer) -> untouched
    """
    if _is_llm_adapter_key(key):
        return key, None
    for pat in BLOCK_PATTERNS:
        m = pat.search(key)
        if m:
            base_idx = int(m.group(2))
            if base_idx not in base_to_target:
                return None, base_idx
            target_idx = base_to_target[base_idx]
            new_key = key[: m.start()] + m.group(1) + str(target_idx) + m.group(3) + key[m.end():]
            return new_key, base_idx
    return key, None


# ---------------------------------------------------------------------------
# expand_manifest.json handling
# ---------------------------------------------------------------------------

_MANIFEST_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mapping"))
_MANIFEST_CACHE = {}


def list_available_manifests():
    """Return manifest filenames found in the mapping/ folder, for a node's dropdown."""
    if not os.path.isdir(_MANIFEST_DIR):
        return []
    return sorted(f for f in os.listdir(_MANIFEST_DIR) if f.endswith(".json"))


def load_manifest(filename):
    if filename in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[filename]
    path = os.path.join(_MANIFEST_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"manifest not found: {path}")
        _MANIFEST_CACHE[filename] = None
        return None
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    _MANIFEST_CACHE[filename] = manifest
    return manifest


def build_base_to_target(manifest):
    """
    Derive {base_block_idx: target_block_idx} from expand_manifest.json.
    Every target index NOT in insertion_positions is one of the original
    (frozen) base blocks, in ascending order.
    """
    if manifest is None:
        return {}
    old_count = manifest["old_block_count"]
    new_count = manifest["new_block_count"]
    inserted = set(manifest["insertion_positions"])
    old_target_indices = [i for i in range(new_count) if i not in inserted]
    if len(old_target_indices) != old_count:
        logger.warning(
            f"manifest inconsistency: expected {old_count} non-inserted target "
            f"blocks, found {len(old_target_indices)}"
        )
    return {base_idx: target_idx for base_idx, target_idx in enumerate(old_target_indices)}


def build_source_to_inserted_targets(manifest):
    """
    Derive {base_block_idx: [inserted_target_idx, ...]} from expand_manifest.json's
    "inserted_to_source" (which maps each newly-inserted TARGET index to the BASE
    index it was deep-copied from at initialization). Used for the experimental
    "extend to new layers" feature: applying a base layer's LoRA delta onto the
    new layer(s) that were originally copied from it.
    """
    if manifest is None:
        return {}
    inserted_to_source = manifest.get("inserted_to_source", {})
    result = {}
    for target_idx_str, base_idx in inserted_to_source.items():
        result.setdefault(int(base_idx), []).append(int(target_idx_str))
    return result


def split_block_key(key):
    """
    Split a MAIN-block key into (prefix, base_idx, suffix, sep):
      prefix: everything up to and including e.g. "net.blocks."
      base_idx: the block index found in the key
      sep: the separator right after the index (e.g. "." or "_")
      suffix: everything after that separator (e.g. "self_attn.q_proj.lora_down.weight")
    Returns None for llm_adapter keys or keys with no block index at all.
    """
    if _is_llm_adapter_key(key):
        return None
    for pat in BLOCK_PATTERNS:
        m = pat.search(key)
        if m:
            prefix = key[: m.start()] + m.group(1)
            base_idx = int(m.group(2))
            sep = m.group(3)
            suffix = key[m.end():]
            return prefix, base_idx, suffix, sep
    return None


def build_insertion_neighbors(manifest):
    """
    For each newly-inserted target index, find the nearest OLD (non-inserted)
    block on each side, as BASE indices: {target_idx: (prev_base_idx, next_base_idx)}.
    Either side may be None near the ends of the network. The "prev" side always
    matches expand_manifest.json's "inserted_to_source" (confirmed by direct
    comparison) since the inserted layer was deep-copied from its immediate
    predecessor at initialization; "next" is the analogous lookup in the other
    direction, which the manifest does not record directly.
    """
    if manifest is None:
        return {}
    new_count = manifest["new_block_count"]
    inserted = set(manifest["insertion_positions"])
    old_target_indices = sorted(i for i in range(new_count) if i not in inserted)
    target_to_base = {t: b for b, t in enumerate(old_target_indices)}

    neighbors = {}
    for t_idx in sorted(inserted):
        prev_target = max((t for t in old_target_indices if t < t_idx), default=None)
        next_target = min((t for t in old_target_indices if t > t_idx), default=None)
        prev_base = target_to_base.get(prev_target) if prev_target is not None else None
        next_base = target_to_base.get(next_target) if next_target is not None else None
        neighbors[t_idx] = (prev_base, next_base)
    return neighbors


def remap_key_to_target(key, target_idx):
    """
    Force-rewrite a key's MAIN block index to `target_idx`, regardless of what
    index it currently has. Used to project an already-remapped base layer's
    tensor onto a newly-inserted layer. Returns None for llm_adapter keys or
    keys with no block index at all.
    """
    if _is_llm_adapter_key(key):
        return None
    for pat in BLOCK_PATTERNS:
        m = pat.search(key)
        if m:
            return key[: m.start()] + m.group(1) + str(target_idx) + m.group(3) + key[m.end():]
    return None


# ---------------------------------------------------------------------------
# Model block-count detection
# ---------------------------------------------------------------------------

def get_model_block_count(model_patcher):
    """Best-effort detection of how many transformer blocks the connected MODEL has."""
    sd = None
    for getter in (
        lambda: model_patcher.model_state_dict(),
        lambda: model_patcher.model.diffusion_model.state_dict(),
        lambda: model_patcher.model.state_dict(),
    ):
        try:
            sd = getter()
            if sd:
                break
        except Exception:
            continue
    if not sd:
        return None
    indices = find_block_indices(sd.keys())
    return (max(indices) + 1) if indices else None
