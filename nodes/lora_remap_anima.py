"""
lora_remap_anima.py

A LoRA Tag-style loader (`<lora:name:weight>` syntax embedded in a text
string, same convention as Power Lora Loader / LoRA Tag Power Loader) for
Anima models, with AUTOMATIC block-index remapping when an old-Anima
(28-block) LoRA is applied to Anima-2.9B (40-block, LLaMA Pro-style block
expansion).

How the auto-remap decision is made:
    1. Detect how many transformer blocks the connected MODEL has, by
       scanning its state_dict for ".blocks.<N>." keys (same technique
       ComfyUI core itself uses for Cosmos-Predict2/Anima block-count
       detection -- see Comfy-Org/ComfyUI PR #15555).
    2. Detect how many blocks the LoRA file itself was trained against,
       by scanning its own keys the same way.
    3. If the model has MORE blocks than the LoRA was trained for, the
       LoRA is remapped using the official expand_manifest.json
       (base -> target block index), derived as follows:
         - expand_manifest.json lists which TARGET block indices are the
           newly-inserted ones ("insertion_positions").
         - Every other target index, taken in ascending order, is one of
           the original (frozen, unchanged) base blocks, in order.
         - So base index i maps to the i-th target index NOT in
           insertion_positions.
    4. If the model has the same (or fewer) blocks as the LoRA, or no
       official manifest is found, the LoRA is applied unmodified.

If your LoRA's key naming doesn't match either of the two patterns below
(dot style `net.blocks.N.` or kohya-style `..._blocks_N_...`), it won't be
detected -- extend BLOCK_PATTERNS in that case.
"""

import logging
import os
import re

import folder_paths
import comfy.utils
import comfy.sd
from safetensors.torch import save_file as st_save_file

from .anima_common import (
    find_block_indices,
    remap_key,
    remap_key_to_target,
    list_available_manifests,
    load_manifest,
    build_base_to_target,
    build_source_to_inserted_targets,
    get_model_block_count,
)

logger = logging.getLogger("AnimaLoRARemap")

REMAP_SUFFIX = "_29Bremap"


def get_remapped_sibling_path(original_path):
    """Path for the cached remapped copy: same folder, same extension, suffix appended before it."""
    base, ext = os.path.splitext(original_path)
    return f"{base}{REMAP_SUFFIX}{ext}"


def save_remapped_lora(path, tensors):
    """Write a remapped LoRA state dict to disk as safetensors. Clones tensors to
    avoid safetensors' 'shared memory' errors on views/slices from the source file."""
    safe_tensors = {k: v.clone().contiguous() for k, v in tensors.items()}
    st_save_file(safe_tensors, path)


# ---------------------------------------------------------------------------
# `<lora:name:weight[:clip_weight]>` tag parsing
# ---------------------------------------------------------------------------

TAG_PATTERN = re.compile(r"<lora:([^:>]+):(-?[\d.]+)(?::(-?[\d.]+))?>")

LORA_EXTENSIONS = [".safetensors", ".pt", ".ckpt", ".sft"]


def resolve_lora_path(name):
    """
    Resolve a LoRA tag name to a full path, tolerating:
      - the extension being omitted (folder_paths.get_full_path needs it exact)
      - the file living in a subfolder of loras/ (matched by basename)
      - case differences in the extension or basename
    Returns None if nothing matches.
    """
    # 1. exact relative-path match (handles names that already include a subfolder/extension)
    path = folder_paths.get_full_path("loras", name)
    if path:
        return path

    # 2. try appending common extensions
    for ext in LORA_EXTENSIONS:
        path = folder_paths.get_full_path("loras", name + ext)
        if path:
            return path

    # 3. fall back to a basename search across every known lora file (any subfolder)
    try:
        all_loras = folder_paths.get_filename_list("loras")
    except Exception:
        all_loras = []

    target = name.lower()
    for rel_path in all_loras:
        base = os.path.splitext(os.path.basename(rel_path))[0]
        if base.lower() == target:
            return folder_paths.get_full_path("loras", rel_path)

    return None


def parse_lora_tags(text, default_weight):
    tags = []
    for m in TAG_PATTERN.finditer(text or ""):
        name = m.group(1).strip()
        w_model = float(m.group(2)) if m.group(2) else default_weight
        w_clip = float(m.group(3)) if m.group(3) else w_model
        tags.append((name, w_model, w_clip))
    stripped = TAG_PATTERN.sub("", text or "")
    stripped = re.sub(r"[ \t]{2,}", " ", stripped).strip()
    return tags, stripped


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AnimaLoRARemapTagLoader:
    """
    LoRA Tag-style loader for Anima. Parses `<lora:name:weight>` tags out of
    a text input (same syntax as Power Lora Loader / LoRA Tag Power Loader),
    and -- if the connected MODEL has more transformer blocks than the LoRA
    was trained for -- automatically remaps the LoRA's block indices using
    the bundled expand_manifest.json before applying it. Old-Anima LoRAs
    used on old-Anima models, or LoRAs already trained on Anima-2.9B, pass
    through unchanged.

    Experimental: when extend_to_new_layers is enabled, the LoRA's effect is
    also projected onto Anima-2.9B's newly-inserted layers, by copying each
    inserted layer's nearest-neighbor source layer's (already-remapped) delta
    onto it, scaled by extend_strength. There is no "correct" answer for what
    a pre-2.9B LoRA should do on layers that didn't exist when it was
    trained -- this is a best-effort approximation, off by default.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_available_manifests() or ["(none found)"]
        return {
            "required": {
                "model": ("MODEL",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "default_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "weight_multiplier": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "auto_remap": ("BOOLEAN", {"default": True}),
                "save_remapped": ("BOOLEAN", {"default": False}),
                "extend_to_new_layers": ("BOOLEAN", {"default": False}),
                "extend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
                "manifest": (manifests,),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "text")
    FUNCTION = "load"
    CATEGORY = "loaders/anima"

    def load(self, model, text, default_weight, weight_multiplier, auto_remap, save_remapped,
              extend_to_new_layers, extend_strength, manifest, clip=None):
        tags, stripped_text = parse_lora_tags(text, default_weight)

        manifest_data = load_manifest(manifest) if auto_remap else None
        base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
        source_to_inserted = build_source_to_inserted_targets(manifest_data) if manifest_data else {}
        old_block_count = manifest_data.get("old_block_count") if manifest_data else None

        out_model = model
        out_clip = clip

        # Computed once: whether the connected MODEL is the expanded (2.9B-style)
        # architecture at all. If not, the whole remap-cache mechanism is skipped
        # so a stale "_29Bremap" file is never applied to a non-expanded model.
        model_block_count = get_model_block_count(out_model) if auto_remap else None
        is_expanded_model = (
            auto_remap
            and model_block_count is not None
            and old_block_count is not None
            and model_block_count > old_block_count
        )

        for name, w_model, w_clip in tags:
            original_path = resolve_lora_path(name)
            if original_path is None:
                logger.warning(
                    f"LoRA not found in loras folder (tried exact name, common "
                    f"extensions, and a subfolder basename search): {name}"
                )
                continue

            if is_expanded_model and base_to_target:
                cached_path = resolve_lora_path(f"{name}{REMAP_SUFFIX}")
                if cached_path is not None:
                    lora_sd = comfy.utils.load_torch_file(cached_path, safe_load=True)
                    logger.info(f"'{name}': using cached remap file {os.path.basename(cached_path)}")
                    logger.info(
                        f"'{name}': NOTE -- this cache reflects whatever extend_to_new_layers/"
                        f"extend_strength were set to when it was SAVED, not the current node "
                        f"settings (extend_to_new_layers={extend_to_new_layers}, "
                        f"extend_strength={extend_strength}). Delete "
                        f"{os.path.basename(cached_path)} and re-run if you've changed these."
                    )
                else:
                    lora_sd = comfy.utils.load_torch_file(original_path, safe_load=True)
                    lora_indices = find_block_indices(lora_sd.keys())
                    lora_max_block = max(lora_indices) if lora_indices else None
                    needs_remap = (
                        lora_max_block is not None
                        and model_block_count > (lora_max_block + 1)
                    )

                    if needs_remap:
                        remapped = {}
                        dropped = 0
                        for k, v in lora_sd.items():
                            new_k, _ = remap_key(k, base_to_target)
                            if new_k is None:
                                dropped += 1
                                continue
                            remapped[new_k] = v
                        logger.info(
                            f"'{name}': remapped for {model_block_count}-block model "
                            f"({len(remapped)} tensors kept, {dropped} dropped as "
                            f"newly-inserted layers with no old counterpart)"
                        )

                        if extend_to_new_layers and source_to_inserted:
                            extended = 0
                            for k, v in lora_sd.items():
                                _, base_idx = remap_key(k, base_to_target)
                                if base_idx is None:
                                    continue
                                target_list = source_to_inserted.get(base_idx)
                                if not target_list:
                                    continue
                                for t_idx in target_list:
                                    new_k = remap_key_to_target(k, t_idx)
                                    if new_k is None:
                                        continue
                                    remapped[new_k] = v * extend_strength
                                    extended += 1
                            logger.info(
                                f"'{name}': [experimental] extended {extended} tensors onto "
                                f"newly-inserted layers via nearest-neighbor copy "
                                f"(strength={extend_strength})"
                            )

                        lora_sd = remapped

                        if save_remapped:
                            remapped_path = get_remapped_sibling_path(original_path)
                            if os.path.exists(remapped_path):
                                logger.info(
                                    f"'{name}': remap cache already exists, skipping save: "
                                    f"{os.path.basename(remapped_path)}"
                                )
                            else:
                                try:
                                    save_remapped_lora(remapped_path, lora_sd)
                                    logger.info(
                                        f"'{name}': saved remapped copy to {os.path.basename(remapped_path)}"
                                    )
                                except Exception as e:
                                    logger.warning(f"'{name}': failed to save remapped copy: {e}")
                    else:
                        logger.info(f"'{name}': applied as-is (already matches model block count)")
            else:
                lora_sd = comfy.utils.load_torch_file(original_path, safe_load=True)
                logger.info(f"'{name}': applied as-is (remap not applicable/enabled)")

            w_model_final = w_model * weight_multiplier
            w_clip_final = w_clip * weight_multiplier

            out_model, out_clip = comfy.sd.load_lora_for_models(
                out_model, out_clip, lora_sd, w_model_final, w_clip_final
            )

        return (out_model, out_clip, stripped_text)


NODE_CLASS_MAPPINGS = {
    "AnimaLoRARemapTagLoader": AnimaLoRARemapTagLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRARemapTagLoader": "Anima LoRA Tag Loader (Auto Remap)",
}
