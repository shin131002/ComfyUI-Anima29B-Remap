"""
lora_remap_extended_anima.py

EXPERIMENTAL variant of AnimaLoRARemapTagLoader. Same base remapping
behavior, but generalizes how the LoRA's effect is projected onto
Anima-2.9B's 12 newly-inserted layers: instead of always copying from the
single "source" layer recorded in expand_manifest.json (which is always
the immediately-PRECEDING old layer), this node blends between the
preceding ("front") and following ("back") old layers using `blend_ratio`.

    blended_value = front_layer_value * blend_ratio + back_layer_value * (1 - blend_ratio)

At blend_ratio = 1.0, this is mathematically identical to the original
node's extend_to_new_layers behavior (100% front, 0% back). Lower values
progressively mix in the following layer instead.

This is a SEPARATE node (not a modification of lora_remap_anima.py) so
the original, simpler node keeps working exactly as before.
"""

import logging
import os

import folder_paths
import comfy.utils
import comfy.sd

from .anima_common import (
    find_block_indices,
    remap_key,
    split_block_key,
    list_available_manifests,
    load_manifest,
    build_base_to_target,
    build_insertion_neighbors,
    get_model_block_count,
)
from .lora_remap_anima import (
    resolve_lora_path,
    save_remapped_lora,
    parse_lora_tags,
)

logger = logging.getLogger("AnimaLoRARemapExtended")

REMAP_SUFFIX_EXT = "_29Bremap_ext"


def get_remapped_sibling_path_ext(original_path):
    base, ext = os.path.splitext(original_path)
    return f"{base}{REMAP_SUFFIX_EXT}{ext}"


def group_by_base_index(lora_sd):
    """{base_idx: {suffix: (prefix, sep, tensor)}} for every MAIN-block tensor."""
    groups = {}
    for k, v in lora_sd.items():
        parsed = split_block_key(k)
        if parsed is None:
            continue
        prefix, base_idx, suffix, sep = parsed
        groups.setdefault(base_idx, {})[suffix] = (prefix, sep, v)
    return groups


def build_blended_extension(groups, neighbors, blend_ratio):
    """
    For each inserted target layer, blend its front/back neighbor tensors
    (matched by subkey suffix) using blend_ratio. Returns {new_key: tensor}.
    """
    extended = {}
    for t_idx, (prev_base, next_base) in neighbors.items():
        prev_group = groups.get(prev_base, {}) if prev_base is not None else {}
        next_group = groups.get(next_base, {}) if next_base is not None else {}
        all_suffixes = set(prev_group) | set(next_group)
        for suffix in all_suffixes:
            prev_entry = prev_group.get(suffix)
            next_entry = next_group.get(suffix)
            if prev_entry and next_entry:
                prefix, sep, v_prev = prev_entry
                _, _, v_next = next_entry
                blended = v_prev * blend_ratio + v_next * (1.0 - blend_ratio)
            elif prev_entry:
                prefix, sep, v_prev = prev_entry
                blended = v_prev * blend_ratio
            elif next_entry:
                prefix, sep, v_next = next_entry
                blended = v_next * (1.0 - blend_ratio)
            else:
                continue
            new_key = f"{prefix}{t_idx}{sep}{suffix}"
            extended[new_key] = blended
    return extended


class AnimaLoRARemapExtendedTagLoader:
    """
    EXPERIMENTAL: same as AnimaLoRARemapTagLoader, but the extension onto
    Anima-2.9B's 12 newly-inserted layers uses a continuous front/back
    blend_ratio instead of always copying from the single preceding layer.
    blend_ratio=1.0 reproduces the original node's behavior exactly.
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
                "blend_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
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
    CATEGORY = "loaders/anima/experimental"

    def load(self, model, text, default_weight, weight_multiplier, auto_remap, save_remapped,
              extend_to_new_layers, blend_ratio, extend_strength, manifest, clip=None):
        tags, stripped_text = parse_lora_tags(text, default_weight)

        manifest_data = load_manifest(manifest) if auto_remap else None
        base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
        neighbors = build_insertion_neighbors(manifest_data) if manifest_data else {}
        old_block_count = manifest_data.get("old_block_count") if manifest_data else None

        out_model = model
        out_clip = clip

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
                cached_path = resolve_lora_path(f"{name}{REMAP_SUFFIX_EXT}")
                if cached_path is not None:
                    lora_sd = comfy.utils.load_torch_file(cached_path, safe_load=True)
                    logger.info(f"'{name}': using cached extended-remap file {cached_path}")
                    logger.info(
                        f"'{name}': NOTE -- this cache reflects whatever extend_to_new_layers/"
                        f"blend_ratio/extend_strength were set to when it was SAVED, not the "
                        f"current node settings. Delete the file and re-run if you've changed these."
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

                        if extend_to_new_layers and neighbors:
                            groups = group_by_base_index(lora_sd)
                            extension = build_blended_extension(groups, neighbors, blend_ratio)
                            for k, v in extension.items():
                                remapped[k] = v * extend_strength
                            logger.info(
                                f"'{name}': [experimental] extended {len(extension)} tensors onto "
                                f"newly-inserted layers via front/back blend "
                                f"(blend_ratio={blend_ratio}, extend_strength={extend_strength})"
                            )

                        lora_sd = remapped

                        if save_remapped:
                            remapped_path = get_remapped_sibling_path_ext(original_path)
                            if os.path.exists(remapped_path):
                                logger.info(f"'{name}': extended-remap cache already exists, skipping save")
                            else:
                                try:
                                    save_remapped_lora(remapped_path, lora_sd)
                                    logger.info(f"'{name}': saved extended-remap copy to {remapped_path}")
                                except Exception as e:
                                    logger.warning(f"'{name}': failed to save extended-remap copy: {e}")
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
    "AnimaLoRARemapExtendedTagLoader": AnimaLoRARemapExtendedTagLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRARemapExtendedTagLoader": "Anima LoRA Tag Loader Extended (Experimental)",
}
