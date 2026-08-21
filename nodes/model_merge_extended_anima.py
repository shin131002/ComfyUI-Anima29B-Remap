"""
model_merge_extended_anima.py

EXPERIMENTAL variant of AnimaModelMerge. Same base merge behavior for the
shared (old) 28 blocks, but generalizes how the 28-block model's weights
are projected onto Anima-2.9B's 12 newly-inserted layers: instead of always
using the single "source" layer recorded in expand_manifest.json (always
the immediately-PRECEDING old layer), this node blends between the
preceding ("front") and following ("back") old layers using `blend_ratio`
before applying extend_ratio.

    old_model_blended  = front_layer * blend_ratio + back_layer * (1 - blend_ratio)
    new_layer (final)  = (1 - extend_ratio) * 2.9B's own value + extend_ratio * old_model_blended

At blend_ratio = 1.0, this is mathematically identical to the original
node's extend_ratio behavior (100% front, 0% back). Lower values
progressively mix in the following layer instead.

This is a SEPARATE node (not a modification of model_merge_anima.py) so
the original, simpler node keeps working exactly as before.
"""

import logging

from .anima_common import (
    remap_key,
    split_block_key,
    list_available_manifests,
    load_manifest,
    build_base_to_target,
    build_insertion_neighbors,
    get_model_block_count,
)
from .model_merge_anima import remap_key_patches

logger = logging.getLogger("AnimaModelMergeExtended")


def group_patches_by_base_index(key_patches):
    """{base_idx: {suffix: (prefix, sep, patch)}} for every MAIN-block patch."""
    groups = {}
    for k, v in key_patches.items():
        parsed = split_block_key(k)
        if parsed is None:
            continue
        prefix, base_idx, suffix, sep = parsed
        groups.setdefault(base_idx, {})[suffix] = (prefix, sep, v)
    return groups


def build_front_back_extension_patches(groups, neighbors):
    """
    For each inserted target layer, collect its front (preceding) and back
    (following) neighbor patches SEPARATELY, re-keyed onto the new target
    index, WITHOUT any arithmetic on the patch values themselves (patches
    from get_key_patches() are ComfyUI's internal patch format, not raw
    tensors, so they can't be scalar-multiplied directly -- blending has to
    happen via add_patches()'s own strength_patch/strength_model instead).
    Returns (front_patches, back_patches), each {new_key: patch}.
    """
    front_patches = {}
    back_patches = {}
    for t_idx, (prev_base, next_base) in neighbors.items():
        if prev_base is not None:
            for suffix, (prefix, sep, patch) in groups.get(prev_base, {}).items():
                front_patches[f"{prefix}{t_idx}{sep}{suffix}"] = patch
        if next_base is not None:
            for suffix, (prefix, sep, patch) in groups.get(next_base, {}).items():
                back_patches[f"{prefix}{t_idx}{sep}{suffix}"] = patch
    return front_patches, back_patches


def apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio):
    """
    Apply the front/back-blended extension onto `m` in place, using two
    sequential add_patches() calls so the actual blending math is done by
    ComfyUI itself (never by us touching patch values directly):

        1st call (front): new = (1 - extend_ratio) * current + (extend_ratio * blend_ratio) * front
        2nd call (back):  new = 1.0 * current + (extend_ratio * (1 - blend_ratio)) * back

    If only one side exists for a given key (edge case), the "shrink" by
    (1 - extend_ratio) is applied on whichever call actually runs first.
    Returns the total number of tensors touched (front + back).
    """
    front_patches, back_patches = build_front_back_extension_patches(groups, neighbors)

    shrink_applied = False
    if front_patches:
        m.add_patches(front_patches, extend_ratio * blend_ratio, 1.0 - extend_ratio)
        shrink_applied = True
    if back_patches:
        shrink = 1.0 if shrink_applied else (1.0 - extend_ratio)
        m.add_patches(back_patches, extend_ratio * (1.0 - blend_ratio), shrink)

    return len(front_patches) + len(back_patches)


class AnimaModelMergeExtended:
    """
    EXPERIMENTAL: same as AnimaModelMerge, but the extension onto
    Anima-2.9B's 12 newly-inserted layers uses a continuous front/back
    blend_ratio (applied to the 28-block model's neighboring layers)
    instead of always using the single preceding layer. blend_ratio=1.0
    reproduces the original node's extend_ratio behavior exactly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_available_manifests() or ["(none found)"]
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "extend_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "manifest": (manifests,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "loaders/anima/experimental"

    def merge(self, model_1, model_2, merge_ratio, extend_ratio, blend_ratio, manifest):
        manifest_data = load_manifest(manifest)
        base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
        neighbors = build_insertion_neighbors(manifest_data) if manifest_data else {}
        old_block_count = manifest_data.get("old_block_count") if manifest_data else None
        new_block_count = manifest_data.get("new_block_count") if manifest_data else None

        block_count_1 = get_model_block_count(model_1)
        block_count_2 = get_model_block_count(model_2)
        logger.info(f"model_1: {block_count_1} blocks, model_2: {block_count_2} blocks, merge_ratio={merge_ratio}")

        # --- Case 1: same architecture on both sides -> direct merge, no remap ---
        if block_count_1 is not None and block_count_1 == block_count_2:
            m = model_1.clone()
            kp2 = model_2.get_key_patches("diffusion_model.")
            m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
            logger.info(f"Direct {block_count_1}-block merge (no remap needed). Output: {block_count_1} blocks.")
            return (m,)

        # --- Case 2: mismatched architectures -> one should be the expanded (2.9B) one ---
        is_1_expanded = new_block_count is not None and block_count_1 == new_block_count
        is_2_expanded = new_block_count is not None and block_count_2 == new_block_count
        is_1_old = old_block_count is not None and block_count_1 == old_block_count
        is_2_old = old_block_count is not None and block_count_2 == old_block_count

        if is_1_expanded and is_2_old and base_to_target:
            m = model_1.clone()
            kp2 = model_2.get_key_patches("diffusion_model.")
            remapped_kp2, dropped = remap_key_patches(kp2, base_to_target)
            m.add_patches(remapped_kp2, 1.0 - merge_ratio, merge_ratio)
            logger.info(
                f"model_1=2.9B(40), model_2=old(28): remapped {len(remapped_kp2)} keys "
                f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio}."
            )
            if extend_ratio > 0.0 and neighbors:
                groups = group_patches_by_base_index(kp2)
                n_touched = apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio)
                logger.info(
                    f"[experimental] extended {n_touched} tensors from model_2 (old) onto "
                    f"newly-inserted layers via front/back blend "
                    f"(blend_ratio={blend_ratio}, extend_ratio={extend_ratio})"
                )
            return (m,)

        if is_2_expanded and is_1_old and base_to_target:
            m = model_2.clone()
            kp1 = model_1.get_key_patches("diffusion_model.")
            remapped_kp1, dropped = remap_key_patches(kp1, base_to_target)
            m.add_patches(remapped_kp1, merge_ratio, 1.0 - merge_ratio)
            logger.info(
                f"model_1=old(28), model_2=2.9B(40): remapped {len(remapped_kp1)} keys "
                f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio}."
            )
            if extend_ratio > 0.0 and neighbors:
                groups = group_patches_by_base_index(kp1)
                n_touched = apply_blended_extension(m, groups, neighbors, blend_ratio, extend_ratio)
                logger.info(
                    f"[experimental] extended {n_touched} tensors from model_1 (old) onto "
                    f"newly-inserted layers via front/back blend "
                    f"(blend_ratio={blend_ratio}, extend_ratio={extend_ratio})"
                )
            return (m,)

        # --- Fallback: unexpected/unrecognized block counts -> best-effort direct merge ---
        logger.warning(
            f"Unrecognized block-count combination (model_1={block_count_1}, "
            f"model_2={block_count_2}, manifest old={old_block_count}/new={new_block_count}). "
            f"Falling back to an unremapped direct merge -- results may be incorrect if the "
            f"architectures actually differ."
        )
        m = model_1.clone()
        kp2 = model_2.get_key_patches("diffusion_model.")
        m.add_patches(kp2, 1.0 - merge_ratio, merge_ratio)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AnimaModelMergeExtended": AnimaModelMergeExtended,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelMergeExtended": "Anima Model Merge Extended (Experimental)",
}
