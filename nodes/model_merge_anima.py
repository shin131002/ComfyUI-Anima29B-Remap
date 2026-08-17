"""
model_merge_anima.py

Merge two Anima MODELs with automatic handling of the 28-block <-> 40-block
(Anima-2.9B) architecture mismatch.

Inputs are named model_1 (top) and model_2 (bottom) to match the visual
slot order in ComfyUI. merge_ratio is the weight of model_1:
    merge_ratio = 1.0  -> output's shared/old-block weights = 100% model_1
    merge_ratio = 0.0  -> output's shared/old-block weights = 100% model_2
    merge_ratio = 0.5  -> 50/50 blend

Architecture handling:
    - If both models have the SAME block count (both 28-block old-Anima
      derivatives, or both 40-block Anima-2.9B derivatives): a direct
      key-for-key merge is done, no remapping needed. Output has that
      same block count.
    - If block counts DIFFER (one 28-block, one 40-block): the output is
      always the 40-block architecture. The 40-block model supplies the
      base (so its 12 newly-inserted blocks -- which have no counterpart
      in a 28-block model -- are preserved untouched, at 100% their
      original value, regardless of merge_ratio). The 28-block model's
      weights are remapped onto their corresponding 40-block indices
      (via the bundled expand_manifest.json) and blended in at the old
      (shared) block positions according to merge_ratio.

Bypassing the node (ComfyUI's Mode: Bypass) falls back to ComfyUI's
default behavior for a two-MODEL-input / one-MODEL-output node: the
first matching input (model_1, the top slot) is passed straight through.
No special code is needed for that -- it's standard ComfyUI behavior.
"""

import logging

from .anima_common import (
    remap_key,
    list_available_manifests,
    load_manifest,
    build_base_to_target,
    get_model_block_count,
)

logger = logging.getLogger("AnimaModelMerge")


def remap_key_patches(key_patches, base_to_target):
    """Apply remap_key() across a dict of {key: patch} from ModelPatcher.get_key_patches()."""
    remapped = {}
    dropped = 0
    for k, v in key_patches.items():
        new_k, _ = remap_key(k, base_to_target)
        if new_k is None:
            dropped += 1
            continue
        remapped[new_k] = v
    return remapped, dropped


class AnimaModelMerge:
    """
    Merge two Anima models (MODEL) with merge_ratio as the weight of
    model_1 (top input). Automatically detects a 28-block/40-block
    (Anima-2.9B) architecture mismatch and remaps + outputs at 40 blocks
    in that case; otherwise merges directly at whatever block count both
    inputs share.
    """

    @classmethod
    def INPUT_TYPES(cls):
        manifests = list_available_manifests() or ["(none found)"]
        return {
            "required": {
                "model_1": ("MODEL",),
                "model_2": ("MODEL",),
                "merge_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "manifest": (manifests,),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "merge"
    CATEGORY = "loaders/anima"

    def merge(self, model_1, model_2, merge_ratio, manifest):
        manifest_data = load_manifest(manifest)
        base_to_target = build_base_to_target(manifest_data) if manifest_data else {}
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
            # model_1 (40-block) is the base -> its new 12 blocks are preserved untouched.
            m = model_1.clone()
            kp2 = model_2.get_key_patches("diffusion_model.")
            remapped_kp2, dropped = remap_key_patches(kp2, base_to_target)
            m.add_patches(remapped_kp2, 1.0 - merge_ratio, merge_ratio)
            logger.info(
                f"model_1=2.9B(40), model_2=old(28): remapped {len(remapped_kp2)} keys "
                f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio} "
                f"(model_1 weight). Output: 40 blocks."
            )
            return (m,)

        if is_2_expanded and is_1_old and base_to_target:
            # model_2 (40-block) is the base -> its new 12 blocks are preserved untouched.
            # merge_ratio still means "weight of model_1", so strengths are swapped
            # relative to the case above.
            m = model_2.clone()
            kp1 = model_1.get_key_patches("diffusion_model.")
            remapped_kp1, dropped = remap_key_patches(kp1, base_to_target)
            m.add_patches(remapped_kp1, merge_ratio, 1.0 - merge_ratio)
            logger.info(
                f"model_1=old(28), model_2=2.9B(40): remapped {len(remapped_kp1)} keys "
                f"({dropped} dropped), blended at old-block positions, ratio={merge_ratio} "
                f"(model_1 weight). Output: 40 blocks."
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
    "AnimaModelMerge": AnimaModelMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaModelMerge": "Anima Model Merge (Auto Remap)",
}
