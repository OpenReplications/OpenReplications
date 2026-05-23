"""Context Retrieval from MVMem (Paper Section 3.2, Equations 2-4).

Precomputes retrieval results over all scenes at once:
- Adaptive segment generation mode (Eq. 2)
- Contiguous scenes V_rel (Eq. 3)
- Relevant scenes S_rel (Eq. 3)
- Relevant references R_rel (Eq. 4)
"""

import json
import logging
from pathlib import Path

from src.memory.schema import MVMem, SegmentMemory, TextualStates
from src.models.mllm import call_mllm_json
from src.prompts.generation import (
    prompt_adaptive_segment_mode,
    prompt_contiguous_scenes,
    prompt_relevant_references,
    prompt_relevant_scenes,
)

logger = logging.getLogger(__name__)


def precompute_retrieval(mvmem: MVMem, scenes: list[str]) -> MVMem:
    """Precompute all retrieval contexts over all scenes.

    This runs Equations (2), (3), and (4) in batch.
    Paper Section 5.1: "we pre-compute Equations (2) and (3) and Equation (4)
    over all segments at once"
    """
    logger.info("Precomputing retrieval contexts...")

    # --- Eq. 2: Adaptive Segment Generation Mode ---
    logger.info("  Computing segment generation modes...")
    try:
        segments_result = call_mllm_json(prompt_adaptive_segment_mode(scenes))
        segment_groups = segments_result if isinstance(segments_result, list) else []
    except Exception as e:
        logger.warning(f"  Segment mode computation failed: {e}, using default (all extrapolation)")
        segment_groups = [[i] for i in range(len(scenes))]

    # Determine mode per scene based on segment groups
    # Scenes within a multi-scene group use interpolation (except last scene in group)
    # Single-scene groups or last scene in group use extrapolation
    scene_modes = {}
    for group in segment_groups:
        for idx, scene_idx in enumerate(group):
            if len(group) > 1 and idx < len(group) - 1:
                scene_modes[scene_idx] = "interpolation"
            else:
                scene_modes[scene_idx] = "extrapolation"
    # Last scene overall always extrapolation
    if scenes:
        scene_modes[len(scenes) - 1] = "extrapolation"

    logger.info(f"  Segment groups: {segment_groups}")
    logger.info(f"  Scene modes: {scene_modes}")

    # --- Eq. 3: Contiguous scenes (V_rel) ---
    logger.info("  Computing contiguous scenes...")
    try:
        cont_result = call_mllm_json(prompt_contiguous_scenes(scenes))
        contiguous = {}
        if isinstance(cont_result, dict):
            for k, v in cont_result.items():
                idx = int(k)
                contiguous[idx] = int(v) if v != "" and v is not None else None
        mvmem.contiguous_scenes = contiguous
    except Exception as e:
        logger.warning(f"  Contiguous scenes computation failed: {e}")
        mvmem.contiguous_scenes = {i: None for i in range(len(scenes))}

    logger.info(f"  Contiguous: {mvmem.contiguous_scenes}")

    # --- Eq. 3: Relevant scenes (S_rel) ---
    logger.info("  Computing relevant scenes...")
    try:
        rel_result = call_mllm_json(prompt_relevant_scenes(scenes))
        if isinstance(rel_result, dict) and "relevant_scenes" in rel_result:
            relevant = {}
            for k, v in rel_result["relevant_scenes"].items():
                relevant[int(k)] = v
            mvmem.relevant_scenes = relevant
        elif isinstance(rel_result, dict):
            mvmem.relevant_scenes = {int(k): v for k, v in rel_result.items()}
    except Exception as e:
        logger.warning(f"  Relevant scenes computation failed: {e}")
        mvmem.relevant_scenes = {}

    # --- Eq. 4: Relevant references (R_rel) ---
    logger.info("  Computing relevant references...")
    try:
        anchors = {r.name: r.caption for r in mvmem.references}
        ref_result = call_mllm_json(prompt_relevant_references(scenes, anchors))
        if isinstance(ref_result, dict):
            mvmem.relevant_refs = {int(k): v for k, v in ref_result.items()}
    except Exception as e:
        logger.warning(f"  Relevant references computation failed: {e}")
        mvmem.relevant_refs = {}

    # --- Initialize segment memories ---
    for i, scene in enumerate(scenes):
        seg = SegmentMemory(
            segment_index=i,
            scene_context=scene,
            generation_mode=scene_modes.get(i, "extrapolation"),
        )
        mvmem.add_or_update_segment(seg)

    logger.info("Retrieval precomputation complete")
    return mvmem


def get_relevant_context(mvmem: MVMem, scene_idx: int) -> dict:
    """Get all relevant context for a specific scene during generation.

    Returns dict with:
    - mode: extrapolation | interpolation
    - contiguous_idx: index of contiguous previous scene (or None)
    - relevant_scene_indices: list of relevant previous scene indices
    - relevant_ref_names: list of relevant reference names
    - relevant_frames: list of frame paths from relevant scenes
    - relevant_states: list of textual states from relevant scenes
    """
    seg = mvmem.get_segment(scene_idx)
    if not seg:
        return {"mode": "extrapolation"}

    # Collect relevant scene indices from S_rel
    rel_scenes = mvmem.relevant_scenes.get(scene_idx, {})
    all_indices = set()
    for category in ["objects", "characters", "environment"]:
        indices = rel_scenes.get(category, [])
        if isinstance(indices, list):
            all_indices.update(indices)

    # Collect frames and states from relevant scenes
    relevant_frames = []
    relevant_states = []
    for idx in sorted(all_indices):
        rel_seg = mvmem.get_segment(idx)
        if rel_seg:
            if rel_seg.begin_frame_path and rel_seg.begin_frame_path.exists():
                relevant_frames.append(rel_seg.begin_frame_path)
            if rel_seg.end_frame_path and rel_seg.end_frame_path.exists():
                relevant_frames.append(rel_seg.end_frame_path)
            relevant_states.append({
                "scene_index": idx,
                "scene_context": rel_seg.scene_context,
                "textual_states": rel_seg.textual_states,
            })

    # Collect relevant reference images
    ref_names = mvmem.relevant_refs.get(scene_idx, [])
    ref_frames = []
    for name in ref_names:
        ref = mvmem.get_reference_by_name(name)
        if ref and ref.image_path.exists():
            ref_frames.append(ref.image_path)

    return {
        "mode": seg.generation_mode,
        "contiguous_idx": mvmem.contiguous_scenes.get(scene_idx),
        "relevant_scene_indices": sorted(all_indices),
        "relevant_ref_names": ref_names,
        "relevant_frames": relevant_frames,
        "relevant_ref_frames": ref_frames,
        "relevant_states": relevant_states,
    }
