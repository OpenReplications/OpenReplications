"""Agentic Auto-Regressive Generation Pipeline (Paper Section 3.2).

For each segment: Retrieve → Synthesize → Refine → Update
Implements the full closed-loop including:
- Adaptive segment generation (Eq. 2)
- Boundary frame synthesis + HITS (Eq. 5)
- Video segment synthesis + HITS (Eq. 6)
- MVMem update (step v)
"""

import json
import logging
from pathlib import Path

from src.config import settings
from src.memory.schema import MVMem, SegmentMemory, TextualStates
from src.models.mllm import call_mllm, call_mllm_json
from src.models.ti2i import generate_image
from src.models.ti2v import generate_video
from src.pipeline.hits import refine_frame, refine_video
from src.pipeline.retrieve import get_relevant_context
from src.prompts.generation import (
    prompt_generate_frame_prompts,
    prompt_generate_video_prompt,
    prompt_refine_frame_prompt,
    prompt_retrieve_end_frame,
)
from src.prompts.judges import prompt_extract_frame_states, prompt_extract_video_states

logger = logging.getLogger(__name__)


def generate_all_frame_prompts(mvmem: MVMem, scenes: list[str]) -> dict[int, dict]:
    """Pre-generate frame prompts for all scenes (E.2.6 step 1).

    Returns dict: scene_idx -> {"begin_frame": prompt, "end_frame": prompt (optional)}
    """
    logger.info("Generating frame prompts for all scenes...")

    ref_descriptions = "\n".join(
        f"- {r.name}: {r.caption}" for r in mvmem.references
    )

    result = call_mllm_json(
        prompt_generate_frame_prompts(scenes, ref_descriptions)
    )

    frame_prompts = {}
    if isinstance(result, dict) and "frame_prompts" in result:
        for entry in result["frame_prompts"]:
            idx = entry.get("scene_index", 0)
            frame_prompts[idx] = {}
            if "begin_frame" in entry:
                frame_prompts[idx]["begin_frame"] = entry["begin_frame"]
            if "end_frame" in entry:
                frame_prompts[idx]["end_frame"] = entry["end_frame"]

    logger.info(f"  Generated prompts for {len(frame_prompts)} scenes")
    return frame_prompts


def generate_segment(
    scene_idx: int,
    mvmem: MVMem,
    scenes: list[str],
    frame_prompts: dict[int, dict],
    output_dir: Path,
) -> MVMem:
    """Generate a single video segment with the full Retrieve-Synthesize-Refine-Update cycle.

    Args:
        scene_idx: Index of the current scene
        mvmem: MVMem instance (modified in place)
        scenes: Full list of scene descriptions
        frame_prompts: Pre-generated frame prompts
        output_dir: Output directory

    Returns:
        Updated MVMem
    """
    seg = mvmem.get_segment(scene_idx)
    if not seg:
        seg = SegmentMemory(segment_index=scene_idx, scene_context=scenes[scene_idx])
        mvmem.add_or_update_segment(seg)

    scene_desc = scenes[scene_idx]
    seg_dir = output_dir / f"segment_{scene_idx:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Segment {scene_idx}: {scene_desc[:80]}...")
    logger.info(f"Mode: {seg.generation_mode}")

    # --- (i) Retrieve context ---
    context = get_relevant_context(mvmem, scene_idx)
    mode = context["mode"]
    cont_idx = context.get("contiguous_idx")

    logger.info(f"  Contiguous: {cont_idx}, Relevant scenes: {context['relevant_scene_indices'][:5]}...")

    # --- (ii) Determine begin frame ---
    # F_i^begin := F_{i-1}^end for i > 0 (continuity)
    begin_frame_path = None
    if scene_idx > 0:
        prev_seg = mvmem.get_segment(scene_idx - 1)
        if prev_seg and prev_seg.end_frame_path and prev_seg.end_frame_path.exists():
            begin_frame_path = prev_seg.end_frame_path
            logger.info(f"  Begin frame: reusing previous segment's end frame")

    # For first segment, synthesize begin frame
    if begin_frame_path is None:
        begin_prompt = frame_prompts.get(scene_idx, {}).get("begin_frame", scene_desc)

        # Refine prompt with memory context (E.2.6 step 2)
        if context["relevant_states"]:
            scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))
            video_states = json.dumps([
                {"scene": s["scene_index"], "context": s["scene_context"]}
                for s in context["relevant_states"][:3]
            ])
            image_mapping = "Reference images from relevant scenes and global references."

            try:
                refined = call_mllm_json(
                    prompt_refine_frame_prompt(
                        scenes_text, scene_idx, begin_prompt, video_states, image_mapping
                    ),
                    images=context.get("relevant_ref_frames", [])[:5],
                )
                if "refined_prompt" in refined:
                    begin_prompt = refined["refined_prompt"]
                    logger.info(f"  Refined begin frame prompt")
            except Exception as e:
                logger.warning(f"  Frame prompt refinement failed: {e}")

        # Generate begin frame
        begin_frame_path = seg_dir / "begin_frame.png"
        begin_frame_path = generate_image(
            prompt=begin_prompt,
            reference_images=context.get("relevant_ref_frames", [])[:4],
            output_path=begin_frame_path,
        )

        # HITS frame refinement
        begin_frame_path, begin_prompt, frame_scores = refine_frame(
            begin_frame_path, begin_prompt, scene_idx, scene_desc,
            mvmem, context, frame_type="begin", output_dir=seg_dir,
        )

    seg.begin_frame_path = begin_frame_path

    # --- (iii) Determine end frame (Eq. 5) ---
    end_frame_path = None
    is_last_scene = scene_idx == len(scenes) - 1

    if mode == "interpolation" and not is_last_scene:
        # For interpolation: need to determine end frame
        # Check if contiguous video exists for frame retrieval (E.2.5)
        if cont_idx is not None:
            cont_seg = mvmem.get_segment(cont_idx)
            if cont_seg and cont_seg.video_path and cont_seg.video_path.exists():
                end_frame_path = _retrieve_end_frame_from_video(
                    cont_seg, scene_desc, seg_dir,
                )

        # If retrieval didn't yield a frame, generate one
        if end_frame_path is None:
            next_scene_idx = scene_idx + 1
            end_prompt = frame_prompts.get(next_scene_idx, {}).get(
                "begin_frame", scenes[next_scene_idx] if next_scene_idx < len(scenes) else ""
            )
            if end_prompt:
                end_frame_path = seg_dir / "end_frame.png"
                end_frame_path = generate_image(
                    prompt=end_prompt,
                    reference_images=context.get("relevant_ref_frames", [])[:4],
                    output_path=end_frame_path,
                )
                # HITS refinement for end frame
                end_frame_path, end_prompt, _ = refine_frame(
                    end_frame_path, end_prompt, scene_idx, scene_desc,
                    mvmem, context, frame_type="end", output_dir=seg_dir,
                )

    elif is_last_scene:
        # Last scene: generate end frame
        end_prompt = frame_prompts.get(scene_idx, {}).get("end_frame", "")
        if end_prompt:
            end_frame_path = seg_dir / "end_frame.png"
            end_frame_path = generate_image(
                prompt=end_prompt,
                reference_images=context.get("relevant_ref_frames", [])[:4],
                output_path=end_frame_path,
            )

    seg.end_frame_path = end_frame_path

    # --- (iv) Extract frame textual states (T_i^F) ---
    try:
        frame_states = call_mllm_json(
            prompt_extract_frame_states(scene_idx, scene_desc, "begin frame"),
            images=[begin_frame_path],
        )
        seg.frame_textual_states = _parse_textual_states(frame_states)
    except Exception as e:
        logger.warning(f"  Frame state extraction failed: {e}")

    # --- (v) Generate video prompt (Eq. 6) ---
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))

    # Get previous video prompt for continuity
    prev_video_prompt = None
    if scene_idx > 0:
        prev_seg = mvmem.get_segment(scene_idx - 1)
        if prev_seg:
            prev_video_prompt = getattr(prev_seg, '_video_prompt', None)

    # Collect video states memory
    video_states_memory = []
    for idx in context.get("relevant_scene_indices", [])[-3:]:
        rel_seg = mvmem.get_segment(idx)
        if rel_seg and rel_seg.textual_states:
            video_states_memory.append({
                "scene": idx,
                "states": rel_seg.textual_states.camera,
            })

    video_prompt_result = call_mllm_json(
        prompt_generate_video_prompt(
            scenes_text=scenes_text,
            scene_idx=scene_idx,
            current_scene=scene_desc,
            video_states=json.dumps(video_states_memory),
            previous_video_prompt=prev_video_prompt,
            has_end_frame=end_frame_path is not None,
        ),
        images=[p for p in [begin_frame_path, end_frame_path] if p],
    )

    video_prompt = video_prompt_result.get("video_prompt", scene_desc)
    logger.info(f"  Video prompt: {video_prompt[:100]}...")

    # --- (vi) Synthesize video segment (Eq. 6) ---
    video_path = seg_dir / "segment.mp4"
    video_path = generate_video(
        prompt=video_prompt,
        begin_frame=begin_frame_path,
        end_frame=end_frame_path,
        reference_images=context.get("relevant_ref_frames", [])[:3],
        output_path=video_path,
    )

    # --- (vii) HITS video refinement ---
    video_path, video_prompt, video_scores = refine_video(
        video_path, video_prompt, scene_idx, scene_desc,
        begin_frame_path, end_frame_path,
        mvmem, context, output_dir=seg_dir,
    )

    seg.video_path = video_path
    seg._video_prompt = video_prompt  # Store for next segment's reference

    # --- (viii) Update MVMem ---
    # Extract full video textual states
    try:
        frame_states_json = _textual_states_to_json(seg.frame_textual_states) if seg.frame_textual_states else "{}"
        full_states = call_mllm_json(
            prompt_extract_video_states(scene_idx, scene_desc, frame_states_json),
            videos=[video_path],
        )
        seg.textual_states = _parse_textual_states_from_video(full_states, seg.frame_textual_states)
    except Exception as e:
        logger.warning(f"  Video state extraction failed: {e}")

    mvmem.add_or_update_segment(seg)
    logger.info(f"  Segment {scene_idx} complete: {video_path}")

    return mvmem


def _retrieve_end_frame_from_video(
    cont_seg: SegmentMemory,
    curr_scene: str,
    output_dir: Path,
) -> Path | None:
    """E.2.5 - MLLM_retr^img: retrieve best end-of-shot frame from contiguous video.

    Extracts candidate frames from the contiguous segment's video,
    then asks the MLLM to pick the best continuation frame.
    """
    import subprocess
    import tempfile

    video_path = cont_seg.video_path
    if not video_path or not video_path.exists():
        return None

    # Extract candidate frames at regular intervals using ffmpeg
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-vf", "fps=1", "-q:v", "2",
                 str(tmpdir / "frame_%03d.png")],
                capture_output=True, check=True, timeout=30,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"  Frame extraction failed: {e}")
            return None

        frames = sorted(tmpdir.glob("frame_*.png"))
        if not frames:
            return None

        # Ask MLLM to pick the best end-of-shot frame
        image_mapping = ", ".join(f"Image {i+1}: frame at {i+1}s" for i in range(len(frames)))
        try:
            result = call_mllm_json(
                prompt_retrieve_end_frame(
                    cont_seg.scene_context, curr_scene, image_mapping
                ),
                images=frames,
            )
            best_idx = result.get("best_index")
            if best_idx is not None and 0 <= best_idx < len(frames):
                dest = output_dir / "end_frame.png"
                import shutil
                shutil.copy2(frames[best_idx], dest)
                logger.info(f"  Retrieved end frame from contiguous video (frame {best_idx})")
                return dest
        except Exception as e:
            logger.warning(f"  End frame retrieval failed: {e}")

    return None


def _textual_states_to_json(ts: TextualStates) -> str:
    """Serialize TextualStates to JSON string."""
    import json
    return json.dumps({
        "visual_arcs": [
            {"entity_name": a.entity_name, "identity": a.identity,
             "identity_changes": a.identity_changes, "motion": a.motion}
            for a in ts.visual_arcs
        ],
        "spatial_relations": [
            {"subject": r.subject, "relation": r.relation, "object": r.object}
            for r in ts.spatial_relations
        ],
        "camera": ts.camera,
    })


def _parse_textual_states(frame_data: dict) -> TextualStates:
    """Parse MLLM-extracted frame states into TextualStates with proper fields."""
    from src.memory.schema import SpatialRelation, VisualArc

    arcs = []
    entities = frame_data.get("entities", {})
    for name, info in entities.items():
        if isinstance(info, dict):
            arcs.append(VisualArc(
                entity_name=name,
                identity=info.get("identity", ""),
                identity_changes="",
                motion="",
            ))

    relations = []
    for rel in frame_data.get("spatial_relations", []):
        if isinstance(rel, dict):
            relations.append(SpatialRelation(
                subject=rel.get("subject", ""),
                relation=rel.get("relation", ""),
                object=rel.get("object", ""),
            ))

    return TextualStates(visual_arcs=arcs, spatial_relations=relations, camera="")


def _parse_textual_states_from_video(video_data: dict, frame_states: TextualStates | None) -> TextualStates:
    """Merge video-extracted states with frame-level states."""
    from src.memory.schema import SpatialRelation, VisualArc

    arcs = list(frame_states.visual_arcs) if frame_states else []
    relations = list(frame_states.spatial_relations) if frame_states else []

    existing_names = {a.entity_name for a in arcs}
    identity_changes = video_data.get("identity_changes", {})
    motions = video_data.get("motions", {})

    for arc in arcs:
        arc.identity_changes = identity_changes.get(arc.entity_name, "")
        arc.motion = motions.get(arc.entity_name, "")

    for name, desc in video_data.get("new_entities", {}).items():
        if name not in existing_names:
            arcs.append(VisualArc(
                entity_name=name,
                identity=desc,
                identity_changes=identity_changes.get(name, ""),
                motion=motions.get(name, ""),
            ))

    camera = video_data.get("camera", "")
    return TextualStates(visual_arcs=arcs, spatial_relations=relations, camera=camera)
