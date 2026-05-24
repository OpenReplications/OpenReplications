"""Hierarchical Test-Time Self-Improvement (HITS) (Paper Section 3.3).

Two levels:
1. Frame-level: verify + refine boundary frames via 8-metric rubric
2. Video-level: verify + refine video segments via 10-metric rubric

Each level uses Edit or Regenerate mode:
- Edit: fix a single issue by editing the prompt
- Regenerate: re-optimize prompt via MAPO, then regenerate
"""

import json
import logging
from pathlib import Path

from src.config import settings
from src.memory.schema import MVMem, PromptDBEntry, SegmentMemory
from src.models.mllm import call_mllm, call_mllm_json
from src.models.ti2i import generate_image
from src.models.ti2v import generate_video
from src.prompts.judges import (
    prompt_extract_frame_states,
    prompt_extract_video_states,
    prompt_judge_frame_consistency,
    prompt_judge_frame_quality,
    prompt_judge_frame_spatial,
    prompt_judge_frame_states,
    prompt_judge_video_inter_consistency,
    prompt_judge_video_intra_quality,
)
from src.prompts.mapo import prompt_mapo_feedback_reasoning

logger = logging.getLogger(__name__)


def refine_frame(
    frame_path: Path,
    frame_prompt: str,
    scene_idx: int,
    scene_description: str,
    mvmem: MVMem,
    context: dict,
    frame_type: str = "begin",
    output_dir: Path | None = None,
) -> tuple[Path, str, dict]:
    """HITS frame-level self-improvement.

    Args:
        frame_path: Path to the synthesized frame
        frame_prompt: The prompt used to generate the frame
        scene_idx: Current scene index
        scene_description: Current scene description
        mvmem: MVMem instance
        context: Retrieval context
        frame_type: "begin" or "end"
        output_dir: Output directory for refined frames

    Returns:
        (best_frame_path, best_prompt, best_scores)
    """
    out = output_dir or settings.output_dir
    max_iters = settings.max_refinement_iterations_frame
    threshold = settings.early_stop_threshold

    best_frame = frame_path
    best_prompt = frame_prompt
    best_scores = {}
    best_avg = 0.0

    candidates = [(frame_path, frame_prompt)]

    for iteration in range(max_iters + 1):  # +1 for initial evaluation
        current_frame = candidates[-1][0]
        current_prompt = candidates[-1][1]

        # --- Extract frame states ---
        try:
            states = call_mllm_json(
                prompt_extract_frame_states(scene_idx, scene_description, current_prompt),
                images=[current_frame],
            )
        except Exception as e:
            logger.warning(f"  Frame state extraction failed: {e}")
            states = {}

        # --- Run judges (Eq. 7) ---
        scores = _judge_frame(
            current_frame, current_prompt, scene_idx, scene_description,
            mvmem, context, states,
        )
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        logger.info(f"  Frame HITS iter {iteration}: avg={avg_score:.1f} scores={scores}")

        if avg_score > best_avg:
            best_avg = avg_score
            best_frame = current_frame
            best_prompt = current_prompt
            best_scores = scores

        # Early stop if good enough
        if avg_score >= threshold:
            logger.info(f"  Frame HITS early stop at iter {iteration} (avg={avg_score:.1f})")
            break

        # Skip refinement on last iteration
        if iteration >= max_iters:
            break

        # --- Decide Edit or Regenerate ---
        low_scores = {k: v for k, v in scores.items() if v < 8}
        if len(low_scores) <= 1 and low_scores:
            # Edit mode: fix a single issue
            issue = list(low_scores.keys())[0]
            logger.info(f"  Edit mode: fixing {issue}")
            refined_prompt = _edit_prompt(current_prompt, scores, issue)
        else:
            # Regenerate mode: MAPO full optimization
            logger.info(f"  Regenerate mode: MAPO optimization")
            refined_prompt = _mapo_optimize(current_prompt, scores, mvmem)

        # Generate N candidates, pick the best (best-of-N)
        n_candidates = settings.max_candidates_per_frame
        ref_images = context.get("relevant_ref_frames", [])
        iter_candidates = []
        for c_idx in range(n_candidates):
            new_path = out / f"frame_{frame_type}_s{scene_idx}_iter{iteration+1}_c{c_idx}.png"
            try:
                new_path = generate_image(
                    prompt=refined_prompt,
                    reference_images=ref_images,
                    output_path=new_path,
                )
                iter_candidates.append((new_path, refined_prompt))
            except Exception as e:
                logger.warning(f"  Frame candidate {c_idx} failed: {e}")
        if iter_candidates:
            candidates.append(iter_candidates[0])  # will be re-evaluated next iteration
            # TODO: evaluate all N and pick best here for efficiency
        else:
            logger.warning(f"  All {n_candidates} frame candidates failed")

    # Update MAPO prompt database
    _update_prompt_db(mvmem, best_prompt, frame_prompt, best_scores, best_avg > 0.8 * threshold)

    return best_frame, best_prompt, best_scores


def refine_video(
    video_path: Path,
    video_prompt: str,
    scene_idx: int,
    scene_description: str,
    begin_frame: Path | None,
    end_frame: Path | None,
    mvmem: MVMem,
    context: dict,
    output_dir: Path | None = None,
) -> tuple[Path, str, dict]:
    """HITS video-level self-improvement.

    Returns:
        (best_video_path, best_prompt, best_scores)
    """
    out = output_dir or settings.output_dir
    max_iters = settings.max_refinement_iterations_video
    threshold = settings.early_stop_threshold

    best_video = video_path
    best_prompt = video_prompt
    best_scores = {}
    best_avg = 0.0

    candidates = [(video_path, video_prompt)]

    for iteration in range(max_iters + 1):
        current_video = candidates[-1][0]
        current_prompt = candidates[-1][1]

        # --- Extract video states ---
        seg = mvmem.get_segment(scene_idx)
        frame_states = json.dumps(seg.frame_textual_states.__dict__) if seg and seg.frame_textual_states else "{}"
        try:
            video_states = call_mllm_json(
                prompt_extract_video_states(scene_idx, scene_description, frame_states),
                videos=[current_video],
            )
        except Exception as e:
            logger.warning(f"  Video state extraction failed: {e}")
            video_states = {}

        # --- Run judges (Eq. 9) ---
        scores = _judge_video(
            current_video, current_prompt, scene_idx, scene_description,
            mvmem, context, video_states,
        )
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        logger.info(f"  Video HITS iter {iteration}: avg={avg_score:.1f}")

        if avg_score > best_avg:
            best_avg = avg_score
            best_video = current_video
            best_prompt = current_prompt
            best_scores = scores

        if avg_score >= threshold:
            logger.info(f"  Video HITS early stop at iter {iteration} (avg={avg_score:.1f})")
            break
        if iteration >= max_iters:
            break

        # --- Decide Edit or Regenerate (same logic as frame HITS) ---
        low_scores = {k: v for k, v in scores.items() if v < 8}
        if len(low_scores) <= 1 and low_scores:
            issue = list(low_scores.keys())[0]
            logger.info(f"  Video Edit mode: fixing {issue}")
            refined_prompt = _edit_prompt(current_prompt, scores, issue)
        else:
            logger.info(f"  Video Regenerate mode: MAPO optimization")
            refined_prompt = _mapo_optimize(current_prompt, scores, mvmem)

        # Generate N video candidates, pick best
        n_candidates = settings.max_candidates_per_video
        iter_candidates = []
        for c_idx in range(n_candidates):
            new_path = out / f"segment_s{scene_idx}_iter{iteration+1}_c{c_idx}.mp4"
            try:
                new_path = generate_video(
                    prompt=refined_prompt,
                    begin_frame=begin_frame,
                    end_frame=end_frame,
                    output_path=new_path,
                )
                iter_candidates.append((new_path, refined_prompt))
            except Exception as e:
                logger.warning(f"  Video candidate {c_idx} failed: {e}")
        if iter_candidates:
            candidates.append(iter_candidates[0])
        else:
            logger.warning(f"  All {n_candidates} video candidates failed")

    _update_prompt_db(mvmem, best_prompt, video_prompt, best_scores, best_avg > 0.8 * threshold)

    return best_video, best_prompt, best_scores


def _judge_frame(
    frame_path: Path,
    frame_prompt: str,
    scene_idx: int,
    scene_desc: str,
    mvmem: MVMem,
    context: dict,
    states: dict,
) -> dict:
    """Run all frame judges (Eq. 7) and aggregate scores."""
    scenes_text = "\n".join(
        f"{s.segment_index}. {s.scene_context}"
        for s in sorted(mvmem.segments, key=lambda x: x.segment_index)
    )
    ref_images = context.get("relevant_frames", []) + context.get("relevant_ref_frames", [])
    image_mapping = f"Image 1: current frame. Images 2+: reference frames."

    all_scores = {}

    # Judge 1: Consistency over images (E.3.3)
    try:
        result = call_mllm_json(
            prompt_judge_frame_consistency(scenes_text, scene_idx, scene_desc, frame_prompt, image_mapping),
            images=[frame_path] + ref_images[:5],
        )
        for key in ["entity_reference_consistency", "environment_reference_consistency", "narrative_progression"]:
            if key in result:
                all_scores[key] = result[key]
    except Exception as e:
        logger.warning(f"  Frame consistency judge failed: {e}")

    # Judge 2: Spatial logicalness (E.3.4)
    try:
        result = call_mllm_json(
            prompt_judge_frame_spatial(scenes_text, scene_idx, scene_desc, frame_prompt, image_mapping),
            images=[frame_path] + ref_images[:5],
        )
        if "spatial_logicalness" in result:
            all_scores["spatial_logicalness"] = result["spatial_logicalness"]
    except Exception as e:
        logger.warning(f"  Frame spatial judge failed: {e}")

    # Judge 3: Textual states consistency (E.3.5)
    if context.get("relevant_states"):
        try:
            video_states = json.dumps([
                {"scene": s["scene_index"], "context": s["scene_context"]}
                for s in context["relevant_states"][:3]
            ])
            result = call_mllm_json(
                prompt_judge_frame_states(
                    scenes_text, scene_idx, scene_desc, video_states, json.dumps(states)
                ),
                images=[frame_path],
            )
            for key in ["objects_state_score", "characters_state_score", "environment_state_score"]:
                if key in result:
                    all_scores[key] = result[key]
        except Exception as e:
            logger.warning(f"  Frame states judge failed: {e}")

    # Judge 4: Basic quality (E.3.6)
    try:
        result = call_mllm_json(
            prompt_judge_frame_quality(frame_prompt, json.dumps(states)),
            images=[frame_path],
        )
        for key in ["instruction_following", "physical_plausibility"]:
            if key in result:
                all_scores[key] = result[key]
    except Exception as e:
        logger.warning(f"  Frame quality judge failed: {e}")

    return all_scores


def _judge_video(
    video_path: Path,
    video_prompt: str,
    scene_idx: int,
    scene_desc: str,
    mvmem: MVMem,
    context: dict,
    video_states: dict,
) -> dict:
    """Run all video judges (Eq. 9) and aggregate scores."""
    all_scores = {}

    # Judge: Intra consistency and quality (E.3.8)
    try:
        result = call_mllm_json(
            prompt_judge_video_intra_quality(
                scene_idx, scene_desc, video_prompt, json.dumps(video_states)
            ),
            videos=[video_path],
        )
        for key in ["character_state", "object_state", "environment_state",
                     "instruction_following", "physical_plausibility", "narrative_progression"]:
            if key in result:
                all_scores[key] = result[key]
    except Exception as e:
        logger.warning(f"  Video intra judge failed: {e}")

    # Judge: Inter consistency (E.3.7) - only if contiguous scene exists
    cont_idx = context.get("contiguous_idx")
    if cont_idx is not None:
        cont_seg = mvmem.get_segment(cont_idx)
        if cont_seg and cont_seg.video_path:
            scenes_text = "\n".join(
                f"{s.segment_index}. {s.scene_context}"
                for s in sorted(mvmem.segments, key=lambda x: x.segment_index)
            )
            try:
                result = call_mllm_json(
                    prompt_judge_video_inter_consistency(
                        scenes_text, cont_idx, cont_seg.scene_context,
                        scene_idx, scene_desc, json.dumps(video_states)
                    ),
                    videos=[cont_seg.video_path, video_path],
                )
                for key in ["inter_entity_consistency", "inter_environment_consistency",
                             "inter_motion_consistency", "camera_consistency"]:
                    if key in result:
                        all_scores[key] = result[key]
            except Exception as e:
                logger.warning(f"  Video inter judge failed: {e}")

    return all_scores


def _edit_prompt(prompt: str, scores: dict, issue: str) -> str:
    """Edit mode: surgical fix for a single issue."""
    from src.prompts.mapo import prompt_mapo_edit_mode
    try:
        result = call_mllm_json(prompt_mapo_edit_mode(prompt, scores, issue))
        return result.get("refined_prompt", prompt)
    except Exception:
        return prompt


def _mapo_optimize(prompt: str, scores: dict, mvmem: MVMem) -> str:
    """MAPO: Memory-Augmented Prompt Optimization.

    Retrieves similar past cases from prompt DB via embedding cosine similarity,
    then contrasts positive/negative examples to derive refinement lessons.
    """
    from src.models.mllm import embed_text

    pos_cases = _retrieve_similar_cases(prompt, mvmem, label="pos", top_k=5)
    neg_cases = _retrieve_similar_cases(prompt, mvmem, label="neg", top_k=5)

    try:
        result = call_mllm_json(prompt_mapo_feedback_reasoning(prompt, scores, pos_cases, neg_cases))
        return result.get("refined_prompt", prompt)
    except Exception:
        return prompt


def _retrieve_similar_cases(
    query: str, mvmem: MVMem, label: str, top_k: int = 5
) -> list[dict]:
    """Retrieve top-k most similar prompt DB entries by embedding cosine similarity."""
    from src.models.mllm import embed_text

    candidates = [e for e in mvmem.prompt_db if e.label == label and e.embedding]
    if not candidates:
        # Fallback to most recent if no embeddings available
        return [
            {"original_prompt": e.original_prompt, "refined_prompt": e.refined_prompt, "rubric_scores": e.rubric_scores}
            for e in mvmem.prompt_db if e.label == label
        ][-top_k:]

    try:
        query_emb = embed_text(query)
    except Exception:
        return [
            {"original_prompt": e.original_prompt, "refined_prompt": e.refined_prompt, "rubric_scores": e.rubric_scores}
            for e in candidates
        ][-top_k:]

    scored = []
    for e in candidates:
        sim = _cosine_similarity(query_emb, e.embedding)
        scored.append((sim, e))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {"original_prompt": e.original_prompt, "refined_prompt": e.refined_prompt, "rubric_scores": e.rubric_scores}
        for _, e in scored[:top_k]
    ]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _update_prompt_db(mvmem: MVMem, refined: str, original: str, scores: dict, is_positive: bool):
    """Update MAPO prompt database with new case and its embedding."""
    from src.models.mllm import embed_text

    embedding = []
    try:
        embedding = embed_text(refined)
    except Exception as e:
        logger.warning(f"  Failed to embed prompt for MAPO DB: {e}")

    mvmem.prompt_db.append(PromptDBEntry(
        original_prompt=original,
        refined_prompt=refined,
        rubric_scores=scores,
        label="pos" if is_positive else "neg",
        embedding=embedding,
    ))
