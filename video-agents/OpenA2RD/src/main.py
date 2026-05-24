"""OpenA2RD: Open-source replication of A²RD.

Usage:
    uv run python -m src.main --storyline examples/diver_story.yaml
    uv run python -m src.main --prompt "A chef prepares a meal" --num-scenes 8
"""

import argparse
import logging
import time
from pathlib import Path

import yaml

from src.config import settings

logger = logging.getLogger(__name__)


def load_storyline(path: str) -> tuple[str, list[str], list[str]]:
    """Load storyline from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return (
        data.get("prompt", ""),
        data.get("scenes", []),
        data.get("reference_images", []),
    )


def run(
    user_prompt: str,
    scenes: list[str],
    ref_images: list[str] | None = None,
    output_dir: Path | None = None,
):
    """Run the full A²RD pipeline."""
    from src.pipeline.init import initialize_mvmem
    from src.pipeline.retrieve import precompute_retrieval
    from src.pipeline.segment_gen import generate_all_frame_prompts, generate_segment

    from src.memory.schema import MVMem

    out = output_dir or settings.output_dir
    out.mkdir(parents=True, exist_ok=True)

    ckpt = out / "checkpoint.json"
    ckpt_prompts = out / "frame_prompts.json"

    total_start = time.time()
    logger.info(f"Starting A²RD with {len(scenes)} scenes")
    logger.info(f"Output directory: {out}")

    # --- Detect resume point ---
    mvmem = None
    frame_prompts = None
    resume_phase = 1  # start from scratch

    if ckpt.exists():
        import json
        mvmem = MVMem.load(ckpt)
        logger.info(f"Loaded checkpoint: {len(mvmem.references)} refs, {len(mvmem.segments)} segments")

        if ckpt_prompts.exists():
            frame_prompts = json.loads(ckpt_prompts.read_text())
            # Convert keys back to int
            frame_prompts = {int(k): v for k, v in frame_prompts.items()}

        # Figure out where to resume
        if not mvmem.references:
            resume_phase = 1
        elif not mvmem.contiguous_scenes and not mvmem.relevant_scenes:
            resume_phase = 2
        elif frame_prompts is None:
            resume_phase = 2.5
        else:
            # Find first segment without a video
            completed = sum(1 for s in mvmem.segments if s.video_path and Path(s.video_path).exists())
            if completed >= len(scenes):
                resume_phase = 4  # all done
            else:
                resume_phase = 3
        logger.info(f"Resuming from phase {resume_phase}")
    else:
        logger.info("No checkpoint found, starting fresh")

    # --- Phase 1: MVMem Initialization ---
    if resume_phase <= 1:
        logger.info("=" * 60)
        logger.info("Phase 1: MVMem Initialization")
        logger.info("=" * 60)
        t0 = time.time()
        mvmem = initialize_mvmem(user_prompt, scenes, ref_images, out)
        mvmem.save(ckpt)
        logger.info(f"Phase 1 complete ({time.time()-t0:.1f}s) — checkpoint saved")
    else:
        logger.info("Phase 1: SKIPPED (loaded from checkpoint)")

    # --- Phase 2: Precompute retrieval ---
    if resume_phase <= 2:
        logger.info("=" * 60)
        logger.info("Phase 2: Precomputing retrieval contexts")
        logger.info("=" * 60)
        t0 = time.time()
        mvmem = precompute_retrieval(mvmem, scenes)
        mvmem.save(ckpt)
        logger.info(f"Phase 2 complete ({time.time()-t0:.1f}s) — checkpoint saved")
    else:
        logger.info("Phase 2: SKIPPED (loaded from checkpoint)")

    # --- Phase 2.5: Pre-generate all frame prompts ---
    if resume_phase <= 2.5 or frame_prompts is None:
        import json as _json
        logger.info("Generating frame prompts for all scenes...")
        frame_prompts = generate_all_frame_prompts(mvmem, scenes)
        ckpt_prompts.write_text(_json.dumps(frame_prompts, indent=2, ensure_ascii=False))
        logger.info("Frame prompts saved")
    else:
        logger.info("Frame prompts: SKIPPED (loaded from checkpoint)")

    # --- Phase 3: Autoregressive generation loop ---
    logger.info("=" * 60)
    logger.info("Phase 3: Autoregressive segment generation")
    logger.info("=" * 60)

    for i in range(len(scenes)):
        seg = mvmem.get_segment(i)
        if seg and seg.video_path and Path(seg.video_path).exists():
            logger.info(f"Segment {i}: SKIPPED (video exists: {seg.video_path})")
            continue

        t0 = time.time()
        logger.info(f"Segment {i}: generating...")
        mvmem = generate_segment(i, mvmem, scenes, frame_prompts, out)
        mvmem.save(ckpt)
        logger.info(f"Segment {i} took {time.time()-t0:.1f}s — checkpoint saved")

    # --- Phase 4: Concatenate final video ---
    logger.info("=" * 60)
    logger.info("Phase 4: Concatenating final video")
    logger.info("=" * 60)
    final_video = _concatenate_segments(mvmem, out)
    if final_video:
        logger.info(f"Final video: {final_video}")

    # --- Summary ---
    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"A²RD pipeline complete in {total_time:.1f}s")
    logger.info(f"Generated {len(scenes)} segments")
    logger.info(f"Output: {out}")

    # List generated videos
    for seg in sorted(mvmem.segments, key=lambda s: s.segment_index):
        if seg.video_path and seg.video_path.exists():
            logger.info(f"  Segment {seg.segment_index}: {seg.video_path}")

    return mvmem


def main():
    parser = argparse.ArgumentParser(description="OpenA2RD: Long Video Generation")
    parser.add_argument("--storyline", type=str, help="Path to storyline YAML")
    parser.add_argument("--prompt", type=str, help="User prompt (if no storyline)")
    parser.add_argument("--num-scenes", type=int, default=8, help="Number of scenes")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.storyline:
        prompt, scenes, refs = load_storyline(args.storyline)
    elif args.prompt:
        prompt = args.prompt
        refs = []
        # Auto-generate scenes from prompt via MLLM
        from src.models.mllm import call_mllm_json
        logger.info(f"Generating {args.num_scenes} scenes from prompt...")
        result = call_mllm_json(
            f"Break this video concept into exactly {args.num_scenes} sequential scene "
            f"descriptions. Each scene is one concise sentence describing a specific "
            f"moment or action.\n\nConcept: {prompt}\n\n"
            f"Return as a JSON list of strings:\n```json\n[\"scene 1\", \"scene 2\", ...]\n```"
        )
        scenes = result if isinstance(result, list) else []
        logger.info(f"Generated {len(scenes)} scenes")
    else:
        parser.error("Provide either --storyline or --prompt")
        return

    out = Path(args.output_dir) if args.output_dir else None
    run(prompt, scenes, refs, out)


def _concatenate_segments(mvmem, output_dir: Path) -> Path | None:
    """Concatenate all segment videos into a single final video using ffmpeg."""
    import subprocess

    segments = sorted(
        [s for s in mvmem.segments if s.video_path and Path(s.video_path).exists()],
        key=lambda s: s.segment_index,
    )
    if not segments:
        logger.warning("No video segments to concatenate")
        return None

    if len(segments) == 1:
        logger.info("Only 1 segment, no concatenation needed")
        return Path(segments[0].video_path)

    # Write ffmpeg concat list
    concat_list = output_dir / "concat_list.txt"
    with open(concat_list, "w") as f:
        for seg in segments:
            f.write(f"file '{Path(seg.video_path).resolve()}'\n")

    final_path = output_dir / "final_video.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(final_path)],
            capture_output=True, check=True, timeout=120,
        )
        logger.info(f"Concatenated {len(segments)} segments -> {final_path}")
        return final_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found, skipping concatenation")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"Concatenation failed: {e.stderr[:200] if e.stderr else e}")
        return None


if __name__ == "__main__":
    main()
