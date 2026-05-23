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

    out = output_dir or settings.output_dir
    out.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    logger.info(f"Starting A²RD with {len(scenes)} scenes")
    logger.info(f"Output directory: {out}")

    # --- Phase 1: MVMem Initialization ---
    logger.info("=" * 60)
    logger.info("Phase 1: MVMem Initialization")
    logger.info("=" * 60)
    t0 = time.time()
    mvmem = initialize_mvmem(user_prompt, scenes, ref_images, out)
    logger.info(f"Phase 1 complete ({time.time()-t0:.1f}s)")

    # --- Phase 2: Precompute retrieval ---
    logger.info("=" * 60)
    logger.info("Phase 2: Precomputing retrieval contexts")
    logger.info("=" * 60)
    t0 = time.time()
    mvmem = precompute_retrieval(mvmem, scenes)
    logger.info(f"Phase 2 complete ({time.time()-t0:.1f}s)")

    # --- Phase 2.5: Pre-generate all frame prompts ---
    logger.info("Generating frame prompts for all scenes...")
    frame_prompts = generate_all_frame_prompts(mvmem, scenes)

    # --- Phase 3: Autoregressive generation loop ---
    logger.info("=" * 60)
    logger.info("Phase 3: Autoregressive segment generation")
    logger.info("=" * 60)

    for i in range(len(scenes)):
        t0 = time.time()
        mvmem = generate_segment(i, mvmem, scenes, frame_prompts, out)
        logger.info(f"Segment {i} took {time.time()-t0:.1f}s")

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
        scenes = []
        refs = []
    else:
        parser.error("Provide either --storyline or --prompt")
        return

    out = Path(args.output_dir) if args.output_dir else None
    run(prompt, scenes, refs, out)


if __name__ == "__main__":
    main()
