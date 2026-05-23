"""OpenA2RD: Open-source replication of A²RD.

Usage:
    uv run python -m src.main --storyline examples/chef_story.yaml
    uv run python -m src.main --prompt "A chef prepares a meal" --num-scenes 8
"""

import argparse
import logging
from pathlib import Path

import yaml

from src.config import settings

logger = logging.getLogger(__name__)


def load_storyline(path: str) -> tuple[str, list[str], list[str]]:
    """Load storyline from YAML file.

    Returns:
        (user_prompt, scenes, ref_image_paths)
    """
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
    """Run the full A²RD pipeline.

    1. Initialize MVMem (plan entities, build DAG, synthesize references)
    2. For each segment: Retrieve → Synthesize → Refine → Update
    3. Output final video
    """
    out = output_dir or settings.output_dir
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting A²RD with {len(scenes)} scenes")
    logger.info(f"Output directory: {out}")

    # Step 1: MVMem Initialization
    # TODO: Implement in pipeline/init.py
    logger.info("Phase 1: MVMem Initialization")

    # Step 2: Precompute retrieval (Equations 2-4, all scenes at once)
    # TODO: Implement in pipeline/retrieve.py
    logger.info("Phase 2: Precomputing retrieval contexts")

    # Step 3: Autoregressive generation loop
    logger.info("Phase 3: Autoregressive segment generation")
    for i, scene in enumerate(scenes):
        logger.info(f"  Segment {i}/{len(scenes)-1}: {scene[:80]}...")

        # 3a. Determine generation mode (Eq. 2)
        # 3b. Retrieve contexts from MVMem (Eq. 3-4)
        # 3c. Synthesize boundary frames (Eq. 5) + HITS frame refinement
        # 3d. Synthesize video segment (Eq. 6) + HITS video refinement
        # 3e. Update MVMem

    logger.info("A²RD pipeline complete")


def main():
    parser = argparse.ArgumentParser(description="OpenA2RD: Long Video Generation")
    parser.add_argument("--storyline", type=str, help="Path to storyline YAML")
    parser.add_argument("--prompt", type=str, help="User prompt (if no storyline)")
    parser.add_argument("--num-scenes", type=int, default=8, help="Number of scenes")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level)

    if args.storyline:
        prompt, scenes, refs = load_storyline(args.storyline)
    elif args.prompt:
        prompt = args.prompt
        scenes = []  # TODO: auto-generate scenes from prompt
        refs = []
    else:
        parser.error("Provide either --storyline or --prompt")
        return

    out = Path(args.output_dir) if args.output_dir else None
    run(prompt, scenes, refs, out)


if __name__ == "__main__":
    main()
