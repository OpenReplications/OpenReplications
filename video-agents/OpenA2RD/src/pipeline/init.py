"""MVMem Initialization Pipeline (Paper Section 3.1).

Steps:
(i)   Planning: Extract entities and environments from storyline
(ii)  Identifying Dependencies: Build DAG over references
(iii) Synthesizing References: Generate reference images in topological order
"""

import json
import logging
from pathlib import Path

from src.config import settings
from src.memory.schema import Entity, MVMem, ReferenceFrame
from src.models.mllm import call_mllm, call_mllm_json
from src.models.ti2i import generate_image
from src.prompts.mvmem_init import (
    prompt_caption_user_image,
    prompt_dependency_graph,
    prompt_plan_entities,
    prompt_plan_environments,
    prompt_synthesize_entity,
    prompt_synthesize_environment,
)

logger = logging.getLogger(__name__)


def initialize_mvmem(
    user_prompt: str,
    scenes: list[str],
    ref_image_paths: list[str] | None = None,
    output_dir: Path | None = None,
) -> MVMem:
    """Initialize MVMem: plan entities/envs, build DAG, synthesize references.

    Implements Equation (1) and Init steps (i)-(iii).
    """
    out = output_dir or settings.output_dir
    refs_dir = out / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    mvmem = MVMem()

    # --- (i) Planning: Extract entities and environments ---
    logger.info("Planning: Extracting entities...")
    entities_result = call_mllm_json(prompt_plan_entities(user_prompt, scenes))
    entities = []
    if isinstance(entities_result, list):
        for e in entities_result:
            entities.append(Entity(
                name=e.get("name", ""),
                description=e.get("description", ""),
                type="character",
            ))
    logger.info(f"  Found {len(entities)} entities: {[e.name for e in entities]}")

    logger.info("Planning: Extracting environments...")
    envs_result = call_mllm_json(prompt_plan_environments(user_prompt, scenes))
    environments = []
    if isinstance(envs_result, list):
        for e in envs_result:
            name = e.get("environment_name", e.get("name", ""))
            desc = e.get("detailed_spartial_description", e.get("description", ""))
            environments.append(Entity(
                name=name,
                description=desc,
                type="environment",
            ))
    logger.info(f"  Found {len(environments)} environments: {[e.name for e in environments]}")

    all_refs = entities + environments
    mvmem.entities = all_refs

    # --- (ii) Identifying Dependencies: Build DAG ---
    logger.info("Building dependency graph...")
    refs_json = json.dumps(
        [{f"{e.name}": e.description} for e in all_refs], indent=2
    )
    dep_result = call_mllm_json(prompt_dependency_graph(refs_json))
    dep_graph = {}
    if isinstance(dep_result, list):
        for item in dep_result:
            dep_graph[item["name"]] = item.get("depends_on")
    mvmem.dependency_graph = dep_graph
    logger.info(f"  Dependency graph: {dep_graph}")

    # --- (iii) Synthesizing References ---
    # Topological sort
    sorted_refs = _topological_sort(all_refs, dep_graph)
    logger.info(f"Synthesis order: {[r.name for r in sorted_refs]}")

    synthesized = {}  # name -> Path
    for ref in sorted_refs:
        logger.info(f"  Synthesizing reference: {ref.name} ({ref.type})")

        # Get dependent reference images for conditioning
        dep_images = []
        dep_name = dep_graph.get(ref.name)
        if dep_name and dep_name in synthesized:
            dep_images.append(synthesized[dep_name])

        # Generate reference image
        if ref.type == "environment":
            prompt = prompt_synthesize_environment(ref.description)
        else:
            prompt = prompt_synthesize_entity(ref.name, ref.description)

        img_path = refs_dir / f"{_safe_filename(ref.name)}.png"
        try:
            img_path = generate_image(
                prompt=prompt,
                reference_images=dep_images if dep_images else None,
                output_path=img_path,
            )
            synthesized[ref.name] = img_path

            mvmem.references.append(ReferenceFrame(
                name=ref.name,
                caption=ref.description,
                image_path=img_path,
                ref_type=ref.type,
            ))
            logger.info(f"    Saved: {img_path}")
        except Exception as e:
            logger.error(f"    Failed to synthesize {ref.name}: {e}")

    # --- Handle user-provided reference images (MLLM_cap) ---
    if ref_image_paths:
        for i, path in enumerate(ref_image_paths):
            p = Path(path)
            if p.exists():
                # Caption user image via MLLM
                try:
                    caption = call_mllm(
                        prompt_caption_user_image(), images=[p]
                    )
                    logger.info(f"  Captioned user ref {i}: {caption[:80]}...")
                except Exception as e:
                    logger.warning(f"  Failed to caption user ref {i}: {e}")
                    caption = f"User-provided reference image {i}"
                mvmem.references.append(ReferenceFrame(
                    name=f"user_ref_{i}",
                    caption=caption,
                    image_path=p,
                    ref_type="user",
                ))

    logger.info(f"MVMem initialized with {len(mvmem.references)} references")
    return mvmem


def _topological_sort(refs: list[Entity], dep_graph: dict[str, str | None]) -> list[Entity]:
    """Sort references by dependency order (roots first)."""
    name_to_ref = {r.name: r for r in refs}
    visited = set()
    result = []

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        dep = dep_graph.get(name)
        if dep and dep in name_to_ref:
            visit(dep)
        if name in name_to_ref:
            result.append(name_to_ref[name])

    for ref in refs:
        visit(ref.name)

    return result


def _safe_filename(name: str) -> str:
    """Convert entity name to safe filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")[:50]
