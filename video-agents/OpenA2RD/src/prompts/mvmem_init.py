"""MVMem Initialization Prompts (Paper Section E.1).

Three MLLM calls for initialization:
- MLLM_plan^bg: Plan background/environment references
- MLLM_plan^ent: Plan entity references
- MLLM_dep: Build dependency DAG
- MLLM_cap: Caption user-provided reference images
"""

import json


def prompt_plan_entities(user_prompt: str, scenes: list[str], ref_captions: str = "") -> str:
    """E.1.1 - MLLM_plan^ent: Extract entities from storyline (Equation 1)."""
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))
    ref_section = f"\nUser-provided Reference Images:\n{ref_captions}" if ref_captions else ""

    return f"""You are an expert cinematographer. Deeply analyze this video story:

User Prompt:
{user_prompt}
{ref_section}

Scenes:
{scenes_text}

Deep Analysis Task:
1. Determine the key characters and objects that will be shown in the video from the user prompt. Write a concise description of how each one originally looks. Focus on only the key characters and objects with motion.

For each character or object, provide:
- 'name': A unique identifier (e.g., "The Protagonist", "Red Silk Dress").
- 'description': A brief description alone without any other details (e.g., appearance, clothing, colors, style) consistent with relevant backgrounds.

2. Generate the list if entities strictly as a JSON object:
```json
[{{"name": "...", "description": "..."}}]
```

3. Next, reflect back to the storyline. Are there any dynamic objects or characters in the storyline that were missed?

4. Finally, refine the list of characters and objects with their short descriptions as a JSON array.

Reasoning:
1. ...
2. ...
3. ...
4. ...
```json
[{{"name": "...", "description": "..."}}]
```"""


def prompt_plan_environments(user_prompt: str, scenes: list[str], ref_captions: str = "") -> str:
    """E.1.1 - MLLM_plan^bg: Extract environments from storyline (Equation 1)."""
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))
    ref_section = f"\nUser-provided Reference Images:\n{ref_captions}" if ref_captions else ""

    return f"""You are an expert spartial cinematographer. Deeply analyze this video story:

User Prompt:
{user_prompt}
{ref_section}

Scenes:
{scenes_text}

Deep Analysis Task:
Step 1. First, identify what are the main environments that the movie takes place in.

Step 2. Then, for each single environment, list what objects, furniture, and elements MUST be inside this space that facilitate all the scenes in the story smoothly. Do not combine multiple environments into one.

Step 3. Next, reflect back to the storyline. For all scenes within/related to that environment, is there anything missing from that environment?

Step 4. Finally, refine the list of environments with key and detailed descriptions of necessary things only.

Step 5. Return your step-level reasonings and deep analysis in detailed text form.

Notes: Do not create unnecessary/transition environments. Stick closely to the story context and planned scenes.

Reasoning:
1. ...
2. ...
3. ...
4. ...
5. ...

```json
[{{"environment_name": "...", "detailed_spartial_description": "description of a single unified environment with all necessary elements"}}]
```"""


def prompt_dependency_graph(refs_json: str) -> str:
    """E.1.1 - MLLM_dep: Build dependency DAG over references."""
    return f"""You are given a set of named visual references (characters, objects, environments) for a video. For each reference, identify what single other reference it depends on for visual consistency during image synthesis.

References:
{refs_json}

Guidelines:
- Environments/backgrounds are often roots, but may depend on other environments.
- Characters typically depend on their primary environment.
- Objects typically depend on the environment or character they are associated with.
- No cycles allowed.

Return strictly as JSON list:
```json
[
  {{"name": "ref_name", "depends_on": "other_ref_name or null"}},
  ...
]
```"""


def prompt_caption_user_image() -> str:
    """MLLM_cap: Caption user-provided reference images."""
    return """Describe this reference image in detail for use in video generation. Focus on:
1. Main subjects and their appearance (clothing, features, pose)
2. Environment and setting
3. Lighting and mood
4. Style and artistic direction

Provide a concise but comprehensive caption."""


def prompt_synthesize_environment(env_description: str) -> str:
    """E.1.2 - Environment synthesis prompt for TI2I."""
    return (
        f"Generate a single unified image viewing this entire environment with all its "
        f"elements following the {env_description}. All elements must be in one cohesive "
        f"scene, not split into multiple sub-images or panels. Do not include any humans "
        f"unless explicitly specified."
    )


def prompt_synthesize_entity(name: str, description: str) -> str:
    """E.1.3 - Entity synthesis prompt for TI2I."""
    return (
        f"Generate a professional, high-fidelity, single image of {name}: {description}. "
        f"A single, centered, full image, cinematic lighting strictly on a solid, clean, "
        f"all-white background. If {name} is an object, exclude all human or animal faces "
        f"unless it's explicitly required."
    )
