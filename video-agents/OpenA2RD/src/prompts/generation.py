"""A²RD Generation Prompts (Paper Section E.2).

Covers:
- E.2.1: Adaptive Segment Generation Mode
- E.2.2: Contiguous scene retrieval (V_rel)
- E.2.3: Relevant scene retrieval (S_rel)
- E.2.4: Relevant reference retrieval (R_rel)
- E.2.5: Best end-of-shot frame retrieval (MLLM_retr^img)
- E.2.6: Frame prompt generation (MLLM_pgen^img)
- E.2.7: Frame synthesis (TI2I)
- E.2.8: Video prompt generation (MLLM_pgen^vid)
"""


def prompt_adaptive_segment_mode(scenes: list[str]) -> str:
    """E.2.1 - Determine extrapolation vs interpolation mode (Eq. 2)."""
    scene_indices = list(range(len(scenes)))
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))

    return f"""Analyze the following video scenes and segment them into contiguous groups.

Scene Indices: {scene_indices}

Scenes:
{scenes_text}

A contiguous segment is a group of consecutive scenes that:
1. Share the same physical environment/location (e.g., interior vs exterior are different environments).
2. Are temporally continuous (no time jump between them).
3. Have action that flows directly from one scene to the next.

Segmentation Rules (Evaluate in order):
- Priority 1 (Moving Environments): In a moving environment (characters/subjects are moving, e.g., walking, driving, running), zoom in/out or framing changes remain continuous.
- Priority 2 (Static Environments): In a static environment (characters/subjects remain in the same place), a significant zoom in/out or framing change (e.g., wide shot to close-up) starts a new segment. This explicitly forces the current segment to end in a static state.
- Hard Boundaries: A new segment always starts when the environment/location changes fundamentally, or there is a time jump. Note that interior and exterior are considered different environments even if related (e.g., inside car vs outside car).

Reason over all the scenes one-by-one first, then return as JSON (list of lists of scene indices):

Reasoning:...

```json
[[...], [...], ...]
```

Note: Every scene index from {scene_indices} must appear exactly once."""


def prompt_contiguous_scenes(scenes: list[str]) -> str:
    """E.2.2 - Obtain V_rel: which previous scene is contiguous (Eq. 3)."""
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))

    return f"""Analyze the following video scenes and determine which previous scene (if any) is spatially and temporally contiguous with each scene.

Scenes:
{scenes_text}

For each scene, determine if there is AT MOST ONE previous scene that is contiguous with it. Two scenes are contiguous if:
1. They share the same physical environment/location
2. They are temporally continuous (no time jump between them)
3. The action flows directly from one to the other

If no previous scene is contiguous, return empty string "" for that scene.

Return as JSON:
```json
{{
  "0": "",
  "1": 0,
  "2": "",
  "3": 2,
  ...
}}
```

Note: Scene 0 always has empty string. Only select AT MOST ONE contiguous scene (the most recent one if multiple exist)."""


def prompt_relevant_scenes(scenes: list[str]) -> str:
    """E.2.3 - Obtain S_rel: top-k relevant previous scenes (Eq. 3)."""
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))

    return f"""Analyze the following video scenes and determine which PREVIOUS scenes contribute to the visual appearance of each scene.

Scenes:
{scenes_text}

For each scene, use chain-of-thought reasoning to identify the TOP 10 most relevant PREVIOUS scenes (with lower indices) for visual consistency.

Step 1: Reasoning
For each scene, think through:
- What objects appear in this scene? Which previous scenes first showed these objects?
- What characters appear in this scene? Which previous scenes established these characters?
- What physical spatial environment is this scene in? Which previous scenes showed this physical spatial arrangement?
- Which scenes are most visually important for maintaining consistency?

Step 2: Selection
Based on your reasoning, select the top 10 most relevant previous scenes for each category:
1. Objects - which previous scenes show objects that appear in this scene
2. Characters - which previous scenes show characters that appear in this scene
3. Environment - which previous scenes show the same physical spatial arrangement and layout

Return as JSON:
```json
{{
  "reasoning": {{
    "0": "Scene 0 reasoning...",
    "1": "Scene 1 reasoning...",
    ...
  }},
  "relevant_scenes": {{
    "0": {{"objects": [], "characters": [], "environment": []}},
    "1": {{"objects": [0], "characters": [0], "environment": [0]}},
    "2": {{"objects": [0, 1], "characters": [1], "environment": [0, 1]}},
    ...
  }}
}}
```

Note:
- Scene 0 always has empty lists.
- Each scene can only reference PREVIOUS scenes (lower indices).
- Each category MUST always include at most three immediately previous scenes.
- Limit to TOP 10 most relevant scenes per category, prioritizing most recent and most visually important."""


def prompt_relevant_references(scenes: list[str], anchors: dict[str, str]) -> str:
    """E.2.4 - Obtain R_rel: relevant global references per scene (Eq. 4)."""
    import json
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))
    anchors_text = json.dumps(
        {name: anchors[name] for name in anchors}, indent=2
    )

    return f"""Analyze the following video scenes and determine which reference entities and environments are relevant to each scene.

Scenes:
{scenes_text}

Reference Entities and Environments:
{anchors_text}

For each scene, identify which reference anchors (characters, objects, environments) appear or are relevant to that scene.

Return as JSON:
```json
{{
  "0": ["anchor_name1", "anchor_name2"],
  "1": ["anchor_name1", "anchor_name3"],
  ...
}}
```

Note:
- Only include anchors that are actually present or relevant in each scene.
- Anchor names must be selected from "{list(anchors.keys())}"."""


def prompt_retrieve_end_frame(old_caption: str, curr_scene: str, image_mapping: str) -> str:
    """E.2.5 - Retrieve best end-of-shot frame for scene continuation (Eq. 5)."""
    return f"""You are a video continuity judge. Given end-of-shot frames ({image_mapping}) from a multi-shot video and a new scene context, determine which frame is the best starting point for the new scene.

Original video context (contiguous scene):
{old_caption}

New scene context:
{curr_scene}

For each frame, analyze whether the subject, motion, and environment can naturally lead into the new scene. Then decide which frame (if any) is the best continuation point.

Return JSON:
```json
{{"best_index": <index or null if none>, "reasoning": "..."}}
```"""


def prompt_generate_frame_prompts(scenes: list[str], ref_descriptions: str) -> str:
    """E.2.6 - Generate frame prompts for all scenes (MLLM_pgen^img, Eq. 5)."""
    scenes_text = "\n".join(f"{i}. {s}" for i, s in enumerate(scenes))

    return f"""You are an expert Cinematographer creating visually consistent frames for a cohesive video story. Given the scenes, environment descriptions, generate detailed image prompts for the beginning frame of each scene and an ending frame for the final scene only.

Scenes:
{scenes_text}

Reference Anchor Descriptions:
{ref_descriptions}

Visual Consistency Guidelines:
1. **Characters:** Maintain identical appearance (face, hair, body type, age) across all scenes unless the story explicitly describes changes. Clothing may change if narratively justified.
2. **Visual Style:** Establish consistent artistic direction, color grading, and cinematographic approach across all frames.
3. **Environments:** Each scene's setting must align with the reference environments. Spatial layout and static elements of recurring locations must remain identical.
4. **Camera Angles:** When continuing from a previous scene, preserve the identical camera angle. You must explicitly say: "This scene maintains the exact camera angle as scene(s) X...".
5. **Lighting:** If the current scene is a continuation of a previous one within the same short period of time, preserve the identical lighting conditions.
6. **Strict Scene Adherence:** You MUST strictly follow the Scenes provided. DO NOT change camera shot types, actions, objects, props, or lighting conditions.
7. **First Appearance Rule:** When a character or significant object appears for the FIRST time in the video, the frame prompt MUST include ALL explicit and implicit appearance details. This serves as the visual reference for all subsequent scenes.
8. **Character Presence Rule:** If an established character is logically present in a scene, explicitly describe them in the frame.

Each frame prompt must explicitly describe:
- Visual style and artistic direction
- Natural lighting effects reflecting the scene's temporal context
- Character and object descriptions
- Environment specifications matching reference descriptions
- Camera angle details

Step 1: Scene frame reasoning: Reasoning about the scenes one by one about entity and environment appearances to ensure smooth transitions and strict visual consistency.

For each scene, analyze:
1. **Explicit Appearances:** What physical attributes are directly stated?
2. **Implied Appearances:** What can be logically inferred based on subsequent scenes?
3. **First Appearance Descriptions:** For first appearances, establish detailed visual descriptions as reference.

Step 2: Return a JSON dictionary with this structure. Scene indexes begin from 0:
```json
{{"frame_prompts": [{{"scene_index": 0, "begin_frame": "detailed image prompt with consistency reasoning"}}, {{"scene_index": <final_scene_index>, "end_frame": "detailed image prompt for final scene with consistency reasoning"}}]}}
```

Step 1 (at least 250 words):...
Step 2:..."""


def prompt_refine_frame_prompt(
    scenes_text: str,
    scene_index: int,
    curr_prompt: str,
    video_states: str,
    image_mapping: str,
) -> str:
    """E.2.6 - Refine frame prompt with memory context (MLLM_pgen^img, step 2)."""
    return f"""Analyze the video states and reference images to refine the current frame prompt for physical plausibility.

Scenes:
{scenes_text}

Current Scene Description (Scene {scene_index}):
(see scenes above)

Current Frame Prompt:
{curr_prompt}

Video States Memory:
{video_states}

Reference Images:
{image_mapping}

Step 1. **Scene Analysis**: Read all provided information and analyze:
- Is the scene contiguous with the previous scene?
- What visual elements must be carried forward from reference images and Video States Memory?
- What relative positional and directional states must be strictly maintained?

Step 2. **Analyze Current Prompt - Camera Angle & Physical Plausibility**: Analyze camera positioning and physical plausibility.

Step 3. **Other Camera Angle Possibilities**: If current camera angle cannot maintain strict positional states, analyze alternatives.

Step 4. **Refine the Prompt**: Only change required elements for physical plausibility and consistency. Keep the rest unchanged.
- MUST specify spatial details consistent with Video States Memory and reference images.
- Refined prompt MUST specify what environmental entities will be visible, will NOT be visible, and spatial details.

Step 5. **Refine your refined prompt**: Make sure all spatial details are clearly specified.

Return the refined prompt in JSON format:
```json
{{"refined_prompt": "the improved prompt"}}
```"""


def prompt_generate_video_prompt(
    scenes_text: str,
    scene_idx: int,
    current_scene: str,
    video_states: str,
    previous_video_prompt: str | None,
    has_end_frame: bool,
    scene_length: int = 8,
) -> str:
    """E.2.8 - Generate video prompt (MLLM_pgen^vid, Eq. 6)."""
    end_frame_text = (
        "that transitions from the beginning frame to the ending frame"
        if has_end_frame
        else "starting from the beginning frame"
    )

    prev_prompt_text = previous_video_prompt if previous_video_prompt else "None (this is the first scene)"

    image_mapping = "[Begin Frame, End Frame]" if has_end_frame else "[Begin Frame]"
    end_constraint = (
        f'Step 3. "End EXACTLY at the ending frame (Image 2). Visually inspect Image 2 and explicitly describe its composition, entity states, and environment in the video prompt as the final state the video must reach."'
        if has_end_frame
        else 'Step 3. "End naturally to best fulfill the Current Scene description"'
    )

    return f"""You are a professional video producer. Generate a detailed video prompt for a {scene_length}-second video {end_frame_text}.

Scenes:
{scenes_text}

Current Scene (Scene {scene_idx}):
{current_scene}

Video States Memory (for continuity):
{video_states}

Previous Scene Video Prompt (for camera/motion continuity):
{prev_prompt_text}

Your task is to generate a comprehensive, narrative prompt for the next video segment that meaningfully progresses the story from the existing prompts and storyline.
Step 1. Read the scenes and understand the overall narrative arc.
Step 2. Start EXACTLY from the beginning frame (Image 1).
{end_constraint}
Step 4. Maintain continuity from Previous Video States and Previous Scene Video Prompt:
- Entity states: character appearance, clothing, accessories must remain consistent
- Environment states: ground texture, landmarks, lighting must remain consistent
- Motion states: ongoing actions, camera movement direction must continue naturally
- CRITICAL: Camera direction must NOT reverse or contradict the previous scene.
**Explicitly specify these continuing elements in your video prompt**
Step 5. Ensure narrative/story logical progression.
Step 6. If the scene involves multiple distinct actions or camera transitions, break into temporal segments.

Image Mapping: {image_mapping}

In the video prompt, cover the following:
A professional {scene_length}-second video featuring [the entities].
The video opens with [opening frame description]... The [entities' activities]...
{"'Finally, the video concludes with [closing frame description]...'" if has_end_frame else "'The video progresses naturally to fulfill the scene description...'"}
[Any motions that must be continued from Previous Video States for continuity if applicable]...
The camera [camera movement/angle description, maintaining continuity with Previous Video States if applicable].
...

Respond with JSON:
```json
{{
  "video_reasoning": "explain your reasoning for how you structured the video prompt and whether temporal segments are needed",
  "video_prompt": "your detailed video prompt"
}}
```"""


SCENE_PROMPT_TEMPLATE = """A professional 8-second video featuring [the entities].
The video opens with [opening frame description]... The [entities' activities]...
[Any motions that must be continued from Previous Video States]...
The camera [camera movement/angle description]."""
