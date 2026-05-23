"""HITS Self-Refinement Prompts: MLLM-Judge (Paper Section E.3).

Covers:
- E.3.1: Extract frame states (MLLM_ext^img)
- E.3.2: Extract video states (MLLM_ext^vid)
- E.3.3: Judge frame - consistency over images (Eq. 7)
- E.3.4: Judge frame - spatial logicalness (Eq. 7)
- E.3.5: Judge frame - textual states (Eq. 7)
- E.3.6: Judge frame - basic quality (Eq. 7)
- E.3.7: Judge video - inter consistency (Eq. 9)
- E.3.8: Judge video - intra consistency and quality (Eq. 9)
"""


def prompt_extract_frame_states(scene_index: int, scene_description: str, frame_prompt: str) -> str:
    """E.3.1 - Extract entity states from a synthesized frame (MLLM_ext^img)."""
    return f"""Analyze this frame and extract the states of all entities.

Current Scene Description (Scene {scene_index}):
{scene_description}

Frame Prompt:
{frame_prompt}

Instructions:
1. Identify every entity (characters, objects, environments) in the frame.

2. For each entity, extract:
   - type: "character", "object", or "environment"
   - identity: Features and appearance that identify this entity across scenes:
     * Physical features (body type, facial structure, distinctive marks)
     * Clothing items and colors
     * Hair style and color
     * Accessories
     * For objects: shape, design, color, material
     * For environments: architectural features, layout, defining characteristics
   - state: Current observable state that may change between scenes:
     * Position: Relative position to other entities ONLY (e.g., "on the left side of X", "behind/perpendicular to Y", "between X and Y"). DO NOT use frame-relative terms like "center", "foreground", "left side of frame".
     * Orientation: Which direction the entity faces relative to other entities or environment features ONLY.
     * Depth: Relative depth to other entities.
     * Posture/arrangement (for characters: standing/sitting, limb positions; for objects: open/closed, arrangement)
     * Condition (clean/dirty, intact/damaged, etc.)

   Be specific with colors (exact shades) and directions. All positions and orientations must be relative to other entities, NOT to the frame or camera.

3. For spatial_relations, describe how entities relate spatially to each other.

Return the scene graph in this exact JSON format:
```json
{{
  "entities": {{
    "entity_name": {{
      "type": "character|object|environment",
      "identity": "features and appearance that identify this entity",
      "state": "current observable state"
    }}
  }},
  "spatial_relations": [
    {{"subject": "entity1", "relation": "relation_type", "object": "entity2"}}
  ]
}}
```"""


def prompt_extract_video_states(scene_index: int, scene_description: str, entities_json: str) -> str:
    """E.3.2 - Extract states from full video segment (MLLM_ext^vid)."""
    return f"""Analyze this video segment comprehensively. Provide at least 400 words total across all sections.

Current Scene Description (Scene {scene_index}):
{scene_description}

Known Entity States (from begin frame):
{entities_json}

Extract the following from the full video segment:

1. **Supplementing Missing Elements**
- Identify any NEW entities (characters, objects, environments) that appear after the begin frame but were not in the known entity states.
- For each new entity, provide a full appearance description.

2. **Identity Changes**
- For all entities (known and new), describe any appearance details revealed throughout the video that were not visible in the begin frame.
- For characters: focus on hair, clothing, physical appearance.
- For objects: focus on color, shape, condition, visible attributes.
- For environments: focus on lighting, weather, background elements, and spatial arrangements.

3. **Motions**
- For each element (known and new), describe its motion using the template: "At the beginning, [...]. In the middle, [...]. At the end, [...]."

4. **Camera Dynamics**
- Describe camera movement, angle changes, zoom, and panning throughout the video using the same temporal template.

Return as JSON:
```json
{{
  "new_entities": {{
    "<entity_name>": "full appearance description"
  }},
  "identity_changes": {{
    "<entity_name>": "changes observed"
  }},
  "motions": {{
    "<entity_name>": "At the beginning, [...]. In the middle, [...]. At the end, [...]."
  }},
  "camera": "At the beginning, [...]. In the middle, [...]. At the end, [...]."
}}
```"""


def prompt_judge_frame_consistency(
    scenes_text: str,
    scene_index: int,
    scene_description: str,
    frame_prompt: str,
    image_mapping: str,
) -> str:
    """E.3.3 - Judge frame consistency over images (Eq. 7)."""
    return f"""Evaluate the generated frame against reference images for consistency and progression.

Scenes:
{scenes_text}

Current Scene (Scene {scene_index}):
{scene_description}

Frame Prompt:
{frame_prompt}

Image Mapping:
{image_mapping}

What's wrong with the current image compared to its references with mappings specified in Image Mapping?

3. Entity Reference Consistency (1-10): Compare each character and object one-by-one against reference images. Check for appearance changes, color changes, clothing changes, distinctive feature changes that are NOT explicitly described in the Frame Prompt.

4. Environment Reference Consistency (1-10): Compare the environment against reference images. Check for spatial conflicts, position inconsistencies, architectural changes that are NOT explicitly described in the Frame Prompt.

5. Narrative Progression (1-10): The frame must NOT be identical to previous reference frames unless the Frame Prompt explicitly describes a static scene. There must be visible change aligning with the storyline.

Respond with JSON:
```json
{{
  "entity_reference_consistency": <score 1-10>,
  "environment_reference_consistency": <score 1-10>,
  "narrative_progression": <score 1-10>,
  "reasoning": "detailed explanation of consistency and progression issues"
}}
```"""


def prompt_judge_frame_spatial(
    scenes_text: str,
    scene_index: int,
    scene_description: str,
    frame_prompt: str,
    image_mapping: str,
) -> str:
    """E.3.4 - Judge frame spatial logicalness (Eq. 7)."""
    return f"""Carefully examine the spatial arrangements and object positions across these images.

Scenes:
{scenes_text}

Current Scene (Scene {scene_index}):
{scene_description}

Frame Prompt:
{frame_prompt}

Image Mapping:
{image_mapping}

Note: Images labeled "anchor_*" show the canonical spatial layout and object arrangements. The current frame MUST match these anchor layouts unless the Frame Prompt explicitly describes changes.

What's wrong with the current image compared to existing images regarding the spatial environment? Reasoning in at least 250 words.

Respond with JSON:
```json
{{
  "spatial_logicalness": <score 1-10, where 10 means nothing wrong>,
  "reasoning": "detailed analysis of spatial issues or 'nothing wrong'"
}}
```"""


def prompt_judge_frame_states(
    scenes_text: str,
    scene_index: int,
    scene_description: str,
    video_states: str,
    current_states: str,
) -> str:
    """E.3.5 - Judge frame textual states consistency (Eq. 7)."""
    return f"""Compare the extracted states with relevant previous scenes and the storyline to identify discrepancies.

Scenes:
{scenes_text}

Current Scene (Scene {scene_index}):
{scene_description}

Video States Memory:
{video_states}

Current Scene States:
{current_states}

Instructions:
Reasoning in at least 250 words to critically analyze consistency smartly:

Step 1: **Analyze Objects**
- What objects appear in the current scene's extracted states?
- How do these objects compare to their states in relevant previous scenes?
- What discrepancies exist (if any)?
- Are these discrepancies justified by the current scene description?

Step 2: **Analyze Characters**
- What characters appear in the current scene's extracted states?
- How do these characters compare to their states in relevant previous scenes?
- What discrepancies exist?
- Are these discrepancies justified?

Step 3: **Analyze Environment**
- What physical spatial environment appears in the current scene's extracted states?
- How does this compare to its state in relevant previous scenes?
- Strictly spot any spatial discrepancies or illogical environmental arrangements.

Return as JSON:
```json
{{
  "objects": "detailed reasoning about objects consistency...",
  "objects_state_score": <score 1-10>,
  "characters": "detailed reasoning about characters consistency...",
  "characters_state_score": <score 1-10>,
  "environment": "detailed reasoning about environment consistency...",
  "environment_state_score": <score 1-10>
}}
```"""


def prompt_judge_frame_quality(frame_prompt: str, video_states: str) -> str:
    """E.3.6 - Judge frame basic quality (Eq. 7)."""
    return f"""Evaluate if the generated frame successfully matches the following prompt and meets quality standards:

Frame Prompt:
{frame_prompt}

Current Frame States:
{video_states}

Evaluate and score each criterion on a scale of 1-10:
1. Instruction Following (1-10): Does the frame faithfully capture every critical component specified in the prompt, especially lighting conditions and environmental details?
2. Physical Plausibility (1-10): Is the frame physically realistic? All objects, characters, and environments must appear physically plausible. The frame must represent one unified moment in space and time. NO split screens, panels, or multiple disconnected scenes.

Respond with JSON:
```json
{{
  "instruction_following": <score 1-10>,
  "physical_plausibility": <score 1-10>,
  "reasoning": "explanation of any issues found in detail"
}}
```"""


def prompt_judge_video_inter_consistency(
    scenes_text: str,
    prev_scene_idx: int,
    prev_scene: str,
    curr_scene_idx: int,
    curr_scene: str,
    video_states: str,
) -> str:
    """E.3.7 - Judge video inter-consistency (Eq. 9)."""
    return f"""Analyze event consistency between two contiguous scenes.

Scenes:
{scenes_text}

Previous Contiguous Scene (Scene {prev_scene_idx}):
{prev_scene}

Current Scene (Scene {curr_scene_idx}):
{curr_scene}

Video States Memory:
{video_states}

Evaluate the following between the two contiguous video segments:

1. Inter Entity Consistency (1-10): Do the same characters maintain consistent appearance across both segments?
2. Inter Environment Consistency (1-10): Does the environment remain consistent across the transition?
3. Inter Motion Consistency (1-10): Do motions and actions flow naturally from one segment to the next?
4. Camera Consistency (1-10): Is the camera angle and movement consistent across the transition?

Respond with JSON:
```json
{{
  "inter_entity_consistency": <score 1-10>,
  "inter_environment_consistency": <score 1-10>,
  "inter_motion_consistency": <score 1-10>,
  "camera_consistency": <score 1-10>,
  "reasoning": "detailed analysis"
}}
```"""


def prompt_judge_video_intra_quality(
    scene_idx: int,
    scene_description: str,
    video_prompt: str,
    video_states: str,
) -> str:
    """E.3.8 - Judge video intra-consistency and basic quality (Eq. 9)."""
    return f"""Evaluate this video segment for intra-consistency and quality.

Current Scene (Scene {scene_idx}):
{scene_description}

Video Prompt:
{video_prompt}

Video States:
{video_states}

Evaluate:
1. Character State Consistency (1-10): Do characters maintain consistent appearance within this segment?
2. Object State Consistency (1-10): Do objects maintain consistent appearance within this segment?
3. Environment State Consistency (1-10): Does the environment remain consistent within this segment?
4. Instruction Following (1-10): Does the video follow the video prompt faithfully?
5. Physical Plausibility (1-10): Is the video physically realistic?
6. Narrative Progression (1-10): Does the video make meaningful narrative progress?

Respond with JSON:
```json
{{
  "character_state": <score 1-10>,
  "object_state": <score 1-10>,
  "environment_state": <score 1-10>,
  "instruction_following": <score 1-10>,
  "physical_plausibility": <score 1-10>,
  "narrative_progression": <score 1-10>,
  "reasoning": "detailed analysis"
}}
```"""
