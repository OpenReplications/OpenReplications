"""MAPO (Memory-Augmented Prompt Optimization) Prompts (Paper Section E.4).

MAPO retrieves similar past refinement cases from the prompt database D,
contrasts positive and negative examples to derive actionable lessons,
and applies targeted edits to produce refined prompts.
"""


def prompt_mapo_feedback_reasoning(
    original_prompt: str,
    rubric_scores: dict,
    similar_positive_cases: list[dict],
    similar_negative_cases: list[dict],
) -> str:
    """E.4 - MAPO: Analyze feedback and derive refinement lessons."""
    pos_text = "\n\n".join(
        f"Positive Case {i+1}:\n"
        f"  Original: {c['original_prompt'][:200]}...\n"
        f"  Refined: {c['refined_prompt'][:200]}...\n"
        f"  Scores: {c['rubric_scores']}"
        for i, c in enumerate(similar_positive_cases[:5])
    ) or "No positive cases available."

    neg_text = "\n\n".join(
        f"Negative Case {i+1}:\n"
        f"  Original: {c['original_prompt'][:200]}...\n"
        f"  Refined: {c['refined_prompt'][:200]}...\n"
        f"  Scores: {c['rubric_scores']}"
        for i, c in enumerate(similar_negative_cases[:5])
    ) or "No negative cases available."

    return f"""You are a prompt optimization expert. Analyze the feedback from judges and derive actionable refinement lessons.

Current Prompt:
{original_prompt}

Current Rubric Scores:
{rubric_scores}

Similar Positive Refinement Cases (improvements that worked):
{pos_text}

Similar Negative Refinement Cases (refinements that made things worse):
{neg_text}

Tasks:
1. **Feedback Reasoning**: Analyze each low score (<8) and identify the root cause.
2. **Lesson Synthesis**: From the positive and negative cases, derive 5-10 actionable lessons (e.g., "Replace abstract scene references with concrete physical anchors", "Specify professional hand positions").
3. **Prompt Refinement**: Apply the lessons to produce a refined prompt that directly addresses the failure modes.

Guidelines from positive cases:
- What patterns made refinements successful?
- What specific changes improved scores?

Guidelines from negative cases:
- What patterns made refinements fail?
- What should be avoided?

Respond with JSON:
```json
{{
  "feedback_reasoning": "analysis of each low score and root causes",
  "lessons": ["lesson 1", "lesson 2", ...],
  "refined_prompt": "the improved prompt with targeted edits applied"
}}
```"""


def prompt_mapo_edit_mode(
    original_prompt: str,
    rubric_scores: dict,
    issue_description: str,
) -> str:
    """MAPO Edit mode: target a single issue for surgical fix."""
    return f"""You are a prompt editor. Make a minimal, targeted edit to fix a single issue.

Original Prompt:
{original_prompt}

Current Scores:
{rubric_scores}

Issue to Fix:
{issue_description}

Rules:
- Only change the specific part related to the issue
- Keep everything else exactly the same
- The edit should be surgical and minimal

Respond with JSON:
```json
{{
  "edit_description": "what was changed and why",
  "refined_prompt": "the minimally edited prompt"
}}
```"""
