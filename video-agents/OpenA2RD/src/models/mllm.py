"""MLLM interface using Google Gemini API (via google-genai SDK)."""

import json
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from src.config import settings

logger = logging.getLogger(__name__)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def call_mllm(
    prompt: str,
    images: list[Path | str] | None = None,
    videos: list[Path | str] | None = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """Call Gemini MLLM with text + optional images/videos.

    Args:
        prompt: Text prompt
        images: List of image file paths to include
        videos: List of video file paths to include
        model: Gemini model name
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        Model response text
    """
    client = get_client()

    contents = []

    # Add images
    if images:
        for img_path in images:
            p = Path(img_path)
            if p.exists():
                contents.append(types.Part.from_image(Image.open(p)))
            else:
                logger.warning(f"Image not found: {img_path}")

    # Add videos
    if videos:
        for vid_path in videos:
            p = Path(vid_path)
            if p.exists():
                with open(p, "rb") as f:
                    video_bytes = f.read()
                contents.append(
                    types.Part.from_bytes(data=video_bytes, mime_type="video/mp4")
                )
            else:
                logger.warning(f"Video not found: {vid_path}")

    # Add text prompt
    contents.append(prompt)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=16384,
                ),
            )
            return response.text
        except Exception as e:
            logger.warning(f"MLLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def call_mllm_json(
    prompt: str,
    images: list[Path | str] | None = None,
    videos: list[Path | str] | None = None,
    model: str = "gemini-2.5-flash",
) -> dict | list:
    """Call MLLM and parse JSON from response."""
    response = call_mllm(prompt, images, videos, model)
    return extract_json(response)


def extract_json(text: str) -> dict | list:
    """Extract JSON from MLLM response text."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```json\s*\n?(.*?)```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object/array
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        # Find matching end
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No valid JSON found in response: {text[:500]}")
