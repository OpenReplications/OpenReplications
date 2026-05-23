# OpenA2RD

Open-source replication of **A²RD: Agentic Autoregressive Diffusion for Long Video Consistency** ([arXiv:2605.06924](https://arxiv.org/abs/2605.06924)).

> Original paper by Do Xuan Long, Yale Song, Min-Yen Kan, Tomas Pfister, Long T. Le (Google Cloud AI Research, NUS).

## What is A²RD?

A²RD is an agentic autoregressive architecture for long video synthesis that decouples creative synthesis from consistency enforcement. It generates video segment-by-segment through a **Retrieve-Synthesize-Refine-Update** closed-loop cycle with three core components:

1. **Multimodal Video Memory (MVMem)** — tracks video world states across modalities (text states, frames, videos)
2. **Adaptive Segment Generation** — switches between extrapolation/interpolation modes per segment
3. **Hierarchical Test-Time Self-Improvement (HITS)** — self-improves frames and videos via MLLM judges + MAPO prompt optimization

## Architecture

```
User Context (P) + Storyline (S) + Optional Refs (R^u)
    │
    ▼
┌──────────────────────────────────┐
│  MVMem Initialization            │
│  - Plan entities & environments  │
│  - Build dependency DAG          │
│  - Synthesize global references  │
└──────────────┬───────────────────┘
               │
    ┌──────────▼──────────┐
    │  For each segment i │◄─────────────────────┐
    │                     │                      │
    │  1. Retrieve        │  (from MVMem)        │
    │  2. Adaptive Mode   │  (extrap/interp)     │
    │  3. Synthesize      │                      │
    │     - Boundary Frames (TI2I + HITS)        │
    │     - Video Segment  (TI2V + HITS)         │
    │  4. Update MVMem    │──────────────────────┘
    └─────────────────────┘
               │
               ▼
         Final Long Video
```

## Quick Start

```bash
# Clone
git clone https://github.com/OpenReplications/OpenReplications.git
cd OpenReplications/video-agents/OpenA2RD

# Setup
cp .env.example .env
# Edit .env with your Gemini API key

# Install
uv sync

# Run
uv run python -m src.main --storyline examples/chef_story.yaml
```

## Configuration

See `.env.example` for all configuration options. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) | - |
| `TI2I_BACKEND` | Image generation backend | `nano_banana` |
| `TI2V_BACKEND` | Video generation backend | `veo` |
| `MAX_REFINEMENT_ITERATIONS_FRAME` | HITS iterations for frames | `2` |
| `MAX_REFINEMENT_ITERATIONS_VIDEO` | HITS iterations for videos | `2` |

## Models Used

| Component | Original Paper | This Replication |
|-----------|---------------|-----------------|
| MLLM | Gemini 3 Flash | Gemini 3 Flash (via API) |
| TI2I | Nano Banana 2 | Nano Banana 2 / SDXL (local) |
| TI2V | Veo 3.1 | Veo 3.1 / LTX-Video / Wan 2.2 (local) |

## Project Structure

```
src/
├── main.py              # Entry point
├── config.py            # Configuration and env loading
├── memory/
│   ├── mvmem.py         # Multimodal Video Memory
│   ├── schema.py        # Memory data structures
│   └── prompt_db.py     # MAPO prompt database
├── pipeline/
│   ├── init.py          # MVMem initialization (planning, deps, ref synthesis)
│   ├── retrieve.py      # Context retrieval from MVMem
│   ├── segment_gen.py   # Adaptive segment generation (extrap/interp)
│   ├── frame_synth.py   # Boundary frame synthesis + HITS
│   ├── video_synth.py   # Video segment synthesis + HITS
│   └── hits.py          # Hierarchical Test-Time Self-Improvement
├── prompts/
│   ├── mvmem_init.py    # E.1: MVMem initialization prompts
│   ├── generation.py    # E.2: A²RD generation prompts
│   ├── judges.py        # E.3: HITS judge prompts
│   ├── mapo.py          # E.4: MAPO refinement prompts
│   └── evaluation.py    # E.5: Consistency evaluation prompts
├── models/
│   ├── mllm.py          # MLLM interface (Gemini)
│   ├── ti2i.py          # Text-Image-to-Image interface
│   └── ti2v.py          # Text-Image-to-Video interface
└── evaluation/
    ├── metrics.py       # Automatic metrics (ViCLIP, DINOv3, etc.)
    ├── narrative.py     # Narrative coherence evaluation
    └── consistency.py   # MLLM-Judge consistency evaluation
```

## References

```bibtex
@article{long2026a2rd,
  title={A$^2$RD: Agentic Autoregressive Diffusion for Long Video Consistency},
  author={Long, Do Xuan and Song, Yale and Kan, Min-Yen and Pfister, Tomas and Le, Long T.},
  journal={arXiv preprint arXiv:2605.06924},
  year={2026}
}
```

## License

MIT
