"""Data structures for MVMem (Multimodal Video Memory).

MVMem := {M_1, ..., M_N} ∪ R ∪ D
Each segment memory M_j := {T_j, F_j, V_j}
  - T_j: Textual States (Visual Arcs, Spatial Relations, Camera States)
  - F_j: Frames {F_j^begin, F_j^end}
  - V_j: Video segment
R: Global reference frames (synthesized + user-provided)
D: Prompt database for MAPO
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Entity:
    name: str
    description: str
    type: str = "character"  # character | object | environment
    depends_on: str | None = None


@dataclass
class VisualArc:
    """Tracks entity identity, changes, and motion across a segment."""
    entity_name: str
    identity: str = ""
    identity_changes: str = ""
    motion: str = ""


@dataclass
class SpatialRelation:
    subject: str
    relation: str
    object: str


@dataclass
class TextualStates:
    """T_j: structured textual representation of segment j's world state."""
    visual_arcs: list[VisualArc] = field(default_factory=list)
    spatial_relations: list[SpatialRelation] = field(default_factory=list)
    camera: str = ""


@dataclass
class SegmentMemory:
    """M_j := {T_j, F_j, V_j} for segment j."""
    segment_index: int
    scene_context: str  # S_j
    textual_states: TextualStates = field(default_factory=TextualStates)
    # T_{j+1}^F: frame-level textual states extracted from begin frame
    frame_textual_states: TextualStates | None = None
    begin_frame_path: Path | None = None
    end_frame_path: Path | None = None
    video_path: Path | None = None
    generation_mode: str = "extrapolation"  # extrapolation | interpolation


@dataclass
class ReferenceFrame:
    """A global reference frame (entity or environment)."""
    name: str
    caption: str
    image_path: Path
    ref_type: str = "entity"  # entity | environment | user


@dataclass
class PromptDBEntry:
    """MAPO prompt database entry D := {(P, P*, Q, l)}."""
    original_prompt: str
    refined_prompt: str
    rubric_scores: dict = field(default_factory=dict)
    label: str = "pos"  # pos | neg
    embedding: list[float] = field(default_factory=list)


@dataclass
class MVMem:
    """Multimodal Video Memory: M := {M_1,...,M_N} ∪ R ∪ D."""
    segments: list[SegmentMemory] = field(default_factory=list)
    references: list[ReferenceFrame] = field(default_factory=list)
    prompt_db: list[PromptDBEntry] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    dependency_graph: dict[str, str | None] = field(default_factory=dict)

    # Precomputed retrieval results (computed once over all scenes)
    contiguous_scenes: dict[int, int | None] = field(default_factory=dict)  # V_rel
    relevant_scenes: dict[int, dict] = field(default_factory=dict)  # S_rel
    relevant_refs: dict[int, list[str]] = field(default_factory=dict)  # R_rel

    def get_segment(self, idx: int) -> SegmentMemory | None:
        for s in self.segments:
            if s.segment_index == idx:
                return s
        return None

    def add_or_update_segment(self, seg: SegmentMemory):
        for i, s in enumerate(self.segments):
            if s.segment_index == seg.segment_index:
                self.segments[i] = seg
                return
        self.segments.append(seg)

    def get_reference_by_name(self, name: str) -> ReferenceFrame | None:
        for r in self.references:
            if r.name == name:
                return r
        return None

    # --- Checkpoint save/load ---

    def save(self, path: Path):
        """Save MVMem state to JSON checkpoint."""
        import json

        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [_serialize(v) for v in obj]
            if isinstance(obj, dict):
                return {str(k): _serialize(v) for k, v in obj.items()}
            return obj

        data = _serialize(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "MVMem":
        """Load MVMem state from JSON checkpoint."""
        import json

        data = json.loads(path.read_text())
        mem = cls()

        # Restore entities
        for e in data.get("entities", []):
            mem.entities.append(Entity(**e))

        # Restore references
        for r in data.get("references", []):
            r["image_path"] = Path(r["image_path"])
            mem.references.append(ReferenceFrame(**r))

        # Restore segments
        for s in data.get("segments", []):
            ts = s.pop("textual_states", {})
            fts = s.pop("frame_textual_states", None)
            for key in ("begin_frame_path", "end_frame_path", "video_path"):
                if s.get(key):
                    s[key] = Path(s[key])
                else:
                    s[key] = None
            # Drop extra keys that aren't in the dataclass
            valid_keys = {f.name for f in SegmentMemory.__dataclass_fields__.values()}
            extra = {k: s.pop(k) for k in list(s) if k not in valid_keys}
            seg = SegmentMemory(**s)
            seg.textual_states = TextualStates(camera=ts.get("camera", ""))
            if fts:
                seg.frame_textual_states = TextualStates(camera=fts.get("camera", ""))
            # Restore extra attrs (like _video_prompt)
            for k, v in extra.items():
                setattr(seg, k, v)
            mem.segments.append(seg)

        # Restore prompt_db
        for p in data.get("prompt_db", []):
            mem.prompt_db.append(PromptDBEntry(**p))

        # Restore precomputed retrieval
        mem.dependency_graph = data.get("dependency_graph", {})
        mem.contiguous_scenes = {int(k): v for k, v in data.get("contiguous_scenes", {}).items()}
        mem.relevant_scenes = {int(k): v for k, v in data.get("relevant_scenes", {}).items()}
        mem.relevant_refs = {int(k): v for k, v in data.get("relevant_refs", {}).items()}

        return mem
