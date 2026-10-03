"""Request bodies of the web API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentRequest(BaseModel):
    method: str = "clicks"  # "clicks" (prototypes) or "learned" (head trained on labels)
    channel: int = 0
    z: int = 0
    pos: list[tuple[float, float]] = Field(default_factory=list)  # (y, x) in full-resolution pixels
    neg: list[tuple[float, float]] = Field(default_factory=list)
    profile_id: str | None = None
    settings: dict = Field(default_factory=dict)
    uncertainty: bool = False
    max_side: int = 1600


class StackJobRequest(BaseModel):
    method: str = "clicks"
    structures: list[dict] = Field(default_factory=list)  # Two or more: [{"name", "color", "pos"}] runs every structure
    channel: int = 0
    ref_z: int | None = None
    pos: list[tuple[float, float]] = Field(default_factory=list)
    neg: list[tuple[float, float]] = Field(default_factory=list)
    profile_id: str | None = None
    z_start: int = 0
    z_end: int | None = None
    z_step: int = 1
    settings: dict = Field(default_factory=dict)


class ProfileRequest(BaseModel):
    structures: list[dict] = Field(default_factory=list)  # Two or more saves a multi-structure profile
    name: str
    description: str = ""
    dataset_id: str
    channel: int = 0
    z: int = 0
    pos: list[tuple[float, float]]
    neg: list[tuple[float, float]] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)


class AnnotationRequest(BaseModel):
    channel: int
    z: int
    pos: list[tuple[float, float]] = Field(default_factory=list)
    neg: list[tuple[float, float]] = Field(default_factory=list)
    extra: list[dict] = Field(default_factory=list)  # Further structures: [{"name", "color", "pos"}]


class StructureSpec(BaseModel):
    name: str
    color: str = "#00c8f0"
    pos: list[tuple[float, float]] = Field(default_factory=list)


class MultiSegmentRequest(BaseModel):
    method: str = "clicks"  # "clicks" or "learned" (a model trained on labels with several structures)
    channel: int = 0
    z: int = 0
    profile_id: str | None = None  # A multi-structure profile, used instead of clicks
    structures: list[StructureSpec] = Field(default_factory=list)
    neg: list[tuple[float, float]] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)
    max_side: int = 1600


class LabelRequest(BaseModel):
    channel: int
    z: int
    mask_png: str | None = None   # Data URL of an edited mask at any resolution; None saves the last result
    structures: list[str] = Field(default_factory=list)  # Two or more: the red channel of mask_png holds the structure index


class ProfileFromHeadRequest(BaseModel):
    channel: int
    name: str
    description: str = ""


class HeadRequest(BaseModel):
    channel: int
    settings: dict = Field(default_factory=dict)
    kind: str = "auto"


class MaskSpec(BaseModel):
    channel: int
    source: str = "current"   # "current", "profile", "learned", "label" or "reference"
    profile_id: str | None = None


class HistoRequest(BaseModel):
    z: int
    bone: MaskSpec
    cells: MaskSpec
    contact_um: float = 3.0
    settings: dict = Field(default_factory=dict)
    max_side: int = 1600


class PathRequest(BaseModel):
    path: str


class RoiRequest(BaseModel):
    polygon: list[tuple[float, float]] | None = None  # (y, x) vertices in full-resolution pixels


class MetaRequest(BaseModel):
    reference_channel: int | None = None
    notes: str | None = None
    voxel_um_override: tuple[float, float, float] | None = None  # (z, y, x) in micrometres
    group: str | None = None  # Study group, such as "control" or "treated"
    settings: dict | None = None  # Segmentation settings last used on this dataset, restored when it is reopened
