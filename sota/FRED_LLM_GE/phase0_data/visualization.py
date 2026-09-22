"""Development-only visual overlays with a hard held-out-label guard."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from .provenance import atomic_write_bytes
from .schema import Annotation, ProjectSplit


class HeldOutVisualizationError(PermissionError):
    pass


def render_annotation_overlay(
    image: Image.Image,
    annotations: tuple[Annotation, ...],
    *,
    project_split: ProjectSplit | str,
    destination: str | Path,
) -> None:
    split = ProjectSplit(project_split)
    if split == ProjectSplit.HELD_OUT_TEST:
        raise HeldOutVisualizationError(
            "routine visual overlays of official held-out test labels are prohibited"
        )
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    for annotation in annotations:
        draw.rectangle(annotation.box_xyxy, outline="red", width=2)
        draw.text(
            (annotation.box_xyxy[0], annotation.box_xyxy[1]),
            f"{annotation.track_id}: {annotation.original_class}",
            fill="yellow",
        )
    import io

    buffer = io.BytesIO()
    rendered.save(buffer, format="PNG")
    atomic_write_bytes(destination, buffer.getvalue())
