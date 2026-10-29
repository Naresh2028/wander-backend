"""Slice 1 of Wander: a vague vibe -> distinct destinations -> postcards in one locked art style."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Sequence

from .prompts import DESTINATION_SYSTEM, POSTCARD_PROMPT, STYLE_ANCHOR_PROMPT
from .schemas import Destination, DestinationSet, TripBrief

POSTCARD_SIZE = "1536x1024"


class WanderClient(Protocol):
    """What the pipeline needs. MetaClient implements it; tests use a fake."""

    def chat_json(self, system: str, user: str, model_cls, *, est_usd: float = ...): ...

    def generate_image(self, prompt: str, *, references: Sequence[Path] = ..., size=..., draft=..., **kw) -> bytes: ...


def _slug(text: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "-" for c in text)
    return "-".join(p for p in keep.split("-") if p)[:48] or "place"


def suggest_destinations(client: WanderClient, brief: TripBrief, n: int = 4) -> list[Destination]:
    user = (
        f"Traveler's vibe: {brief.vibe}\n"
        f"Trip length: {brief.days} days. Budget level: {brief.budget}.\n"
        f"Travelling from: {brief.travelling_from or 'not stated'}. Companions: {brief.companions or 'not stated'}.\n"
        f"Avoid: {', '.join(brief.avoid) if brief.avoid else 'nothing stated'}.\n"
        f"Return exactly {n} destinations."
    )
    result = client.chat_json(DESTINATION_SYSTEM, user, DestinationSet)
    dests = result.destinations[:n]
    if len(dests) < n:
        raise ValueError(f"Spark returned {len(dests)} destinations, expected {n}.")
    return dests


def ensure_style_anchor(client: WanderClient, state_dir: Path, *, draft: bool) -> Path:
    """The 'Wander look' reference image. Generated once (the cache makes re-runs free)."""
    path = Path(state_dir) / "style_anchor.webp"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(client.generate_image(STYLE_ANCHOR_PROMPT, size=POSTCARD_SIZE, draft=draft))
    return path


def make_postcards(
    client: WanderClient,
    anchor: Path,
    destinations: Sequence[Destination],
    out_dir: Path,
    *,
    draft: bool,
    workers: int = 4,
) -> dict[str, Path]:
    """One postcard per destination, all anchored on the same style reference."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(d: Destination) -> tuple[str, Path]:
        prompt = POSTCARD_PROMPT.format(name=d.name, country=d.country, visual_scene=d.visual_scene)
        data = client.generate_image(prompt, references=[anchor], size=POSTCARD_SIZE, draft=draft)
        path = out_dir / f"{_slug(d.name)}.webp"
        path.write_bytes(data)
        return d.name, path

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(pool.map(one, destinations))


def run_first_slice(
    client: WanderClient,
    brief: TripBrief,
    *,
    state_dir: Path,
    out_root: Path,
    n: int = 4,
    draft: bool = True,
) -> Path:
    """Run the whole slice and return the path of the generated index.html."""
    from .render import render_contact_sheet  # local import keeps the module graph simple

    run_dir = Path(out_root) / time.strftime("run-%Y%m%d-%H%M%S")
    dests = suggest_destinations(client, brief, n)
    anchor = ensure_style_anchor(client, state_dir, draft=draft)
    cards = make_postcards(client, anchor, dests, run_dir / "postcards", draft=draft)
    return render_contact_sheet(brief, dests, cards, run_dir)
