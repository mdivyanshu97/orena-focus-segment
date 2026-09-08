"""Frame sampling for the video tracks (SEGMENT / PROCEDURE) that reproduces the
sampling + overlay used in TRAINING.

Two things must match training exactly or temporal questions degrade:

1. UNIFORM sampling of up to ``num_frames`` frames across the clip, longest-side
   capped at 768 (BILINEAR, downscale-only) — same as sampling.FrameSampler /
   extract_frames.

2. ABSOLUTE-TIME clock overlay. In training we sampled from the FULL video and burned
   the clock at ``frame_index / fps`` = absolute video time; the ``time``-format gold
   answers are on that absolute timeline (e.g. "00:10:12"). The challenge ships clips
   PRE-TRIMMED to [start_time, end_time], and its own ``overlayed/`` variant burns a
   CLIP-RELATIVE clock (starts at 00:00:00) — which would put the model in a different
   time reference than it trained on. So we IGNORE the provided overlayed/ variant,
   decode the ``plain/`` clip, and RE-DRAW the clock at absolute time:
       absolute_seconds = start_time + (frame_offset_in_clip / fps)
   matching training and the absolute-time gold answers.

The clock is drawn only when ``overlay=True`` (the caller enables it for temporal
questions only, exactly like run_benchmark's want_overlay).
"""
from __future__ import annotations

import os
from pathlib import Path

import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = None


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _get_font(img_w: int):
    global _FONT
    if _FONT is None:
        size = max(16, img_w // 28)
        # The bundled TTF comes first: the pytorch base image ships NO system
        # fonts, and a load_default() fallback renders a ~10px clock the model
        # was never trained on — silently breaking every time-format answer.
        bundled = os.path.join(os.path.dirname(__file__), "DejaVuSans-Bold.ttf")
        for path in (
            bundled,
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        ):
            try:
                _FONT = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        if _FONT is None:
            _FONT = ImageFont.load_default()
    return _FONT


def _draw_clock(im: Image.Image, seconds: float) -> Image.Image:
    """Burn an absolute-time clock top-left (yellow on black outline), as in training."""
    text = _fmt_ts(seconds)
    draw = ImageDraw.Draw(im)
    font = _get_font(im.size[0])
    x, y = 8, 6
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(255, 255, 0))
    return im


def _resize(im: Image.Image, max_long_side: int) -> Image.Image:
    w, h = im.size
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return im


def sample_clip(clip_path: Path, start_time: float, num_frames: int,
                overlay: bool, max_long_side: int = 768,
                window: tuple[float, float] | None = None) -> list[Image.Image]:
    """Uniformly sample up to ``num_frames`` frames from a pre-trimmed clip.

    ``start_time`` is the clip's absolute offset on the original video timeline; used
    only to draw the absolute-time clock when ``overlay`` is True. Decoding always
    starts at the clip's own beginning (the clip IS the window).

    ``window`` (lo, hi), in CLIP-RELATIVE seconds, restricts sampling to a sub-range —
    used by the temporal cascade to zoom around the model's previous answer. The clock
    stays on the absolute timeline (start_time + position), so zoomed passes read the
    same time reference the model trained on.
    """
    vr = decord.VideoReader(str(clip_path), ctx=decord.cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps()) or 5.0
    n = len(vr)
    if window is not None:
        lo = max(0, min(int(round(window[0] * fps)), n - 1))
        hi = max(lo + 1, min(int(round(window[1] * fps)), n))
    else:
        lo, hi = 0, n
    span = hi - lo
    k = min(num_frames, span)
    if k <= 1:
        indices = [lo]
    else:
        indices = [lo + int(round(i * (span - 1) / (k - 1))) for i in range(k)]
    # dedup preserving order
    seen: set[int] = set()
    indices = [i for i in indices if not (i in seen or seen.add(i))]

    batch = vr.get_batch(indices).asnumpy()  # (k, H, W, 3)
    out: list[Image.Image] = []
    for j, idx in enumerate(indices):
        im = _resize(Image.fromarray(batch[j]), max_long_side)
        if overlay:
            # absolute video time = clip offset + position within clip
            abs_seconds = float(start_time) + (idx / fps)
            im = _draw_clock(im, abs_seconds)
        out.append(im)
    return out
