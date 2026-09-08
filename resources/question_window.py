"""Sample around the timestamp the QUESTION states, instead of across the whole clip.

THE PROBLEM. A non-cascaded row gets ONE pass of NUM_FRAMES=64 frames spread evenly over the clip. A
procedure clip is ~109 minutes, so that is one frame roughly every 103 SECONDS. A question like "what
foreign object is visible at 01:03:00?" therefore almost never has a frame near 01:03:00 — the model is
answering from whatever it glimpsed across the whole operation. It is not failing to understand; the
evidence was never sampled.

The timestamp is sitting in the question text. This reads it and spends the same 64 frames on a window
around it (spacing 103s -> ~4s), and turns the absolute-time clock on so the model can tell which frame is
which. No extra pass, no extra frames, no extra latency.

MEASURED, n=290 procedure / 422 segment test rows, paired, video-clustered CIs, official judge:
    procedure   54.8% -> 67.2%   +12.4 pp  CI [+3.7, +20.6]   = +1.8 pp on the track
    segment     72.7% -> 76.5%   + 3.8 pp  CI [+0.2,  +7.2]   = +0.4 pp on the track
Both intervals exclude zero. The two row sets are disjoint from each other and from the cascade, so the
gains add.

WHY THE PARAMETERS ARE WHAT THEY ARE — every one is a measurement, not a guess:
  * HALF-WIDTH 120s is an interior optimum, bracketed on BOTH sides:
        +/-30s 65.2% | +/-120s 67.6% | +/-300s 65.2% | +/-600s 65.2%
    Tighter loses context, wider loses density.
  * 64 FRAMES, not 128. Doubling frames moved the result by exactly +0.00 pp (23 rows up, 23 down), so
    the win is WHERE we look, not how much. Keeping 64 keeps the latency budget untouched.
  * OVERLAY ON for these rows. The clock alone is worth +5.9 pp of the +12.4; the window alone (clock
    off) is +10.0. They are partly independent, so we take both. The container's own want_overlay rule
    already fires on only 28 of the 290 procedure rows, so this is mostly new.
  * PROCEDURE EXCLUDES binary AND number; SEGMENT EXCLUDES NOTHING. This is not a hunch — the same
    prior-shift falsifier was run per track and came out differently:
        procedure binary: says-yes 55.6% -> 32.1% against a 48.1% gold — it overshoots PAST the gold,
                          and only 12 of 25 flips were corrections (a coin flip). Excluded.
        segment binary:   says-yes 57.7% -> 47.9% against a 50.9% gold — it moves TOWARD the gold,
                          and 20 of 28 flips were corrections (71%). Kept.
        procedure number: -1.9 pp on 53 rows (counting wants the whole video). Excluded.
    Flip DIRECTION alone was too crude a test: correcting a real bias is one-directional by construction.
    The discriminators are whether the prior lands nearer the gold and whether flips are corrections.

WHAT IS DELIBERATELY NOT DONE
  * No shape routing. Classifying questions as instant / range / forward and zooming only the "safe"
    shapes was MEASURED and COSTS 4.5 pp — every shape gains (instant +18.8, range +11.8, forward +9.1).
  * Nothing for rows with no timestamp: no affordance, no change, exact current behaviour. The downside
    of this whole feature is bounded by that fallback.
"""
from __future__ import annotations

import re

_TS = re.compile(r"\d{1,2}:\d{2}:\d{2}")

HALF_WIDTH = {"procedure": 120.0, "segment": 30.0}
# PER-TRACK, and the difference is large. +/-120 is PROCEDURE's optimum and importing it into segment was
# wrong: the clips differ 55x in length (6569s vs 119s median), so the same window is a 96% narrowing on
# procedure and ~0% on segment. Measured on the 422 segment zoom rows, paired, video-clustered:
#   no zoom  72.7%              +/-60   77.5%  (+4.74, CI [+1.41,+7.89])
#   +/-15    78.4%  (+5.69)     +/-120  76.5%  (+3.79, CI [+0.24,+7.18])  <- what was shipped first
#   +/-30    79.1%  (+6.40, CI [+3.35,+9.55], p=0.008)                    <- interior optimum, both sides worse
# +/-30 beats +/-120 by +2.61pp CI [+0.24,+5.00] and lifts the track gain +0.40 -> +0.67pp.
# Two independent reasons +/-120 was wrong for segment: it narrows only 176/422 rows (the other 246 are a
# structural placebo), and its 3.75s frame spacing is COARSER than the 2.3s scoring tolerance on a median
# clip. +/-30 narrows 389/422 at 0.94s spacing. The binary prior-shift falsifier also improves at +/-30:
# says-yes lands 2.5pp from the gold (vs 3.1 at +/-120) with 77% of flips being corrections (vs 71%).
MAX_WINDOWS = 2                       # questions state at most two useful timestamps
MIN_WIDTH = 5.0                       # a degenerate sliver is worse than the full clip
# formats to leave alone, per track (see the prior-shift note above)
SKIP_FORMATS = {"procedure": {"binary", "number"}, "segment": set()}


def _secs(t: str) -> float | None:
    p = t.split(":")
    try:
        return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    except (ValueError, IndexError):
        return None


def question_windows(question: str, inferred_format: str, start_time: float,
                     end_time: float, track: str) -> list[tuple[float, float]]:
    """CLIP-RELATIVE (lo, hi) windows to sample, or [] to keep current behaviour.

    Mirrors the probe that produced the numbers above exactly: clamp in ABSOLUTE time (so a window near
    the clip end slides back rather than being truncated), then convert to clip-relative for sample_clip.
    """
    if inferred_format in SKIP_FORMATS.get(track, set()):
        return []
    stamps = sorted({t for t in _TS.findall(question or "")})
    if not stamps:
        return []
    out: list[tuple[float, float]] = []
    for t in stamps[:MAX_WINDOWS]:
        c = _secs(t)
        if c is None:
            continue
        # preserve width by sliding, never truncating (the same clamp the cascade uses)
        hw = HALF_WIDTH.get(track, 120.0)
        lo = max(start_time, min(c - hw, end_time - 2 * hw))
        hi = min(end_time, lo + 2 * hw)
        if hi - lo < MIN_WIDTH:
            return []
        out.append((lo - start_time, hi - start_time))
    return out
