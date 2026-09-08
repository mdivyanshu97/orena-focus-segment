"""Decide whether to burn the absolute-time clock overlay onto a question's frames.

In training/eval (run_benchmark.want_overlay) the clock was drawn when the question's
capability was temporal — specifically ``answer_format == 'time'`` OR the primary
capability was one of {temporal_localization, duration_estimation, temporal_ordering}.

At inference we have neither the capability nor answer_format, only the question text
(and our inferred format). We approximate want_overlay from text.

MEASURED on 20,560 seg+proc GT questions (42% truly got overlay):
  * time-format-only  : 98.3% agreement, 0 false-positives, but MISSES 355 non-time
                        temporal questions (duration/ordering) that trained WITH a clock.
  * moderate-words    : 87.4% agreement, only 86 false-negatives (0.4% — temporal Qs
                        that would WRONGLY lose their clock), at the cost of 2510
                        false-positives (an extra clock on temporal-adjacent questions;
                        the model saw clocks throughout training and tolerates them).

We choose MODERATE-WORDS: on the video tracks (each 40% of the prize, temporal-heavy)
the priority is not to STARVE a temporal question of its clock. A harmless extra clock
on a non-temporal question costs far less than a missing clock on a real "when/how-long/
order" question. This is a tunable knob — A/B against time-format-only once GPUs free up.
"""
from __future__ import annotations

_TEMPORAL_WORDS = (
    "when", "how long", "duration", "before", "after", "first", "last",
    "earlier", "later", "order", "sequence", "which point", "at what time",
    "timestamp", "how much time",
)


def want_overlay(question: str, inferred_format: str) -> bool:
    """True if the absolute-time clock should be burned onto this question's frames."""
    if inferred_format == "time":
        return True
    s = (question or "").lower()
    return any(w in s for w in _TEMPORAL_WORDS)
