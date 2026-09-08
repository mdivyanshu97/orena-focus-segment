"""Batch-level I/O hardening shared by all three tracks.

The per-question `try/except` in `inference.py` protects against a bad *question*.
It does NOT protect the batch-level prologue — reading `request.json`, building the
system prompt, loading the model — and anything that raises there kills the whole
case before `/output/answer.json` exists. On the platform that is reported as
"The algorithm failed on one or more cases", which forfeits every question in the
case; a container that answers *badly* merely scores low, so always writing a
well-formed `answer.json` is strictly better than dying.

Two facilities:

`load_requests_tolerant` — `focus.Request` is a plain dataclass with six required
fields and no `**kwargs`, so `focus.load_requests` raises `TypeError` on *any*
unexpected key. The organizers already added an FO class mid-challenge (evaluator
v0.3.4), so a new field in `request.json` is a live risk, not a hypothetical one.
This loader keeps the known fields, drops unknown ones, coerces the numeric fields
(a JSON string `"132.5"` otherwise poisons `Request.duration`, which does a raw
`-` on two `str`), fills blank optionals, and skips only the individual rows it
cannot repair — never the batch.

`answer_every_question` — a context manager that guarantees one `Response` per
qID. Whatever happens inside, on exit it writes `answer.json` containing every
answer collected so far, padded with empty answers for any qID still missing, and
ordered to match `request.json`.

Both are deliberately dependency-free (stdlib + `focus`) so they cannot themselves
introduce an import-time failure.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

from focus import Request, Response, save_items

log = logging.getLogger(__name__)

# focus.Request's six required fields, with the coercion each one needs.
_STR_FIELDS = ("qID", "videoID", "procedure_type", "question")
_NUM_FIELDS = ("start_time", "end_time")


def _coerce_row(row: dict, index: int) -> Request | None:
    """Build a Request from a raw dict, or None if it cannot be repaired."""
    if not isinstance(row, dict):
        log.error("request[%d] is %s, not an object — skipped", index, type(row).__name__)
        return None
    qid = row.get("qID")
    if qid is None or str(qid) == "":
        # Without a qID the answer cannot be attributed to anything, so this row
        # is the one case that is genuinely unrecoverable.
        log.error("request[%d] has no qID — skipped", index)
        return None
    clean: dict = {}
    for f in _STR_FIELDS:
        v = row.get(f)
        clean[f] = "" if v is None else str(v)
    for f in _NUM_FIELDS:
        try:
            clean[f] = float(row.get(f))  # tolerates "132.5" and 132 alike
        except (TypeError, ValueError):
            # An unusable time bound is survivable: sampling falls back to the
            # whole clip and only the absolute-clock offset is affected.
            log.warning("request[%s] has unusable %s=%r — defaulting to 0.0",
                        clean["qID"], f, row.get(f))
            clean[f] = 0.0
    extra = set(row) - set(_STR_FIELDS) - set(_NUM_FIELDS)
    if extra:
        # Forward-compatibility: the batch survives fields added after this build.
        log.info("request[%s] carries unknown field(s) %s — ignored",
                 clean["qID"], sorted(extra))
    try:
        return Request(**clean)
    except Exception:  # noqa: BLE001 — dataclass signature changed under us
        log.exception("request[%s] could not be constructed — skipped", clean["qID"])
        return None


def load_requests_tolerant(path: Path) -> list[Request]:
    """Read request.json without letting one malformed row lose the batch."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]  # a single un-listed request object
    if not isinstance(rows, list):
        raise TypeError(f"request.json holds {type(rows).__name__}, expected a list")
    out = [r for i, row in enumerate(rows) if (r := _coerce_row(row, i)) is not None]
    if len(out) != len(rows):
        log.error("request.json: %d of %d rows unusable", len(rows) - len(out), len(rows))
    return out


def _qids_from_disk(path: Path) -> list[str]:
    """Best-effort qID list, used when the normal load never completed."""
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = [rows]
        return [str(r["qID"]) for r in rows
                if isinstance(r, dict) and r.get("qID") not in (None, "")]
    except Exception:  # noqa: BLE001 — this is already the fallback path
        log.exception("could not recover any qID from %s", path)
        return []


class SetupFailure(RuntimeError):
    """The environment itself is broken (GPU/driver/model load) — no question was
    ever going to be answered. Raised by inference.py around load + warm-up.

    Policy (2026-08-06, after the segment v3 silent 0.0): a SetupFailure must NOT
    be converted into an answer.json full of empty strings. On the platform a
    written answer.json marks the job SUCCEEDED and scores 0 on every question,
    which buries an infrastructure fault (there, the base image lacked kernels
    for the new RTX PRO 6000 GPUs) inside a "completed" submission. A loud FAILED
    job is diagnosable and re-runnable; a quiet 0.0 wastes a submission slot.
    """


@contextmanager
def answer_every_question(request_path: Path, output_path: Path,
                          responses: list[Response]):
    """Guarantee an answer.json with one Response per qID — with one deliberate
    exception, and never a torn file.

    * A mid-batch failure (bad question, decode error after real answers exist)
      still writes every answer collected so far, padded with empty answers for
      the remainder: partial credit beats none.
    * A SetupFailure with ZERO collected answers re-raises instead of writing.
      Empty-answer padding exists to save a partially-answered batch, not to
      dress an environment fault as a completed run (see SetupFailure).
    * The write is ATOMIC (tmp file + os.replace): a kill mid-write can no longer
      leave a truncated, unparseable answer.json — the platform sees either the
      complete file or none.
    """
    # NOTE deliberately NOT try/finally: the write must not live in a `finally`,
    # where it would also run while a SetupFailure propagates and re-create the
    # silent success this exists to prevent. The re-raise path skips the write by
    # construction — control never reaches it.
    try:
        yield
    except SetupFailure:
        if not responses:
            log.error("SETUP FAILURE with no answers — refusing to write an "
                      "all-empty answer.json; failing loudly instead")
            raise  # job must show FAILED, not a completed 0.0
        # Setup died after some answers existed (should not happen, but if it
        # does, those answers are real work): fall through and save them.
        log.exception("SetupFailure after %d answer(s) — keeping them", len(responses))
    except Exception:  # noqa: BLE001 — last resort: never leave /output empty
        log.exception("BATCH-LEVEL FAILURE — writing empty answers for unanswered qIDs")
    order = _qids_from_disk(request_path)
    by_id = {str(r.qID): r for r in responses}
    final = [by_id.get(q, Response(qID=q, content="", latency=0.0)) for q in order]
    # Anything answered that request.json did not list (should not happen, but
    # dropping a real answer would be worse than an extra row).
    final += [r for q, r in by_id.items() if q not in set(order)]
    padded = [q for q in order if q not in by_id]
    try:
        import os
        output_path.mkdir(parents=True, exist_ok=True)
        tmp = output_path / "answer.json.tmp"
        save_items(final, tmp)
        os.replace(tmp, output_path / "answer.json")  # atomic on POSIX
        log.info("Wrote %d response(s) (%d empty placeholder(s)).", len(final), len(padded))
    except Exception:  # noqa: BLE001 — truly nothing left to try
        log.exception("FAILED TO WRITE answer.json")
    # RUN_SUMMARY: the padding counter the platform's metrics JSON does not have
    # (questions_unanswered counts MISSING rows; pads are present-but-empty and
    # score as ordinary wrong answers). One fixed-format line, printed LAST so it
    # survives log truncation; padded qIDs listed (first 20) so a partial batch
    # is diagnosable from the job page alone.
    build_sha = "unknown"
    try:
        build_sha = (Path(__file__).parent / "BUILD_SHA").read_text().strip()
    except Exception:  # noqa: BLE001 — the summary must never be the thing that fails
        pass
    print(f"RUN_SUMMARY answered={len(order) - len(padded)} padded={len(padded)} "
          f"total={len(order)} build_sha={build_sha}"
          + (f" padded_qids={','.join(padded[:20])}" if padded else ""),
          flush=True)
