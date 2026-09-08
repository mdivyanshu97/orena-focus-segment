"""
ORena SAVE FOCUS challenge — SEGMENT track.

Our real submission: LoRA-fine-tuned Qwen3-VL-4B-Instruct (combined all-tracks adapter,
merged into resources/base). Answers a batch of (video-clip, question) pairs.

Inputs (mounted read-only at /input):
  request.json            LIST of focus.Request — one per question
  plain/<qID>.mp4         the video clip (already trimmed to [start_time, end_time])
  overlayed/<qID>.mp4     clip with a CLIP-RELATIVE clock — we do NOT use this (see below)
  FO_definitions.json     FO class definitions (we use our bundled copy instead)
  batch.json              convenience index

Output: /output/answer.json — LIST of focus.Response.

KEY DESIGN NOTES (mirror the FRAME container, plus video specifics):
* Model + prompts + infer_format identical to the frame track (shared resources/).
* answer_format inferred from the question (infer_format, 98.9% on segment GT, all
  strict formats 100%); training-matched hint via prompts.build_question; output via
  prompts.clean_answer.
* System prompt from bundled FO_definitions.txt (matches training byte-for-byte), NOT
  /input's (which adds an unseen 'Absorbable Hemostatic Agent' class).
* FRAME SAMPLING + OVERLAY reproduce training (video_sampler.sample_clip):
  - uniform up to NUM_FRAMES frames, longest-side<=768 (BILINEAR, downscale-only);
  - ABSOLUTE-TIME clock re-drawn on the plain clip (start_time + offset), because the
    time-format gold answers are absolute and the provided overlayed/ variant is
    clip-relative (starts 00:00:00) — a different reference than the model trained on.
  - overlay applied only to temporal questions (overlay_decision.want_overlay).
"""

import json
import logging
import re
import sys
import time
from pathlib import Path

# decord must be imported AFTER torch (known CUDA-init ordering issue). engines imports
# torch; import it before video_sampler (which imports decord).
from focus import Request, Response

RESOURCES_PATH = Path(__file__).parent / "resources"
sys.path.insert(0, str(RESOURCES_PATH))

import prompts  # noqa: E402
from engines import QwenVLEngine  # noqa: E402  (imports torch)
from infer_format import infer_format  # noqa: E402
from overlay_decision import want_overlay  # noqa: E402
from question_window import question_windows  # noqa: E402
from robust_io import SetupFailure, answer_every_question, load_requests_tolerant  # noqa: E402
from video_sampler import sample_clip  # noqa: E402  (imports decord, after torch)

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
VIDEO_DIR = INPUT_PATH / "plain"          # decode plain clips; we re-draw the clock ourselves
MODEL_PATH = str(RESOURCES_PATH / "base")

NUM_FRAMES = 64           # matches the deployed adapter's training budget (see resources/base/merge_provenance.json)
MAX_LONG_SIDE = 768
# 32-token default cap: answers are short in every ordinary format, and the cap removes
# the runaway-generation tail the knowledge card provoked (measured: 10 timeouts/224s
# max uncapped -> zero timeouts/7.5s max capped, same accuracy). Genuine multi-event
# "time points" questions need room for comma-separated timestamp lists, so only that
# mechanically identifiable route receives the 128-token allowance used by the upstream
# challenge example.
MAX_NEW_TOKENS = 32
MULTI_TIME_MAX_NEW_TOKENS = 128

# ── temporal mini-cascade (single-timestamp time questions only) ─────────────
# One refinement pass on a ±15s window around the first answer. Measured on all 1,577
# single-ts segment time rows (2026-08-03): 0.504 -> 0.550 accuracy (122 wins / 49
# losses), p99 11.9s against the 15s budget. Clip median is ~119s, so the window
# shrinks the frame gap from ~1.9s to ~0.5s where precision is the failure mode.
# The former 9s self-imposed guard was removed: the platform budget is pooled, the
# evaluated "shipped" route always used the refinement answer, and measured workloads
# have substantial pooled headroom. Later-pass failures retain the first-pass answer.
CASCADE_WIDTH = 30.0      # full window width (p1 ± 15s)
_TS_RE = re.compile(r"(\d+):(\d{2}):(\d{2})")


def _ts_seconds(text: str):
    m = _TS_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def _is_multi_time(question: str, fmt: str) -> bool:
    return fmt == "time" and "time points" in (question or "").lower()


def _is_single_time(question: str, fmt: str) -> bool:
    # Multi-event "time points" questions have 1..N golds; zooming on one answer
    # would drop the others. Cascade only the single-timestamp templates.
    return fmt == "time" and not _is_multi_time(question, fmt)


def _max_new_tokens(question: str, fmt: str) -> int:
    return MULTI_TIME_MAX_NEW_TOKENS if _is_multi_time(question, fmt) else MAX_NEW_TOKENS


def clip_path_for(req: Request) -> Path:
    return VIDEO_DIR / f"{req.qID}.mp4"


def run() -> int:
    t_start = time.monotonic()
    log.info("=== ORena SAVE FOCUS — SEGMENT inference start ===")

    # Every answer collected so far. `answer_every_question` reads this list on exit
    # and writes one Response per qID in request.json no matter what happens inside
    # the block — a case that produces no answer.json is forfeited outright, which is
    # strictly worse than one that answers badly.
    responses: list[Response] = []
    with answer_every_question(INPUT_PATH / "request.json", OUTPUT_PATH, responses):
        requests = load_requests_tolerant(INPUT_PATH / "request.json")
        if not requests:
            log.error("request.json contains no usable requests")
            return 1
        log.info("Batch of %d question(s)", len(requests))

        # include_knowledge=True ships the surgical knowledge card (A/B: +0.023 official-judge
        # segment pre_eval, gains in causal/functional/purpose buckets; zero latency cost
        # under the route-scoped token caps).
        system = prompts.system_prompt(include_fo_defs=True, include_knowledge=True)
        log.info("System prompt: %d chars (FO defs + knowledge card)", len(system))
        if len(system) < 500:
            log.error("System prompt too short — bundled FO_definitions.txt missing?")
            return 1

        # Model load + warm-up are wrapped as SetupFailure: if the environment
        # itself is broken (no usable GPU, kernel mismatch, missing weights), no
        # question was ever going to be answered, and the job must FAIL loudly
        # rather than "succeed" with 2000 empty answers — that is exactly how the
        # segment v3 submission became a completed 0.0 on the RTX PRO 6000s.
        try:
            log.info("--- Loading merged model once (%s) ---", MODEL_PATH)
            engine = QwenVLEngine(MODEL_PATH, device="cuda")
            engine.load()
            log.info("Model loaded (setup %.1fs). Warm-up…", time.monotonic() - t_start)
            from PIL import Image
            _warm = engine.generate([Image.new("RGB", (64, 64))],
                                    "Is a foreign object visible?",
                                    system, max_new_tokens=4)
            log.info("Warm-up ok -> %r", _warm)
        except Exception as exc:
            raise SetupFailure(f"model load / warm-up failed: {exc!r}") from exc

        n_failed = 0
        t_batch = time.monotonic()
        for i, req in enumerate(requests, start=1):
            t0 = time.monotonic()
            try:
                fmt = infer_format(req.question)
                ov = want_overlay(req.question, fmt)
                # QUESTION-ANCHORED ZOOM (resources/question_window.py). Only for rows the mini-cascade
                # will NOT touch: cascaded rows already get a refinement pass, and the two populations
                # must stay disjoint or the measured deltas stop adding. Measured +3.8 pp on the 422
                # segment rows that state a timestamp and are not cascaded, CI [+0.2, +7.2], at identical
                # compute. No timestamp -> [] -> exact current behaviour.
                wins = [] if _is_single_time(req.question, fmt) else question_windows(
                    req.question, fmt, req.start_time, req.end_time, "segment")
                if wins:
                    ov = True          # the zoom branch always burns the clock; keep the log honest
                    per = max(8, NUM_FRAMES // len(wins))
                    frames = []
                    for w in wins:
                        frames.extend(sample_clip(clip_path_for(req), req.start_time, per,
                                                  True, MAX_LONG_SIDE, window=w))
                else:
                    frames = sample_clip(clip_path_for(req), req.start_time, NUM_FRAMES, ov,
                                         MAX_LONG_SIDE)
                question = prompts.build_question(req.question, fmt)
                max_new = _max_new_tokens(req.question, fmt)
                raw = engine.generate(frames, question, system, max_new_tokens=max_new)
                answer = prompts.clean_answer(raw, fmt)
                # mini-cascade: one ±15s refinement pass for single-timestamp questions.
                # ITS OWN try/except, ON PURPOSE. Found by the container-vs-benchmark audit
                # (2026-08-12): this block used to sit inside the per-question try whose handler
                # does `answer = ""`, so a crash in the REFINEMENT pass discarded the valid pass-1
                # answer and scored the row zero. The refinement can only ever improve the answer,
                # so its failure must be a no-op, not a loss. Zero measured cost — the platform has
                # reported 0 forfeited / 0 unanswered on every segment run.
                if _is_single_time(req.question, fmt):
                    try:
                        p1 = _ts_seconds(answer)
                        if p1 is not None:
                            clip_dur = float(req.end_time) - float(req.start_time)
                            w = min(CASCADE_WIDTH, clip_dur)
                            lo = p1 - float(req.start_time) - w / 2.0
                            lo = max(0.0, min(lo, clip_dur - w))
                            frames2 = sample_clip(clip_path_for(req), req.start_time, NUM_FRAMES,
                                                  True, MAX_LONG_SIDE, window=(lo, lo + w))
                            raw2 = engine.generate(frames2, question, system,
                                                   max_new_tokens=max_new)
                            a2 = prompts.clean_answer(raw2, fmt)
                            if _ts_seconds(a2) is not None:
                                answer = a2
                    except Exception:
                        log.exception("[%d/%d] qID=%s refinement pass failed; keeping the "
                                      "first-pass answer %r", i, len(requests), req.qID, answer)
            except Exception:
                n_failed += 1
                log.exception("[%d/%d] qID=%s failed; empty answer", i, len(requests), req.qID)
                answer = ""
            latency = time.monotonic() - t0
            responses.append(Response(qID=req.qID, content=answer, latency=latency))
            log.info("[%d/%d] qID=%s fmt=%s ov=%s zoom=%s cap=%s -> %r (%.2fs)",
                     i, len(requests), req.qID, locals().get("fmt", "?"),
                     locals().get("ov", "?"), len(locals().get("wins") or []),
                     locals().get("max_new", "?"), answer, latency)

        log.info("Inference: %d answered (%d failed) in %.1fs",
                 len(responses), n_failed, time.monotonic() - t_batch)

    log.info("Total %.1fs", time.monotonic() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
