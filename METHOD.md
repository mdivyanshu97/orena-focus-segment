# DISCOVR-SEGMENT method

## Overview

DISCOVR-SEGMENT answers foreign-object questions about short surgical video
segments. It combines a merged Qwen3-VL checkpoint with deterministic
question routing, timestamp-aware frame sampling, constrained answer
generation, and batch-level output hardening.

The selected challenge container uses:

- base architecture: `Qwen/Qwen3-VL-4B-Instruct`;
- serving precision: merged bfloat16 checkpoint;
- default frame budget: 64;
- maximum frame side: 768 pixels;
- deterministic decoding;
- no network access or external inference service.

## Inputs and output

For each Grand Challenge batch, the container receives:

- `/input/request.json`: one request object per question;
- `/input/plain/<qID>.mp4`: the corresponding trimmed video segment;
- `/input/overlayed/<qID>.mp4`: a platform-provided relative-time variant;
- `/input/FO_definitions.json` and `/input/batch.json`.

The method decodes the plain video and draws its own absolute-time overlay when
needed. This preserves the time reference used during training. It writes an
ordered list of responses to `/output/answer.json`.

## Model and prompt

The released checkpoint is a bfloat16 merge of a rank-16, alpha-32 LoRA
adapter into Qwen3-VL-4B-Instruct. The adapter was trained across the FRAME,
SEGMENT, and PROCEDURE tracks and warm-started from a full-visibility
SSG-VQA adapter.

The system prompt contains:

- the bundled foreign-object definitions used during training;
- a compact surgical knowledge card;
- the instruction to answer only the supplied question.

The user prompt adds a mechanically inferred answer-format instruction. Output
cleanup then enforces formats such as binary, integer, class name, timestamp,
or timestamp list.

## Frame routing

| Question route | Evidence supplied to the model |
|---|---|
| Ordinary question without an explicit timestamp | 64 frames sampled across the complete segment |
| Non-cascade question containing timestamps | Up to two question-anchored windows, each centered on a stated timestamp; 64 frames are divided across the windows |
| Single-timestamp time question, pass 1 | 64 frames across the complete segment |
| Single-timestamp time question, pass 2 | 64 frames from a 30-second window centered on the first predicted timestamp |
| Multi-event timestamp-list question | Question-anchored sampling without the single-answer refinement cascade |

Question-anchored SEGMENT windows use a ±30-second half-width. The
single-timestamp refinement window is 30 seconds total, or approximately
±15 seconds around the first prediction.

## Temporal overlays

The platform overlay starts its clock at the beginning of each trimmed clip.
The model was trained with absolute procedure time, so DISCOVR-SEGMENT instead
decodes the plain clip and redraws the clock as:

`absolute time = request start_time + frame offset`.

The overlay is enabled for time-format questions and for questions whose
wording indicates temporal localization, duration, or ordering. A
question-anchored window always receives the overlay.

## Single-timestamp refinement

For a single-timestamp question:

1. The model predicts a timestamp from 64 frames spread across the full clip.
2. If that answer contains a valid timestamp, the timestamp is converted to a
   clip-relative center.
3. A 30-second window is clamped to the clip boundaries.
4. A second 64-frame pass produces the final timestamp.
5. If the refinement pass fails or returns no valid timestamp, the valid
   first-pass answer is retained.

Multi-event “time points” questions skip this cascade so that refinement around
one prediction cannot discard other valid events.

## Reliability

The runtime loads and warms the model once per batch. Each question has an
independent failure boundary. At batch exit, the output layer:

- preserves request order;
- emits one response for every recoverable `qID`;
- pads unanswered requests with an empty response;
- writes `answer.json` atomically through a temporary file.

A model-loading or GPU setup failure before any answer is produced is raised
loudly instead of being disguised as a successful all-empty submission.

## Reproducibility

The repository does not contain challenge videos or the 8.9 GB checkpoint.
Run:

```bash
python scripts/download_weights.py
python scripts/verify_release.py
./do_build.sh
```

The downloader pins the model repository revision, and the verifier checks the
selected source files and `model.safetensors` against the recorded SHA-256
values.
