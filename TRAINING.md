# Training provenance

The released checkpoint is a bfloat16 merge of a LoRA adapter into
`Qwen/Qwen3-VL-4B-Instruct`.

Recorded configuration:

- adapter role: continued adapter;
- LoRA rank / alpha: `16 / 32`;
- tracks: FRAME, SEGMENT, and PROCEDURE;
- primary SFT export: `alltracks_train_v2_64f.jsonl`;
- warm start: full-visibility SSG-VQA adapter;
- merge method: `peft.merge_and_unload`;
- serving frame budget: 64 frames.

The original container-embedded merge record is published with the model
weights as `merge_provenance.json`. Its local filesystem paths are historical
provenance and are not required for inference.

Raw challenge videos and challenge-provided annotations are not redistributed.
The final selected inference image does not depend on private runtime services
or external network access.

The full reconstructed recipe, data counts, frame-generation procedure,
optimizer settings, historical dataset-snapshot note, and selected artifact
hashes are documented in [METHOD_DESCRIPTION.md](METHOD_DESCRIPTION.md).
