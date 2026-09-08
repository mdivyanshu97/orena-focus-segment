"""Model inference engines for the ORena FOCUS benchmark.

Each engine wraps a VLM family behind a common interface::

    engine = build_engine("Qwen/Qwen3-VL-8B-Instruct", device="cuda:0")
    engine.load()
    raw_text = engine.generate(images=[PIL.Image, ...], question="...", system="...")

Images are passed as a list of PIL.Image (already sampled frames). For the
FRAME track this is a single image; for SEGMENT/PROCEDURE it is a list of
uniformly-sampled frames. Engines decide how to format the multi-image / video
message for their processor.

Supported families (auto-detected from the model id):
  * Qwen3-VL          (Qwen3VLForConditionalGeneration)
  * Qwen2.5-VL        (Qwen2_5_VLForConditionalGeneration)
  * Generic AutoModelForImageTextToText fallback (InternVL, etc. via chat template)
"""

from __future__ import annotations

import logging
from typing import Protocol

import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 64


class Engine(Protocol):
    model_id: str

    def load(self) -> None: ...

    def generate(
        self, images: list[Image.Image], question: str, system: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str: ...


# ── Qwen family (2.5-VL and 3-VL share the messages/vision-info API) ──────


class QwenVLEngine:
    """Engine for Qwen2.5-VL and Qwen3-VL instruct models."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto",
                 attn: str | None = None, load_4bit: bool = False,
                 adapter: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn = attn
        self.adapter = adapter  # optional PEFT/LoRA adapter dir applied after base load
        self.load_4bit = load_4bit
        self.model = None
        self.processor = None
        low = model_id.lower()
        # Qwen3-VL uses Qwen3VLForConditionalGeneration; Qwen3.5/3.6 (arch qwen3_5)
        # and Qwen2-VL/2.5-VL resolve via AutoModelForImageTextToText.
        self._is_qwen3vl = "qwen3-vl" in low or "qwen3_vl" in low or ("qwen3" in low and "vl" in low and "qwen3.5" not in low and "qwen3.6" not in low)
        # The name heuristic cannot see through a local directory: the container
        # loads from resources/base, which contains no "qwen3-vl". Fall back to the
        # checkpoint's declared architecture, which is authoritative.
        if not self._is_qwen3vl:
            try:
                import json as _json
                from pathlib import Path as _Path
                _cfg = _Path(model_id) / "config.json"
                if _cfg.is_file():
                    _arch = _json.loads(_cfg.read_text()).get("architectures") or []
                    self._is_qwen3vl = any("Qwen3VL" in a for a in _arch)
            except Exception:  # noqa: BLE001 — heuristic only; auto-class still works
                pass

    def load(self) -> None:
        from transformers import AutoProcessor
        logger.info("Loading %s …", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs: dict = {"torch_dtype": self.dtype, "device_map": self.device}
        if self.attn:
            kwargs["attn_implementation"] = self.attn
        if self.load_4bit:
            import torch as _t
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_t.bfloat16, bnb_4bit_use_double_quant=True,
            )
            kwargs["torch_dtype"] = _t.bfloat16
        # Resolve the right class robustly across Qwen2-VL / 2.5-VL / 3-VL.
        # AutoModelForImageTextToText reads the config's architecture, so it works
        # for plain Qwen2-VL checkpoints (e.g. SurgVidLM) and local paths too.
        if self._is_qwen3vl:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        else:
            from transformers import AutoModelForImageTextToText as Cls
        self.model = Cls.from_pretrained(self.model_id, **kwargs).eval()
        if self.adapter:
            from peft import PeftModel
            logger.info("Applying LoRA adapter %s …", self.adapter)
            self.model = PeftModel.from_pretrained(self.model, self.adapter).eval()
            # merge so generate() runs at full speed (no per-layer adapter overhead)
            try:
                self.model = self.model.merge_and_unload()
                logger.info("Adapter merged into base weights.")
            except Exception as e:  # merge can fail for some quantized bases; keep unmerged
                logger.warning("merge_and_unload failed (%s); running with adapter attached.", e)
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.max_length = None
        logger.info("Model ready (%s%s).", "Qwen3-VL" if self._is_qwen3vl else "Auto-ITT",
                    "+LoRA" if self.adapter else "")

    def _content(self, images: list[Image.Image], question: str) -> list[dict]:
        # Multi-image message: one image entry per frame, then the question.
        content: list[dict] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question})
        return content

    @torch.no_grad()
    def generate(self, images, question, system, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> str:
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._content(images, question)},
        ]
        # Qwen3.5/3.6 default to a <think> reasoning trace (too slow for the
        # latency budget). Suppress it via enable_thinking=False when supported.
        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ── Generic image-text-to-text fallback (InternVL3-HF, etc.) ──────────────


class GenericITTEngine:
    """Fallback using transformers AutoModelForImageTextToText + chat template.

    Works for models that expose an HF-native processor with image support
    (e.g. OpenGVLab/InternVL3-*-hf, llava-style). Passes images via the
    standard {"type": "image"} content blocks.
    """

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto",
                 attn: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn = attn
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        logger.info("Loading %s (generic ITT) …", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        kwargs: dict = {"torch_dtype": self.dtype, "device_map": self.device,
                        "trust_remote_code": True}
        if self.attn:
            kwargs["attn_implementation"] = self.attn
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).eval()
        logger.info("Model ready (generic ITT).")

    @torch.no_grad()
    def generate(self, images, question, system, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> str:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": question})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(
            self.model.device
        )
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ── InternVL (HF-native port, e.g. OpenGVLab/InternVL3_5-8B-HF) ───────────


class InternVLEngine:
    """Engine for the HF-native InternVL ports (repo id ending in '-HF').

    Uses AutoModelForImageTextToText + AutoProcessor.apply_chat_template with
    {"type":"image","image":PIL} blocks; no qwen_vl_utils needed. The processor
    pulls PIL images straight from the message dicts. trust_remote_code NOT
    required for the *-HF variant.
    """

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto",
                 attn: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.attn = attn
        self.model = None
        self.processor = None

    def load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        logger.info("Loading %s (InternVL) …", self.model_id)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        kwargs: dict = {"torch_dtype": self.dtype, "device_map": self.device}
        if self.attn:
            kwargs["attn_implementation"] = self.attn
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).eval()
        logger.info("Model ready (InternVL).")

    @torch.no_grad()
    def generate(self, images, question, system, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> str:
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question})
        messages = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed[0], skip_special_tokens=True).strip()


# ── MiniCPM-V (custom model.chat API, trust_remote_code) ──────────────────


class MiniCPMVEngine:
    """Engine for openbmb/MiniCPM-V-4_5 (and 2.6) via the repo's model.chat()."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "auto",
                 attn: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.attn = attn or "sdpa"  # "eager" unsupported by remote code
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        logger.info("Loading %s (MiniCPM-V) …", self.model_id)
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True,
            attn_implementation=self.attn, torch_dtype=torch.bfloat16,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        logger.info("Model ready (MiniCPM-V).")

    @torch.no_grad()
    def generate(self, images, question, system, max_new_tokens=DEFAULT_MAX_NEW_TOKENS) -> str:
        msgs = [{"role": "user", "content": list(images) + [question]}]
        answer = self.model.chat(
            msgs=msgs, tokenizer=self.tokenizer, system_prompt=system,
            sampling=True, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=max_new_tokens, enable_thinking=False,
            use_image_id=False, max_slice_nums=1, stream=False,
        )
        return (answer if isinstance(answer, str) else answer[0]).strip()


def build_engine(model_id: str, device: str = "cuda", dtype: str = "auto",
                 attn: str | None = None, load_4bit: bool = False,
                 engine: str | None = None, adapter: str | None = None) -> Engine:
    """Factory: pick the right engine for a model id (or force via `engine`)."""
    low = model_id.lower()
    # Qwen VL family: 2-VL/2.5-VL/3-VL, plus the newer Qwen3.5/Qwen3.6 (arch qwen3_5,
    # which are multimodal despite the name lacking "vl"), plus SurgVidLM.
    qwen_vl = ("qwen" in low and "vl" in low) or "surgvidlm" in low \
        or "qwen2vl" in low or "qwen2_vl" in low \
        or "qwen3.5" in low or "qwen3.6" in low or "qwen3_5" in low
    if engine == "qwen" or qwen_vl:
        return QwenVLEngine(model_id, device=device, dtype=dtype, attn=attn,
                            load_4bit=load_4bit, adapter=adapter)
    if "internvl" in low:
        return InternVLEngine(model_id, device=device, dtype=dtype, attn=attn)
    if "minicpm" in low:
        return MiniCPMVEngine(model_id, device=device, dtype=dtype, attn=attn)
    return GenericITTEngine(model_id, device=device, dtype=dtype, attn=attn)
