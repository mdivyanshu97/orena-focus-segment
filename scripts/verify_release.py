#!/usr/bin/env python3
"""Verify source files and, when downloaded, the merged model checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "inference.py": "4eaa5da09e5563154d8c9f9d9515f7c470492d89446ed0d8ae35ca877078372d",
    "requirements.txt": "307f1dd2031f13b0e696bb63fcce743a19da144d5417119fa4c24c1825906a1b",
    "resources/engines.py": "6ace41be45e22d2e00c989737fa1bc58817fa69b8cbc191a325b4bf9c8b82906",
    "resources/infer_format.py": "256165783ae8a6905639b8cb7c6015a8f791a9926d0e103eb93c54bd391f5509",
    "resources/overlay_decision.py": "28cb99fc6f43646f8cb5bab269ff197ba45e63173dfd67911e645b1fce5d4a2b",
    "resources/prompts.py": "15b7938f749943f99be3692de94cd9a5c439f3df14adcc9548fcd24e30c905cb",
    "resources/question_window.py": "6be06cf51e05efda69e59073638319416d30bac925cd2cbc465bcc2e263eb794",
    "resources/robust_io.py": "65a564595a83b11ef7f79b822f77d4834869894b60a876ac24e71cc97780de71",
    "resources/video_sampler.py": "4e63fe61b1a1b34e655d2c090a966fb910bb91d565d511fe3747a87e3a0fbf87",
    "resources/base/model.safetensors": "622fd66547b2ad88f9fcf9c74a22450f44b4c88cef8fcf1a9b464de2a51dcff3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failed = False
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.exists():
            if relative.startswith("resources/base/"):
                print(f"SKIP {relative}: run scripts/download_weights.py")
                continue
            print(f"MISS {relative}")
            failed = True
            continue
        actual = sha256(path)
        status = "OK" if actual == expected else "FAIL"
        print(f"{status} {relative} {actual}")
        failed |= actual != expected
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
