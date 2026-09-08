#!/usr/bin/env python3
"""Download the exact public SEGMENT checkpoint into the Docker build tree."""

from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "Div97/orena-focus-segment-fullvis-w64"
REVISION = "ab709ec95ab4c5cc73eb97664191a8f8d76cc59a"
TARGET = Path(__file__).resolve().parents[1] / "resources" / "base"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=TARGET,
        ignore_patterns=["README.md", ".gitattributes"],
    )
    print(f"Downloaded {REPO_ID}@{REVISION} to {TARGET}")


if __name__ == "__main__":
    main()
