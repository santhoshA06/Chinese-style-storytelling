from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any
from app.config import settings

class StorageService:
    def __init__(self):
        self.base_dir = settings.output_dir

    def create_storybook_dir(self) -> tuple[str, Path]:
        storybook_id = f"storybook_{int(time.time())}"
        p = self.base_dir / storybook_id
        p.mkdir(parents=True, exist_ok=True)
        return storybook_id, p

    def save_json(self, path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)