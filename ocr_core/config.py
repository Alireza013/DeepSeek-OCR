from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    base_size: int
    image_size: int
    crop_mode: bool


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Gundam": ModelConfig(1024, 640, True),
    "Tiny": ModelConfig(512, 512, False),
    "Small": ModelConfig(640, 640, False),
    "Base": ModelConfig(1024, 1024, False),
    "Large": ModelConfig(1280, 1280, False),
}

TASK_PROMPTS = {
    "Markdown": {
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "has_grounding": True,
    },
    "Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "Locate": {
        "prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.",
        "has_grounding": True,
    },
    "Describe": {
        "prompt": "<image>\nDescribe this image in detail.",
        "has_grounding": False,
    },
    "Custom": {"prompt": "", "has_grounding": False},
}

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
