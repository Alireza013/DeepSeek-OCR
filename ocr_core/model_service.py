import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer

from .config import MODEL_CONFIGS, MODEL_NAME, TASK_PROMPTS
from .output_formatter import (
    clean_output,
    draw_bounding_boxes,
    embed_images,
    extract_grounding_references,
)


class OCRResult:
    def __init__(self, text, markdown, raw, annotated=None, crops=None):
        self.text = text
        self.markdown = markdown
        self.raw = raw
        self.annotated = annotated
        self.crops = crops or []

    def as_tuple(self):
        return self.text, self.markdown, self.raw, self.annotated, self.crops


class OCRModelService:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                use_safetensors=True,
            ).eval().cuda()
        return self._tokenizer, self._model

    @staticmethod
    def _prompt(task: str, custom_prompt: str):
        if task == "Custom":
            return f"<image>\n{custom_prompt.strip()}", "<|grounding|>" in custom_prompt
        if task == "Locate":
            return f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image.", True
        task_config = TASK_PROMPTS[task]
        return task_config["prompt"], task_config["has_grounding"]

    def infer(self, image: Image.Image, mode: str, task: str, custom_prompt: str) -> OCRResult:
        if image is None:
            return OCRResult("Upload an image", "", "")
        if task in ("Custom", "Locate") and not (custom_prompt or "").strip():
            return OCRResult("Enter a prompt first", "", "")
        image = ImageOps.exif_transpose(image.convert("RGB"))
        config = MODEL_CONFIGS[mode]
        prompt, has_grounding = self._prompt(task, custom_prompt or "")
        tokenizer, model = self._load()

        temp_path: Optional[str] = None
        output_dir: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temporary:
                image.save(temporary, "JPEG", quality=95)
                temp_path = temporary.name
            output_dir = tempfile.mkdtemp(prefix="deepseek-ocr-")
            captured = StringIO()
            with redirect_stdout(captured):
                model.infer(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    image_file=temp_path,
                    output_path=output_dir,
                    base_size=config.base_size,
                    image_size=config.image_size,
                    crop_mode=config.crop_mode,
                )
            raw_output = captured.getvalue()
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)
            if output_dir:
                import shutil
                shutil.rmtree(output_dir, ignore_errors=True)

        raw = "\n".join(
            line for line in raw_output.splitlines()
            if not any(marker in line for marker in ("image:", "other:", "PATCHES", "====", "BASE:", "%|", "torch.Size"))
        ).strip()
        if not raw:
            return OCRResult("No text extracted", "", "")

        annotated = None
        crops = []
        if has_grounding and "<|ref|>" in raw:
            refs = extract_grounding_references(raw)
            if refs:
                annotated, crops = draw_bounding_boxes(image, refs, True)
        markdown = embed_images(clean_output(raw, True), crops)
        return OCRResult(clean_output(raw), markdown, raw, annotated, crops)
