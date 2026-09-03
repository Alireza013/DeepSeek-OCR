import ast
import base64
import re
from io import BytesIO
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .font import load_font


_REFERENCE_PATTERN = re.compile(
    r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL
)


def extract_grounding_references(text: str) -> List[Tuple[str, str, str]]:
    return _REFERENCE_PATTERN.findall(text or "")


def _boxes(value: str) -> Sequence[Sequence[float]]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError("Grounding coordinates must be a list")
    return parsed


def draw_bounding_boxes(image: Image.Image, refs, extract_images: bool = False):
    img_w, img_h = image.size
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    font = load_font(30)
    crops = []
    color_map = {}
    rng = np.random.default_rng(42)

    for _, label, raw_coordinates in refs:
        if label not in color_map:
            color_map[label] = tuple(int(value) for value in rng.integers(50, 255, 3))
        color = color_map[label]
        for box in _boxes(raw_coordinates):
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = (
                int(box[0] / 999 * img_w),
                int(box[1] / 999 * img_h),
                int(box[2] / 999 * img_w),
                int(box[3] / 999 * img_h),
            )
            x1, x2 = sorted((max(0, x1), min(img_w, x2)))
            y1, y2 = sorted((max(0, y1), min(img_h, y2)))
            if extract_images and label.lower() == "image":
                crops.append(image.crop((x1, y1, x2, y2)))
            width = 5 if label.lower() == "title" else 3
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            overlay_draw.rectangle((x1, y1, x2, y2), fill=color + (60,))
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            tag_y = max(0, y1 - text_height - 4)
            draw.rectangle((x1, tag_y, x1 + text_width + 4, tag_y + text_height + 4), fill=color)
            draw.text((x1 + 2, tag_y + 2), label, font=font, fill=(255, 255, 255))

    annotated.paste(overlay, (0, 0), overlay)
    return annotated, crops


def clean_output(text: str, include_images: bool = False) -> str:
    if not text:
        return ""
    result = text
    image_number = 0
    for full_match, label, _ in extract_grounding_references(text):
        if label.lower() == "image":
            replacement = f"\n\n**[Figure {image_number + 1}]**\n\n" if include_images else ""
            result = result.replace(full_match, replacement, 1)
            image_number += int(include_images)
        else:
            result = re.sub(rf"(?m)^[^\n]*{re.escape(full_match)}[^\n]*\n?", "", result)
    return result.strip()


def embed_images(markdown: str, crops: Sequence[Image.Image]) -> str:
    result = markdown
    for index, image in enumerate(crops, start=1):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        result = result.replace(
            f"**[Figure {index}]**",
            f"\n\n![Figure {index}](data:image/png;base64,{encoded})\n\n",
            1,
        )
    return result
