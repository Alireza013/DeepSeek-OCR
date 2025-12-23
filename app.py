import os
import re
import sys
import fitz
import torch
import shutil
import spaces
import base64
import tempfile
import warnings
import numpy as np
import gradio as gr
from io import StringIO, BytesIO
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont, ImageOps

MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda()

#  SAFE FONT LOADING 
def load_font(size=30):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

#  CONFIG 
MODEL_CONFIGS = {
    "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False}
}

TASK_PROMPTS = {
    "📋 Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "📝 Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "📍 Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "🔍 Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "✏️ Custom": {"prompt": "", "has_grounding": False}
}

#  EXTRACTION 
def extract_grounding_references(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)

def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = load_font(30)

    crops = []
    color_map = {}
    np.random.seed(42)

    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )

        color = color_map[label]
        coords = eval(ref[2])
        color_a = color + (60,)

        for box in coords:
            x1 = int(box[0] / 999 * img_w)
            y1 = int(box[1] / 999 * img_h)
            x2 = int(box[2] / 999 * img_w)
            y2 = int(box[3] / 999 * img_h)

            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))

            width = 5 if label == 'title' else 3

            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops

def clean_output(text, include_images=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            text = re.sub(rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)
    return text.strip()

def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(
            f'**[Figure {i+1}]**',
            f'\n\n![Figure {i+1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown

#  MAIN INFERENCE 
@spaces.GPU(duration=60)
def process_image(image, mode, task, custom_prompt):
    if image is None:
        return "Upload image", "", "", None, []

    if task in ["✏️ Custom", "📍 Locate"] and not custom_prompt.strip():
        return "Enter prompt", "", "", None, []

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)

    config = MODEL_CONFIGS[mode]

    if task == "✏️ Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task == "📍 Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    image.save(tmp.name, "JPEG", quality=95)
    tmp.close()

    out_dir = tempfile.mkdtemp()

    stdout = sys.stdout
    sys.stdout = StringIO()

    model.infer(
        tokenizer=tokenizer,
        prompt=prompt,
        image_file=tmp.name,
        output_path=out_dir,
        base_size=config["base_size"],
        image_size=config["image_size"],
        crop_mode=config["crop_mode"],
    )

    raw_output = sys.stdout.getvalue()
    sys.stdout = stdout

    result = "\n".join([
        l for l in raw_output.split("\n")
        if not any(s in l for s in ["image:", "other:", "PATCHES", "====", "BASE:", "%|", "torch.Size"])
    ]).strip()

    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)

    if not result:
        return "No text extracted", "", "", None, []

    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)

    img_out = None
    crops = []

    if has_grounding and "<|ref|>" in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)

    markdown = embed_images(markdown, crops)

    return cleaned, markdown, result, img_out, crops

#  PDF 
@spaces.GPU(duration=60)
def process_pdf(path, mode, task, custom_prompt, page_num):
    doc = fitz.open(path)
    total_pages = len(doc)

    if page_num < 1 or page_num > total_pages:
        doc.close()
        return f"Invalid page number (PDF has {total_pages} pages)", "", "", None, []

    page = doc.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
    img = Image.open(BytesIO(pix.tobytes("png")))
    doc.close()

    return process_image(img, mode, task, custom_prompt)

#  FILE ROUTER 
def process_file(path, mode, task, custom_prompt, page_num):
    if not path:
        return "Upload file", "", "", None, []
    if path.lower().endswith(".pdf"):
        return process_pdf(path, mode, task, custom_prompt, page_num)
    return process_image(Image.open(path), mode, task, custom_prompt)

#  UI HELPERS 
def toggle_prompt(task):
    if task == "✏️ Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Use <|grounding|> for boxes")
    elif task == "📍 Locate":
        return gr.update(visible=True, label="Text to Locate")
    return gr.update(visible=False)

def select_boxes(task):
    if task == "📍 Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()

def get_pdf_page_count(file_path):
    if not file_path or not file_path.lower().endswith(".pdf"):
        return 1
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count

def load_image(file_path, page_num=1):
    if not file_path:
        return None
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), alpha=False)
        img = Image.open(BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    return Image.open(file_path)

def update_page_selector(file_path):
    if not file_path:
        return gr.update(visible=False)
    if file_path.lower().endswith(".pdf"):
        page_count = get_pdf_page_count(file_path)
        return gr.update(
            visible=True,
            maximum=page_count,
            value=1,
            minimum=1,
            label=f"Select Page (1-{page_count})"
        )
    return gr.update(visible=False)

#  UI 
with gr.Blocks(theme=gr.themes.Soft(), title="DeepSeek-OCR") as demo:

    gr.Markdown("""
    # 🚀 DeepSeek-OCR Demo
    Upload a document (Image or PDF) to extract informations contain on an input.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                file_in = gr.File(label="Upload Document", file_types=["image", ".pdf"], type="filepath")
                
                page_selector = gr.Number(label="Page", value=1, minimum=1, step=1, visible=False, precision=0)
                
                preview_img = gr.Image(label="Preview", type="pil", height=400, interactive=False)

            mode = gr.Dropdown(list(MODEL_CONFIGS.keys()), value="Gundam", label="Mode")
            task = gr.Dropdown(list(TASK_PROMPTS.keys()), value="📋 Markdown", label="Task")
            prompt = gr.Textbox(label="Prompt", lines=2, visible=False)
            btn = gr.Button("Extract", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs() as tabs:
                with gr.Tab("Text", id="tab_text"):
                    text_out = gr.Textbox(lines=20, show_copy_button=True, show_label=False, interactive=True)
                with gr.Tab("Markdown Preview", id="tab_md"):
                    md_out = gr.Markdown("")
                with gr.Tab("Boxes", id="tab_boxes"):
                    img_out = gr.Image(type="pil", height=500, show_label=False)
                with gr.Tab("Cropped Images", id="tab_crops"):
                    gallery = gr.Gallery(show_label=False, columns=3, height=400)
                with gr.Tab("Raw Text", id="tab_raw"):
                    raw_out = gr.Textbox(lines=20, show_copy_button=True)

    file_in.change(update_page_selector, [file_in], [page_selector])
    file_in.change(load_image, [file_in, page_selector], [preview_img])
    
    page_selector.change(load_image, [file_in, page_selector], [preview_img])

    text_out.change(lambda x: x, inputs=text_out, outputs=md_out)

    task.change(toggle_prompt, [task], [prompt])
    task.change(select_boxes, [task], [tabs])

    def run(file_path, image_preview, mode, task, custom_prompt, page_num):
        if image_preview is not None:
             return process_image(image_preview, mode, task, custom_prompt)
             
        if file_path:
            return process_file(file_path, mode, task, custom_prompt, int(page_num))
            
        return "Please upload a file first.", "", "", None, []

    submit_event = btn.click(
        run,
        [file_in, preview_img, mode, task, prompt, page_selector],
        [text_out, md_out, raw_out, img_out, gallery],
    )
    submit_event.then(select_boxes, [task], [tabs])

#  LAUNCH 
if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )