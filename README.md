# DeepSeek-OCR Studio

A Gradio web application powered by **DeepSeek-OCR** for converting images and PDF documents into editable text, Markdown, visual grounding boxes, and cropped figures.

## Features

- Image and PDF upload with page selection and preview
- Markdown conversion, free OCR, image description, text location, and custom prompts
- Editable text output with a live Markdown preview
- Bounding-box visualization and extracted figure crops
- Safe grounding-coordinate parsing with `ast.literal_eval`
- Lazy model loading and reliable temporary-file cleanup
- Portable font selection without a hard-coded operating-system path
- Client-side light and dark theme switching

## Requirements

- Python 3.10 or newer
- A CUDA-compatible GPU is strongly recommended
- Model weights are downloaded from Hugging Face during the first inference
- Gradio 6.0 or newer

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Run locally

```powershell
$env:GRADIO_SHARE="false"
python app.py
```

The default server address and port can be changed with `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`.

## Run in Google Colab

Upload or clone this repository into the Colab runtime, then run the cells in `DeepSeek-OCR.ipynb` in order. The notebook searches below its current working directory for `requirements.txt` and `app.py`, so it does not depend on a fixed Colab path.

For a public Gradio link in Colab, set `GRADIO_SHARE=true` before running the application. The application uses port `7860` by default.

## Project structure

```text
.
├── app.py
├── ocr_core/
│   ├── config.py
│   ├── document_service.py
│   ├── font.py
│   ├── model_service.py
│   └── output_formatter.py
├── DeepSeek-OCR.ipynb
├── requirements.txt
└── README.md
```

`app.py` owns the Gradio composition and event wiring. The `ocr_core` package separates configuration, document I/O, model inference, font loading, and output formatting so each responsibility can be tested and changed independently.

## Usage

1. Upload an image or PDF document.
2. Select a PDF page when applicable.
3. Select a model mode and task.
4. Enter a prompt for `Locate` or `Custom` tasks when requested.
5. Click `Extract text`.
6. Review or edit the text, Markdown, raw output, boxes, and cropped figures.

This application uses *DeepSeek-OCR*, a specialized multimodal model designed for high-accuracy text extraction and document understanding.