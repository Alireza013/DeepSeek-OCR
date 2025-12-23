# 🚀 DeepSeek-OCR Web UI

A powerful and interactive Optical Character Recognition (OCR) web application powered by **DeepSeek-OCR**. This tool allows users to convert images and PDF documents into structured Markdown, extract raw text, and visually locate elements within documents using a modern Gradio interface.

## ✨ Key Features

* **Multimodal Input:** Unified upload interface for both Images (`.jpg`, `.png`) and PDF documents (`.pdf`).
* **Smart PDF Handling:** Automatically converts PDF pages to images with a built-in page selector.
* **Interactive Editing:** Real-time text editor where changes to the extracted text are instantly reflected in the Markdown preview.
* **Visual Grounding:** accurate bounding box visualization to locate specific text or elements within the image.
* **Multiple Modes:** specialized configurations (e.g., *Gundam*, *Tiny*, *Base*) for different performance needs and image resolutions.
* **Structured Output:** Generates clean Markdown with embedded cropped images for figures/charts.

## 🛠️ Installation (Based on `DeepSeek-OCR` file)

1.  **Install dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    apt-get install -y fonts-dejavu-core
    ```

    *Note: A GPU with CUDA support is highly recommended for reasonable inference speeds.*

## 🚀 Usage

1.  Run the application:
    ```bash
    python app.py
    ```

2.  Open your browser and navigate to the URLs provided in the terminal (public or local URL).

3.  **How to use:**
    * **Upload:** Drop an image or PDF file in the input box.
    * **Preview:** Use the page selector (for PDFs) to view specific pages.
    * **Settings:** Choose the Model Mode (e.g., 'Gundam' for high detail) and Task (e.g., 'Markdown' or 'Locate').
    * **Extract:** Click the "Extract" button.
    * **Edit:** Go to the "Text" tab to refine the output; the "Markdown Preview" will update automatically.

## 📦 Requirements

Major dependencies include:
* `torch` & `torchvision`
* `transformers`
* `gradio`
* `PyMuPDF` (fitz)
* `accelerate`

See [requirements.txt](requirements.txt) for the full list.

## 🤖 Model

This application uses *DeepSeek-OCR*, a specialized multimodal model designed for high-accuracy text extraction and document understanding.