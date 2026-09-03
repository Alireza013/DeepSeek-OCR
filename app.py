import os
import gradio as gr
import spaces
from ocr_core.config import MODEL_CONFIGS, TASK_PROMPTS
from ocr_core.document_service import DocumentService
from ocr_core.model_service import OCRModelService

document_service = DocumentService()
ocr_service = OCRModelService()

def update_prompt(task):
    if task == "Custom":
        return gr.update(visible=True, label="Custom Prompt", placeholder="Use <|grounding|> to detect bounding boxes")
    if task == "Locate":
        return gr.update(visible=True, label="Target Text to Locate", placeholder="e.g. Invoice Number, Total Amount")
    return gr.update(visible=False, value="")

def select_result_tab(task):
    if task == "Locate":
        return gr.update(selected="tab_boxes")
    return gr.update()

def update_page_selector(file_path):
    if not file_path or not file_path.lower().endswith(".pdf"):
        return gr.update(visible=False, value=1)
    page_count = document_service.page_count(file_path)
    return gr.update(visible=True, minimum=1, maximum=page_count, value=1, label=f"PDF Page (1-{page_count})")

def preview_document(file_path, page_number=1):
    if not file_path:
        return None
    return document_service.load_page(file_path, int(page_number or 1))

def clear_outputs():
    return (
        None,  # preview image
        gr.update(visible=False, value=1),  # page_selector
        "",    # text_output
        "",    # markdown_output
        "",    # raw_output
        None,  # boxes_output
        []     # crops_output
    )

@spaces.GPU(duration=60)
def run_ocr(file_path, image_preview, mode, task, custom_prompt, page_number):
    if image_preview is None and not file_path:
        return "Please upload an image or PDF document first.", "", "", None, []
    try:
        image = image_preview or document_service.load_page(file_path, int(page_number or 1))
        return ocr_service.infer(image, mode, task, custom_prompt or "").as_tuple()
    except Exception as error:
        return f"Processing failed: {error}", "", "", None, []

# CSS with Splash Screen & Smooth Entrance Animation
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

*, *::before, *::after {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    box-sizing: border-box;
    transition: background-color 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                border-color 0.35s cubic-bezier(0.4, 0, 0.2, 1),
                color 0.25s ease,
                box-shadow 0.35s ease;
}

/* Splash / Loading Overlay */
#app-splash-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #ffffff;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: opacity 0.5s cubic-bezier(0.4, 0, 0.2, 1), visibility 0.5s ease;
}

.dark #app-splash-screen {
    background: #0b0f19;
}

#app-splash-screen.splash-hidden {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
}

/* Spinner Animation */
.splash-spinner {
    width: 52px;
    height: 52px;
    border: 4px solid rgba(2, 132, 199, 0.15);
    border-top: 4px solid #0284c7;
    border-radius: 50%;
    animation: spin 0.85s linear infinite;
    margin-bottom: 1.25rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.splash-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.02em;
}

.dark .splash-title {
    color: #f8fafc;
}

/* Entrance Animation for the Entire App Container */
.gradio-container {
    max-width: 1440px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 1.5rem !important;
    animation: appEntrance 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes appEntrance {
    0% {
        opacity: 0;
        transform: translateY(12px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Header Container */
.header-row {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    border-radius: 14px !important;
    padding: 1.5rem 2rem !important;
    margin-bottom: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.25) !important;
    border: none !important;
}

.dark .header-row {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    border: 1px solid #334155 !important;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.5) !important;
}

.header-text h1 {
    color: #ffffff !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    margin: 0 0 0.35rem 0 !important;
    letter-spacing: -0.03em !important;
}

.header-text p {
    color: rgba(255, 255, 255, 0.88) !important;
    font-size: 0.96rem !important;
    margin: 0 !important;
}

/* Theme Toggle Button */
#theme-toggle-btn {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    border-radius: 50% !important;
    background: rgba(255, 255, 255, 0.18) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    color: #ffffff !important;
    font-size: 1.25rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    padding: 0 !important;
}

#theme-toggle-btn:hover {
    background: rgba(255, 255, 255, 0.3) !important;
    transform: scale(1.08);
}

.theme-icon-animate {
    animation: iconPopSpin 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
}

@keyframes iconPopSpin {
    0% { transform: rotate(0deg) scale(0.7); opacity: 0.3; }
    50% { transform: rotate(190deg) scale(1.15); }
    100% { transform: rotate(360deg) scale(1); opacity: 1; }
}

/* Side-by-Side Dropdowns */
.dropdown-pair-row {
    display: flex !important;
    flex-direction: row !important;
    gap: 0.75rem !important;
    align-items: flex-end !important;
    width: 100% !important;
}

.dropdown-pair-row > div {
    flex: 1 1 50% !important;
    min-width: 0 !important;
}

/* Action Button */
.extract-btn {
    margin-top: 1rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

.extract-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px -2px rgba(2, 132, 199, 0.3) !important;
}

footer {
    display: none !important;
}
"""

# JavaScript: Handles Theme Initialization and Smooth Splash Dissolve
JS_THEME_AND_LOADER_INIT = """
() => {
    // 1. Synchronize Dark / Light mode based on preferences
    const isDark = document.body.classList.contains('dark') || 
                   (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (isDark) {
        document.body.classList.add('dark');
    }
    const btn = document.querySelector('#theme-toggle-btn');
    if (btn) {
        btn.innerText = document.body.classList.contains('dark') ? '☀️' : '🌙';
    }

    // 2. Hide Splash Screen smoothly once UI has fully mounted
    setTimeout(() => {
        const splash = document.getElementById('app-splash-screen');
        if (splash) {
            splash.classList.add('splash-hidden');
        }
    }, 450);
}
"""

JS_THEME_TOGGLE = """
() => {
    const isDark = document.body.classList.toggle('dark');
    const btn = document.querySelector('#theme-toggle-btn');
    if (btn) {
        btn.innerText = isDark ? '☀️' : '🌙';
        btn.classList.remove('theme-icon-animate');
        void btn.offsetWidth;
        btn.classList.add('theme-icon-animate');
    }
}
"""

with gr.Blocks(title="VisionDoc Studio | AI OCR & Document Intelligence") as demo:
    # Fullscreen Splash Screen Overlay
    gr.HTML("""
    <div id="app-splash-screen">
        <div class="splash-spinner"></div>
        <div class="splash-title">VisionDoc Studio</div>
    </div>
    """)

    # Header Row
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=10, elem_classes="header-text"):
            gr.Markdown(
                """
                # VisionDoc Studio
                Intelligent multimodal extraction — transform scans, PDF files, and images into structured Markdown.
                """
            )
        with gr.Column(scale=1, min_width=50):
            theme_toggle = gr.Button("🌙", elem_id="theme-toggle-btn")

    theme_toggle.click(None, None, None, js=JS_THEME_TOGGLE)

    with gr.Row(equal_height=False):
        # Settings Column
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Document Upload",
                file_types=["image", ".pdf"],
                type="filepath",
                file_count="single"
            )
            page_selector = gr.Number(
                label="PDF Page",
                value=1,
                minimum=1,
                step=1,
                visible=False,
                precision=0
            )
            preview = gr.Image(
                label="Preview",
                type="pil",
                height=320,
                interactive=False
            )
            
            with gr.Row(elem_classes="dropdown-pair-row", equal_height=True):
                mode = gr.Dropdown(
                    choices=list(MODEL_CONFIGS),
                    value="Gundam",
                    label="Model Mode",
                    info="Select resolution mode",
                    allow_custom_value=False,
                    filterable=False
                )
                task = gr.Dropdown(
                    choices=list(TASK_PROMPTS),
                    value="Markdown",
                    label="OCR Task",
                    info="Select extraction task",
                    allow_custom_value=False,
                    filterable=False
                )

            prompt = gr.Textbox(
                label="Custom Prompt",
                lines=3,
                visible=False
            )
            
            extract_button = gr.Button(
                "Extract Content",
                variant="primary",
                size="lg",
                elem_classes="extract-btn"
            )

        # Outputs Column (Clean state on initial load)
        with gr.Column(scale=2):
            with gr.Tabs() as result_tabs:
                with gr.Tab("Editable Text", id="tab_text"):
                    text_output = gr.Textbox(lines=22, show_label=False, value="")
                with gr.Tab("Markdown Preview", id="tab_md"):
                    markdown_output = gr.Markdown(value="")
                with gr.Tab("Bounding Boxes", id="tab_boxes"):
                    boxes_output = gr.Image(type="pil", height=520, show_label=False)
                with gr.Tab("Cropped Figures", id="tab_crops"):
                    crops_output = gr.Gallery(show_label=False, columns=3, height=420)
                with gr.Tab("Raw Model Output", id="tab_raw"):
                    raw_output = gr.Textbox(lines=22, show_label=False, value="")

    # Event Listeners
    file_input.change(update_page_selector, [file_input], [page_selector])
    file_input.change(preview_document, [file_input, page_selector], [preview])
    page_selector.change(preview_document, [file_input, page_selector], [preview])
    
    file_input.clear(
        clear_outputs,
        inputs=None,
        outputs=[preview, page_selector, text_output, markdown_output, raw_output, boxes_output, crops_output]
    )

    task.change(update_prompt, [task], [prompt])
    task.change(select_result_tab, [task], [result_tabs])
    text_output.change(lambda text: text, [text_output], [markdown_output])

    submit = extract_button.click(
        run_ocr,
        [file_input, preview, mode, task, prompt, page_selector],
        [text_output, markdown_output, raw_output, boxes_output, crops_output],
        show_progress="full",
    )
    submit.then(select_result_tab, [task], [result_tabs])

    # Trigger splash fade-out and theme synchronization on page load
    demo.load(None, None, None, js=JS_THEME_AND_LOADER_INIT)

if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        share=os.getenv("GRADIO_SHARE", "true").lower() == "true",
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
        css=CSS,
    )