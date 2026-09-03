from io import BytesIO
from pathlib import Path
import pymupdf
from PIL import Image

class DocumentService:
    @staticmethod
    def page_count(file_path: str) -> int:
        if not file_path or Path(file_path).suffix.lower() != ".pdf":
            return 1
        with pymupdf.open(file_path) as document:
            return len(document)

    @staticmethod
    def load_page(file_path: str, page_number: int = 1) -> Image.Image:
        if Path(file_path).suffix.lower() != ".pdf":
            with Image.open(file_path) as image:
                return image.copy()
        with pymupdf.open(file_path) as document:
            if not document:
                raise ValueError("The PDF does not contain any pages")
            page_index = max(0, min(int(page_number) - 1, len(document) - 1))
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), alpha=False)
            return Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")