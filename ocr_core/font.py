from pathlib import Path
from typing import Optional
from PIL import ImageFont

# Candidate font files to search for locally and in system paths
_FONT_CANDIDATES = (
    "PlusJakartaSans-Bold.ttf",
    "PlusJakartaSans-Regular.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
    "LiberationSans-Regular.ttf",
)

def load_font(size: int = 30, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Load a readable TrueType font for image overlays without depending on fixed OS paths."""
    base_dir = Path(__file__).resolve().parent.parent  # Project root directory
    
    # 1. Custom provided path
    candidates = [Path(font_path)] if font_path else []
    
    # 2. Look for project-local font files in the project root
    for name in _FONT_CANDIDATES:
        candidates.append(base_dir / name)
        
    # 3. System lookup by filename
    for name in _FONT_CANDIDATES:
        candidates.append(Path(name))
        
    # Try loading the first candidate that succeeds
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except (OSError, TypeError):
            continue
            
    # Fallback to standard bitmap default font if none are found
    return ImageFont.load_default()