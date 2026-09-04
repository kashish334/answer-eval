"""
Step 1 - Extract
-----------------
Turns page images from the scanned answer sheet into raw text.

Two pluggable extractors are provided:

1. TesseractExtractor  - free, fully offline, works okay on clean handwriting.
                          Used as the default / fallback engine.
2. ClaudeVisionExtractor - uses a multimodal LLM (Claude) to read the page image.
                          Much stronger on messy handwriting, and can be prompted
                          to explicitly flag corrections (insertions / strike-throughs)
                          and skip diagram regions - which plain OCR cannot do.
                          Only used if ANTHROPIC_API_KEY is set in the environment.

extract_pages() picks whichever engine is available and returns a uniform structure
so the rest of the pipeline doesn't care which one ran.
"""

import os
import io
import base64
import json
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path


@dataclass
class PageExtraction:
    page_number: int
    raw_text: str
    engine: str
    mean_ocr_confidence: Optional[float] = None  # 0-100, Tesseract only
    has_correction_markup: bool = False           # e.g. ~~struck~~ or ^inserted^
    image_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(pil_image):
    """Deskew, denoise, and binarize a scanned page for better OCR results."""
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold handles uneven scan lighting better than a global one
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # Deskew using minAreaRect on text pixel coordinates
    coords = np.column_stack(np.where(thresh < 255))
    angle = 0.0
    if len(coords) > 100:
        rect_angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + rect_angle) if rect_angle < -45 else -rect_angle
        if abs(angle) < 5:  # only correct small skews, big values usually mean noise
            (h, w) = thresh.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
    return thresh


def detect_diagram_regions(pil_image, text_density_threshold=0.02):
    """
    Very lightweight layout heuristic: large contiguous blank/graphic blocks with
    little text-like contour density are flagged as non-text (diagram/map) regions
    so OCR isn't run on them and downstream steps know to skip that span.
    Returns a list of (x, y, w, h) boxes.
    """
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape
    diagram_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.05 * w_img * h_img:
            continue  # too small to be a diagram block
        region = edges[y:y + h, x:x + w]
        density = np.count_nonzero(region) / (w * h)
        if density < text_density_threshold:
            diagram_boxes.append((x, y, w, h))
    return diagram_boxes


# ---------------------------------------------------------------------------
# Engine 1: Tesseract (offline default)
# ---------------------------------------------------------------------------

class TesseractExtractor:
    name = "tesseract"

    def extract(self, pil_image, page_number: int) -> PageExtraction:
        import pytesseract
        import cv2
        import numpy as np

        # Heavy binarization tends to wipe out connected cursive strokes entirely,
        # which is why Tesseract can return zero words rather than just poor text.
        # Use a lighter grayscale + contrast pass instead, and an explicit PSM mode
        # that assumes a single uniform block of text (better fit for a lined
        # answer sheet than Tesseract's default auto-segmentation).
        gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
        contrast = cv2.convertScaleAbs(gray, alpha=1.4, beta=0)
        config = "--psm 6"

        data = pytesseract.image_to_data(
            contrast, config=config, output_type=pytesseract.Output.DICT
        )

        # Group words into lines using Tesseract's block/par/line indices, instead
        # of flattening the whole page into one space-joined string. Without real
        # line breaks, structure.py has no way to detect a question number at the
        # "start of a line" - which was silently dropping entire answers.
        lines = {}
        confidences = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            conf = float(data["conf"][i])
            if conf >= 0:
                confidences.append(conf)
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)

        text = "\n".join(" ".join(words) for words in lines.values())
        mean_conf = sum(confidences) / len(confidences) if confidences else None

        return PageExtraction(
            page_number=page_number,
            raw_text=text,
            engine=self.name,
            mean_ocr_confidence=mean_conf,
        )


# ---------------------------------------------------------------------------
# Engine 2: Claude vision (optional, stronger on handwriting)
# ---------------------------------------------------------------------------

class ClaudeVisionExtractor:
    name = "claude-vision"

    PROMPT = (
        "You are transcribing a single page from a handwritten CBSE Class X "
        "Social Science exam answer sheet. Transcribe exactly what is written, "
        "in reading order, including the question numbers in the margin.\n\n"
        "Formatting rules:\n"
        "- Wrap any struck-through / crossed-out text like this: ~~struck text~~\n"
        "- Wrap any caret-inserted text (written above the line) like this: ^inserted text^\n"
        "- If a diagram, map, or table interrupts the answer, insert the marker "
        "[DIAGRAM] at that point instead of describing it.\n"
        "- If a word is genuinely illegible, write [ILLEGIBLE] rather than guessing.\n"
        "- Do not summarize or correct grammar - transcribe verbatim.\n"
        "Return only the transcription, no preamble."
    )

    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def extract(self, pil_image, page_number: int) -> PageExtraction:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": img_b64
                    }},
                    {"type": "text", "text": self.PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        has_markup = "~~" in text or "^" in text

        return PageExtraction(
            page_number=page_number,
            raw_text=text,
            engine=self.name,
            has_correction_markup=has_markup,
        )


# ---------------------------------------------------------------------------
# Engine 3: Gemini vision (free-tier alternative to Claude)
# ---------------------------------------------------------------------------

class GeminiVisionExtractor:
    name = "gemini-vision"

    PROMPT = ClaudeVisionExtractor.PROMPT  # identical transcription instructions

    def __init__(self):
        from gemini_keys import RotatingGeminiClient
        self.client = RotatingGeminiClient(model_name="gemini-2.5-flash")

    def extract(self, pil_image, page_number: int) -> PageExtraction:
        response = self.client.generate_content([self.PROMPT, pil_image])
        text = (response.text or "").strip()
        has_markup = "~~" in text or "^" in text

        return PageExtraction(
            page_number=page_number,
            raw_text=text,
            engine=self.name,
            has_correction_markup=has_markup,
        )


# ---------------------------------------------------------------------------
# Engine 4: OpenAI vision (third rotation option)
# ---------------------------------------------------------------------------

class OpenAIVisionExtractor:
    name = "openai-vision"

    PROMPT = ClaudeVisionExtractor.PROMPT

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY from env

    def extract(self, pil_image, page_number: int) -> PageExtraction:
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
        )
        text = (response.choices[0].message.content or "").strip()
        has_markup = "~~" in text or "^" in text

        return PageExtraction(
            page_number=page_number,
            raw_text=text,
            engine=self.name,
            has_correction_markup=has_markup,
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def get_extractor():
    """Prefer Claude vision, then Gemini vision (rotating across any configured
    GEMINI_API_KEY_1/_2/_3... keys), else fall back to Tesseract."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeVisionExtractor()
        except Exception as e:
            print(f"[extract] Claude vision unavailable, trying next option: {e}")
    try:
        return GeminiVisionExtractor()
    except Exception as e:
        print(f"[extract] Gemini vision unavailable, falling back to Tesseract: {e}")
    return TesseractExtractor()


def pdf_to_page_images(pdf_path: str, first_page: int, last_page: int, dpi: int = 200):
    return convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)


def extract_pages(pdf_path: str, first_page: int, last_page: int,
                   save_dir: Optional[str] = None) -> List[PageExtraction]:
    extractor = get_extractor()
    images = pdf_to_page_images(pdf_path, first_page, last_page)

    results = []
    for offset, img in enumerate(images):
        page_number = first_page + offset
        result = extractor.extract(img, page_number)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"page_{page_number}.png")
            img.save(path)
            result.image_path = path

        results.append(result)
    return results


if __name__ == "__main__":
    # Quick manual test: extract pages 4-8 (Q21-25) of the sample answer sheet.
    pages = extract_pages(
        "data/answer_sheet.pdf", first_page=4, last_page=9, save_dir="data/pages"
    )
    for p in pages:
        print(f"--- Page {p.page_number} ({p.engine}) ---")
        print(p.raw_text[:500])
        print()