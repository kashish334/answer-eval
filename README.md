# Answer Sheet Evaluation Pipeline

Timed assignment (CNV AI Labs) — extracts, structures, scores, and
confidence-flags a scanned CBSE Class X Social Science answer sheet against
a hand-written rubric.

**Source sheet & rubric:** see [`source.MD`](./source.MD)
**Approach note :** see [`Approach.md`](./Approach.md)

## Demo Video of Project

https://drive.google.com/file/d/14bDvctZgqKKdPIl_OtmJoQ8YIfoqXdDR/view?usp=sharing

## Git Repository link

https://github.com/kashish334/answer-eval

## Project structure

```
backend/     FastAPI + the 4-step pipeline (extract, structure, score, confidence)
frontend/    React (Vite) UI — upload a PDF, view results
output/      output.json / output.csv written after each run
```

## 1. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need **Tesseract OCR** and **poppler** installed on your system (used
for the offline OCR fallback and PDF→image conversion):

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils
```

**Optional but recommended** — set an API key to use an LLM-vision extractor
and LLM-as-grader instead of the offline fallbacks (much better on messy
handwriting and semantic scoring). Checked in this priority order, first
match wins:

```bash
# 1. Claude — vision extraction + LLM grading
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Gemini — rotates across GEMINI_API_KEY_1 / _2 / _3 on quota errors,
#    falls back to a single GEMINI_API_KEY if no numbered keys are set
export GEMINI_API_KEY_1=...

# 3. none set -> fully offline: Tesseract OCR + embedding-only scoring
```

Run the backend:

```bash
uvicorn main:app --reload --port 8000
```

Check it's up: open http://localhost:8000/health — should return `{"status": "ok"}`.

### Run the pipeline standalone (no server), for quick testing

```bash
cd backend
python pipeline.py
```

This processes `data/answer_sheet.pdf` against `rubric.json` and writes
`output/output.json` and `output/output.csv`.

## 2. Frontend setup

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Upload `backend/data/answer_sheet.pdf`, click
**Run evaluation**, and the results will render once the backend responds.

## Data / rubric

- `backend/data/answer_sheet.pdf` — CBSE Class X Social Science model answer
  script (Q.P. Code 087), sourced from
  https://www.cbse.gov.in/cbsenew/model-answer.html
- `backend/data/question_paper.pdf` — the matching official question paper (Q.P. 32/3/2)
- `backend/rubric.json` — the answer key for the 5 questions attempted (Q21–Q25),
  written from the NCERT/CBSE syllabus content for this paper

## Pipeline design notes

**Extraction** tries, in order: Claude vision → Gemini vision (rotating across
any `GEMINI_API_KEY_N` keys on quota errors) → Tesseract OCR, so the pipeline
degrades gracefully rather than failing outright if no LLM key is set.

**Structuring** does *not* rely purely on OCR'd handwritten question numbers
in the margin. On this sample, plain OCR only recovered 1 of 5 margin numbers
correctly, so segmentation instead anchors on the page each question lives on
(`rubric.json`'s `answer_page_hint` — identified by a human reviewer up
front). Where two questions share a page, the split point is found using, in
order of reliability: a printed section header (e.g. "SECTION-C"), the OCR'd
margin number if it happened to survive, or a fuzzy match against the next
question's first rubric key point. A regex pass over margin numbers is also
defined (`structure.py::_question_number_pattern`) as a path for cleaner
scans where handwritten digits OCR reliably, but on this sample document it
is not the operative mechanism — the page-hint anchoring is.

**Scoring** combines an LLM-as-grader (primary, when a key is configured)
with sentence-embedding similarity as an independent cross-check signal. The
two are expected to roughly agree; large disagreement between them is itself
used as a low-confidence signal.

**Confidence** is rule-based, not a black box: each label traces back to a
concrete cause (empty/short extraction, correction markup, diagram
interruption, page-span stitching, or LLM/embedding disagreement), so a
grader can see *why* something was flagged, not just that it was.

## Known limitations

- **Page range is hardcoded to the sample document.** `run_pipeline()`
  defaults to pages 4–9, matching `data/answer_sheet.pdf`. The frontend's
  upload flow uses this same default for any file, so uploading a
  differently-paginated scan will extract the wrong pages rather than fail
  loudly. A generalized version would need to auto-detect the answer section
  boundaries per upload rather than assume a fixed page window.
- **Structuring depends on a human-supplied page hint per question**, not on
  fully automatic question-number detection (see design note above). This is
  a reasonable adaptation given real OCR performance on handwritten margin
  digits, but it means the pipeline currently needs a rubric author to know
  roughly where each answer starts, rather than segmenting cold from raw
  text.
- **LLM calls make evaluation slow** (dominated by vision-extraction +
  grading API round-trips per page/question) and require an API key for best
  quality; the fully offline path (Tesseract + embeddings only) is
  noticeably weaker on handwriting and paraphrased answers.
- **No committed sample output.** `backend/output/` is gitignored so the repo
  doesn't show evaluator-facing evidence of a completed run without
  installing and running the pipeline — worth checking in one reference
  `output.csv` / `output.json` alongside the code.
