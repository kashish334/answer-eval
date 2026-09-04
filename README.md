# Answer Sheet Evaluation Pipeline

CNV AI Labs timed assignment — extracts, structures, scores, and confidence-flags
a scanned CBSE Class X Social Science answer sheet against a rubric.

## Project structure

```
backend/     FastAPI + the 4-step pipeline (extract, structure, score, confidence)
frontend/    React (Vite) UI — upload a PDF, view results
output/      output.json / output.csv written after each run (created automatically)
```

## 1. Backend setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You also need **Tesseract OCR** and **poppler** installed on your system (used for
the offline OCR fallback and PDF→image conversion):

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils
```

**Optional but recommended** — set an Anthropic API key to use the Claude-vision
extractor and LLM-as-grader instead of the offline fallbacks (much better on messy
handwriting and semantic scoring):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
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
