"""
FastAPI backend. Exposes:
  GET  /health    - liveness check
  POST /evaluate  - accepts an answer-sheet PDF upload, runs the full pipeline,
                     returns the same JSON shape as pipeline.run_pipeline().

Run with:  uvicorn main:app --reload --port 8000
"""

import os
import shutil
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any

from pipeline import run_pipeline, save_output
from dotenv import load_dotenv
load_dotenv()  # reads backend/.env into os.environ before pipeline modules check it

app = FastAPI(title="Answer Sheet Evaluation API")

# Allow the local Vite dev server to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateResponse(BaseModel):
    results: List[Any]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        results = run_pipeline(tmp_path, save_pages_dir=None)
        save_output(results)  # also persist a copy under output/ for the deliverable
        return {"results": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    finally:
        os.unlink(tmp_path)
