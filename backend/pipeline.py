"""
Orchestrates Extract -> Structure -> Score -> Confidence for a given answer sheet
PDF, and produces the final output list required by the assignment:
question no., extracted answer text, score, score vs. answer key, confidence, reason.
"""

import json
import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()  # reads backend/.env so `python pipeline.py` also picks up ANTHROPIC_API_KEY

from extract import extract_pages
from structure import structure_answers
from score import score_answers
from confidence import flag_confidence


def load_rubric(rubric_path: str = "rubric.json") -> dict:
    with open(rubric_path) as f:
        return json.load(f)


def run_pipeline(pdf_path: str, rubric_path: str = "rubric.json",
                  first_page: int = 4, last_page: int = 9,
                  save_pages_dir: Optional[str] = "data/pages") -> List[dict]:
    rubric = load_rubric(rubric_path)
    expected_numbers = [q["question_no"] for q in rubric["questions"]]

    pages = extract_pages(pdf_path, first_page, last_page, save_dir=save_pages_dir)
    structured = structure_answers(pages, rubric)
    scores = score_answers(structured, rubric)
    confidences = flag_confidence(structured, scores)

    output = []
    for q in rubric["questions"]:
        q_no = q["question_no"]
        ans = structured.get(q_no)
        sc = scores.get(q_no)
        conf = confidences.get(q_no)

        output.append({
            "question_no": q_no,
            "question_text": q["question_text"],
            "extracted_answer_text": ans.text if ans else "",
            "source_pages": ans.source_pages if ans else [],
            "spans_multiple_pages": ans.spans_multiple_pages if ans else False,
            "score": sc.score if sc else 0,
            "max_score": q["max_score"],
            "score_vs_answer_key": sc.score_detail if sc else "",
            "matched_points": sc.matched_points if sc else [],
            "missed_points": sc.missed_points if sc else [],
            "confidence": conf.confidence if conf else "low",
            "confidence_reason": conf.reason if conf else "pipeline error",
        })
    return output


def save_output(results: List[dict], out_json: str = "output/output.json",
                 out_csv: str = "output/output.csv"):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_no", "extracted_answer_text", "score", "max_score",
            "score_vs_answer_key", "confidence", "confidence_reason"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})


if __name__ == "__main__":
    results = run_pipeline("data/answer_sheet.pdf")
    save_output(results)
    print(json.dumps(results, indent=2))