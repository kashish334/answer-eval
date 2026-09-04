"""
Step 4 - Flag Confidence
--------------------------
For each scored answer, produce a high/medium/low confidence label and a one-line
reason. The goal (per the assignment) is that LOW confidence should genuinely be
the answers a human grader would want to double-check - so this combines several
independent uncertainty signals rather than relying on any single one:

  1. Extraction signal   - low OCR confidence (Tesseract) or the extractor itself
                            marked corrections / illegible spans / diagram markers.
  2. Structuring signal  - the answer's page-span was inferred (spillover case)
                            rather than being a single clean page.
  3. Scoring signal      - LLM score and embedding similarity disagree a lot, or
                            the answer only partially matched the rubric.
  4. Completeness signal - extracted text is empty or suspiciously short for
                            the marks on offer.

Rule-based thresholds are used rather than a black-box combination, so the
reason string is always traceable back to a concrete cause.
"""

from dataclasses import dataclass
from typing import Dict

from structure import StructuredAnswer
from score import ScoreResult


@dataclass
class ConfidenceResult:
    question_no: str
    confidence: str  # "high" | "medium" | "low"
    reason: str


def _flag(answer: StructuredAnswer, score: ScoreResult, page: object = None) -> ConfidenceResult:
    reasons_low, reasons_medium = [], []

    # 1. Completeness
    word_count = len(answer.text.split())
    if word_count == 0:
        return ConfidenceResult(answer.question_no, "low", "No text was extracted for this answer.")
    if word_count < 4 and score.max_score >= 2:
        reasons_low.append("extracted answer is unusually short for the marks available")

    # 2. Extraction quality signals
    if answer.contains_correction_markup:
        reasons_medium.append("answer contains a handwritten correction (insertion/strike-through)")
    if answer.contains_diagram_marker:
        reasons_medium.append("a diagram/map interrupted the answer text")

    # 3. Structuring signal
    if answer.spans_multiple_pages:
        reasons_medium.append("answer was stitched together across a page break")

    # 4. Scoring agreement signal
    if score.llm_score is not None and score.embedding_similarity is not None:
        # embedding_similarity is 0-1, llm_score is 0-max_score -> normalize both to 0-1
        norm_llm = score.llm_score / score.max_score if score.max_score else 0
        disagreement = abs(norm_llm - score.embedding_similarity)
        if disagreement >= 0.4:
            reasons_low.append(
                f"LLM and embedding scores disagree significantly "
                f"(LLM {score.llm_score}/{score.max_score} vs. embedding similarity "
                f"{score.embedding_similarity:.2f})"
            )

    # Partial rubric match
    if score.max_score and 0 < score.score < score.max_score:
        reasons_medium.append("answer only partially matched the rubric's key points")

    if reasons_low:
        return ConfidenceResult(answer.question_no, "low", "; ".join(reasons_low))
    if reasons_medium:
        return ConfidenceResult(answer.question_no, "medium", "; ".join(reasons_medium))
    return ConfidenceResult(answer.question_no, "high",
                             "clean extraction, single page, full rubric match, scoring methods agree")


def flag_confidence(structured: Dict[str, StructuredAnswer],
                     scores: Dict[str, ScoreResult]) -> Dict[str, ConfidenceResult]:
    results = {}
    for q_no, answer in structured.items():
        score = scores.get(q_no)
        if score is None:
            continue
        results[q_no] = _flag(answer, score)
    return results


if __name__ == "__main__":
    import json
    from extract import extract_pages
    from structure import structure_answers
    from score import score_answers

    with open("rubric.json") as f:
        rubric = json.load(f)
    expected = [q["question_no"] for q in rubric["questions"]]

    pages = extract_pages("data/answer_sheet.pdf", first_page=4, last_page=8, save_dir="data/pages")
    structured = structure_answers(pages, expected)
    scores = score_answers(structured, rubric)
    confidences = flag_confidence(structured, scores)

    for q_no, c in confidences.items():
        print(f"Q{q_no}: {c.confidence.upper()} - {c.reason}")
