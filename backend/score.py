"""
Step 3 - Score
---------------
Scores each structured answer against the rubric's key points based on MEANING,
not keyword overlap. Two signals are combined:

1. Embedding similarity (sentence-transformers) - fast, offline-after-first-download,
   used as a sanity-check signal and as the fallback if no LLM is configured.
2. LLM-as-grader (Claude) - given the rubric key points + max marks + the student's
   extracted answer, returns which key points were matched and a score with a
   justification. This is what correctly rewards a "near-correct answer phrased
   differently" - the actual assignment requirement - since embeddings alone are a
   weaker proxy for that.

If both are available, the LLM score is primary and the embedding similarity is
recorded alongside it as a cross-check signal (large disagreement -> lower confidence,
see confidence.py).
"""

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from structure import StructuredAnswer


@dataclass
class ScoreResult:
    question_no: str
    score: float
    max_score: float
    matched_points: List[str] = field(default_factory=list)
    missed_points: List[str] = field(default_factory=list)
    method: str = "embedding"
    embedding_similarity: Optional[float] = None
    llm_score: Optional[float] = None
    score_detail: str = ""


# ---------------------------------------------------------------------------
# Embedding-based scorer (offline fallback + cross-check signal)
# ---------------------------------------------------------------------------

class EmbeddingScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    MATCH_THRESHOLD = 0.35   # lowered from 0.45: OCR noise measurably drags down
                              # cosine similarity even for genuinely correct content
    FUZZY_WORD_RATIO = 0.78  # tolerance for OCR letter-substitution errors

    def score(self, answer_text: str, key_points: List[str], max_score: float,
              points_required: int) -> ScoreResult:
        from sentence_transformers import util
        import difflib
        import re as _re

        if not answer_text.strip():
            return ScoreResult(question_no="", score=0, max_score=max_score,
                                missed_points=key_points, method="embedding",
                                score_detail="No text extracted for this answer.")

        # Score against individual OCR lines rather than the whole paragraph -
        # a single strong line shouldn't get diluted by 90 other noisy words
        # when computing similarity to a short 6-word rubric phrase.
        chunks = [c.strip() for c in answer_text.split("\n") if c.strip()]
        if not chunks:
            chunks = [answer_text]

        chunk_embs = self.model.encode(chunks, convert_to_tensor=True)
        point_embs = self.model.encode(key_points, convert_to_tensor=True)
        sim_matrix = util.cos_sim(point_embs, chunk_embs)  # [n_points, n_chunks]

        answer_tokens = _re.findall(r"[A-Za-z]{4,}", answer_text)

        matched, missed, per_point_sim = [], [], []
        for i, point in enumerate(key_points):
            best_sim = float(sim_matrix[i].max())
            per_point_sim.append(best_sim)

            # Fuzzy literal-word fallback: catches cases where a key rubric
            # word survived OCR intact (or nearly so) but the surrounding
            # sentence was too garbled for the embedding to recognise.
            point_words = _re.findall(r"[A-Za-z]{4,}", point)
            fuzzy_hit = any(
                difflib.SequenceMatcher(None, pw.lower(), tok.lower()).ratio() >= self.FUZZY_WORD_RATIO
                for pw in point_words for tok in answer_tokens
            )

            if best_sim >= self.MATCH_THRESHOLD or fuzzy_hit:
                matched.append(point)
            else:
                missed.append(point)

        matched_count = min(len(matched), points_required)
        score = round((matched_count / points_required) * max_score, 2)
        mean_sim = round(sum(per_point_sim) / len(per_point_sim), 3) if per_point_sim else 0.0

        return ScoreResult(
            question_no="", score=score, max_score=max_score,
            matched_points=matched, missed_points=missed,
            method="embedding", embedding_similarity=mean_sim,
            score_detail=f"{matched_count}/{points_required} rubric points matched "
                         f"(best-line similarity >= {self.MATCH_THRESHOLD}, or a rubric "
                         f"keyword survived OCR closely enough to match directly)."
        )


# ---------------------------------------------------------------------------
# LLM-as-grader (primary method when an API key is configured)
# ---------------------------------------------------------------------------

GRADING_PROMPT_TEMPLATE = """You are grading a CBSE Class X Social Science exam answer.

Question: {question_text}
Maximum marks: {max_score}
Key points expected in a full-marks answer (any {points_required} of these earn full marks):
{key_points_list}

Student's extracted answer:
\"\"\"{answer_text}\"\"\"

Grade based on MEANING, not exact wording - a correct point phrased differently
still counts as matched. Respond with ONLY valid JSON in this exact shape, no
other text:
{{
  "score": <number, 0 to {max_score}>,
  "matched_points": [<key points the answer actually covers, in your own words>],
  "missed_points": [<key points not covered>],
  "justification": "<one sentence explaining the score>"
}}"""


class LLMScorer:
    def __init__(self):
        from anthropic import Anthropic
        self.client = Anthropic()

    def score(self, question_text: str, answer_text: str, key_points: List[str],
              max_score: float, points_required: int) -> ScoreResult:
        prompt = GRADING_PROMPT_TEMPLATE.format(
            question_text=question_text,
            max_score=max_score,
            points_required=points_required,
            key_points_list="\n".join(f"- {p}" for p in key_points),
            answer_text=answer_text if answer_text.strip() else "[NO TEXT EXTRACTED]",
        )
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in response.content if b.type == "text").strip()
        raw = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ScoreResult(question_no="", score=0, max_score=max_score,
                                method="llm", score_detail="LLM response could not be parsed.")

        return ScoreResult(
            question_no="", score=float(parsed.get("score", 0)), max_score=max_score,
            matched_points=parsed.get("matched_points", []),
            missed_points=parsed.get("missed_points", []),
            method="llm", llm_score=float(parsed.get("score", 0)),
            score_detail=parsed.get("justification", ""),
        )


# ---------------------------------------------------------------------------
# LLM-as-grader: Gemini (free-tier alternative to Claude)
# ---------------------------------------------------------------------------

class GeminiScorer:
    def __init__(self):
        from gemini_keys import RotatingGeminiClient
        self.client = RotatingGeminiClient(model_name="gemini-2.5-flash")

    def score(self, question_text: str, answer_text: str, key_points: List[str],
              max_score: float, points_required: int) -> ScoreResult:
        prompt = GRADING_PROMPT_TEMPLATE.format(
            question_text=question_text,
            max_score=max_score,
            points_required=points_required,
            key_points_list="\n".join(f"- {p}" for p in key_points),
            answer_text=answer_text if answer_text.strip() else "[NO TEXT EXTRACTED]",
        )
        response = self.client.generate_content(prompt)
        raw = (response.text or "").strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ScoreResult(question_no="", score=0, max_score=max_score,
                                method="llm", score_detail="Gemini response could not be parsed.")

        return ScoreResult(
            question_no="", score=float(parsed.get("score", 0)), max_score=max_score,
            matched_points=parsed.get("matched_points", []),
            missed_points=parsed.get("missed_points", []),
            method="llm", llm_score=float(parsed.get("score", 0)),
            score_detail=parsed.get("justification", ""),
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def score_answers(structured: Dict[str, StructuredAnswer], rubric: dict) -> Dict[str, ScoreResult]:
    llm_scorer = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            llm_scorer = LLMScorer()
        except Exception as e:
            print(f"[score] Claude scorer unavailable, trying next option: {e}")
    if llm_scorer is None:
        try:
            llm_scorer = GeminiScorer()
        except Exception as e:
            print(f"[score] Gemini scorer unavailable, falling back to embeddings only: {e}")

    embedding_scorer = None
    try:
        embedding_scorer = EmbeddingScorer()
    except Exception as e:
        print(f"[score] Embedding cross-check unavailable: {e}")

    results = {}
    for q in rubric["questions"]:
        q_no = q["question_no"]
        ans = structured.get(q_no)
        answer_text = ans.text if ans else ""

        embedding_result = None
        if embedding_scorer:
            embedding_result = embedding_scorer.score(
                answer_text, q["key_points"], q["max_score"], q["points_required_for_full_marks"]
            )

        if llm_scorer:
            result = llm_scorer.score(
                q["question_text"], answer_text, q["key_points"],
                q["max_score"], q["points_required_for_full_marks"]
            )
            if embedding_result:
                result.embedding_similarity = embedding_result.embedding_similarity
        elif embedding_result:
            result = embedding_result
        else:
            result = ScoreResult(question_no=q_no, score=0, max_score=q["max_score"],
                                  method="none", score_detail="No scoring engine available.")

        result.question_no = q_no
        results[q_no] = result

    return results


if __name__ == "__main__":
    from extract import extract_pages
    from structure import structure_answers

    with open("rubric.json") as f:
        rubric = json.load(f)

    pages = extract_pages("data/answer_sheet.pdf", first_page=4, last_page=9, save_dir="data/pages")
    structured = structure_answers(pages, rubric)
    scores = score_answers(structured, rubric)

    for q_no, s in scores.items():
        print(f"Q{q_no}: {s.score}/{s.max_score} ({s.method}) - {s.score_detail}")