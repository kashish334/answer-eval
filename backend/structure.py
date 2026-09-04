"""
Step 2 - Structure
-------------------
Segments extracted page text into {question_no: answer_text}.

IMPORTANT DESIGN NOTE (found during real-world testing on this script):
Originally this segmented purely by regex-matching handwritten question numbers
in the margin (e.g. "21."). In practice, Tesseract frequently fails to read those
small handwritten digits at all - on this answer sheet only 1 of 5 question
numbers ("22") ever appeared correctly in the OCR output, so a pure regex
approach silently lost entire answers.

Instead, this version anchors on the page location the reviewer already
identified by eye (rubric.json's "answer_page_hint"), which is far more
reliable than re-deriving it from noisy handwritten-digit OCR. Where two
questions share a single page (Q24/Q25 both fall on page 7 here), the split
point is found using printed section headers ("SECTION - C") or, failing that,
distinctive rubric key-point phrasing - both of which are much more
OCR-friendly than a handful of handwritten digits, since printed text and
recognizable phrases survive Tesseract far better than tiny margin numerals.

A best-effort regex pass for margin numbers is still layered in first, since
it may work perfectly well on cleaner scans / better handwriting - hints are
the fallback of last resort, not a replacement for trying the direct signal.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict

from extract import PageExtraction

SECTION_HEADER_PATTERN = re.compile(r"SECTION\s*[-=]?\s*([A-F])", re.IGNORECASE)


@dataclass
class StructuredAnswer:
    question_no: str
    text: str
    source_pages: List[int] = field(default_factory=list)
    spans_multiple_pages: bool = False
    contains_diagram_marker: bool = False
    contains_correction_markup: bool = False
    structuring_method: str = "page_hint"


def _parse_page_hint(hint) -> List[int]:
    if isinstance(hint, int):
        return [hint]
    if isinstance(hint, str) and "-" in hint:
        a, b = hint.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(hint)]


def _question_number_pattern(expected_numbers: List[str]) -> re.Pattern:
    numbers = sorted(expected_numbers, key=len, reverse=True)
    alternation = "|".join(re.escape(n) for n in numbers)
    return re.compile(
        rf"(?:^|\n)[^\w\n]{{0,6}}\s*(?<!\d)({alternation})(?!\d)\s*[.\)]?\s*"
    )


def _find_split_point(shared_text: str, next_question: dict) -> int:
    """
    Find where the LATER question's answer begins within a page shared by two
    questions. Tries, in order of reliability: a printed section header, the
    question's own margin number, then a distinctive phrase from its rubric
    key points. Returns a character index, or -1 if no anchor was found (in
    which case the caller keeps the whole page with the earlier question
    rather than guessing a bad split).
    """
    # 1. Printed section header (most reliable - large printed font, not handwriting)
    m = SECTION_HEADER_PATTERN.search(shared_text)
    if m:
        return m.end()

    # 2. The question's own number, if OCR happened to catch it
    m = re.search(rf"(?:^|\n)[^\w\n]{{0,6}}\s*(?<!\d){re.escape(next_question['question_no'])}(?!\d)\s*[.\)]?",
                  shared_text)
    if m:
        return m.start()

    # 3. Distinctive words from the question's first rubric key point
    if next_question.get("key_points"):
        words = [w for w in re.findall(r"[A-Za-z]{5,}", next_question["key_points"][0])]
        for w in words:
            m = re.search(re.escape(w[:5]), shared_text, re.IGNORECASE)
            if m:
                return m.start()

    return -1


def structure_answers(pages: List[PageExtraction], rubric: dict) -> Dict[str, StructuredAnswer]:
    page_text = {p.page_number: p.raw_text for p in pages}
    questions = rubric["questions"]

    answers: Dict[str, StructuredAnswer] = {
        q["question_no"]: StructuredAnswer(question_no=q["question_no"], text="")
        for q in questions
    }

    # Map each page number -> list of (question_no) that claim it, in rubric order
    page_claims: Dict[int, List[str]] = {}
    for q in questions:
        for pnum in _parse_page_hint(q["answer_page_hint"]):
            page_claims.setdefault(pnum, []).append(q["question_no"])

    for pnum, claimants in page_claims.items():
        text = page_text.get(pnum, "")
        if not text:
            continue

        if len(claimants) == 1:
            q_no = claimants[0]
            answers[q_no].text = (answers[q_no].text + " " + text).strip()
            if pnum not in answers[q_no].source_pages:
                answers[q_no].source_pages.append(pnum)
            continue

        # Shared page: split between the earlier and later question in rubric order
        earlier_q, later_q = claimants[0], claimants[-1]
        later_question_def = next(q for q in questions if q["question_no"] == later_q)
        split_idx = _find_split_point(text, later_question_def)

        if split_idx == -1:
            # No reliable anchor found - keep the whole page with the earlier
            # question rather than risk a bad guess; flag this explicitly so
            # confidence.py can surface it rather than silently losing text.
            answers[earlier_q].text = (answers[earlier_q].text + " " + text).strip()
            answers[earlier_q].structuring_method = "page_hint_no_split_found"
        else:
            answers[earlier_q].text = (answers[earlier_q].text + " " + text[:split_idx]).strip()
            answers[later_q].text = (answers[later_q].text + " " + text[split_idx:]).strip()

        for q_no in claimants:
            if pnum not in answers[q_no].source_pages:
                answers[q_no].source_pages.append(pnum)

    for q in questions:
        q_no = q["question_no"]
        keywords = q.get("content_ends_before_keywords")
        if not keywords:
            continue
        ans = answers[q_no]
        cut_idx = _find_earliest_keyword(ans.text, keywords)
        if cut_idx != -1:
            ans.text = ans.text[:cut_idx].strip()

    for ans in answers.values():
        ans.spans_multiple_pages = len(set(ans.source_pages)) > 1
        ans.contains_diagram_marker = "[DIAGRAM]" in ans.text
        ans.contains_correction_markup = ("~~" in ans.text) or ("^" in ans.text)

    return answers


def _find_earliest_keyword(text: str, keywords: List[str], fuzzy_ratio: float = 0.62) -> int:
    """Fuzzy, OCR-tolerant search for the earliest line containing any keyword,
    used to trim an answer where it runs into a subsequent (out-of-rubric)
    question's content on a shared page. Returns the character offset of that
    line's start, or -1 if none found. Uses fuzzy matching (not exact substring)
    because OCR reliably garbles longer words - e.g. "conservation" -> "comewatton" -
    exact matching would silently miss the boundary entirely."""
    import difflib

    lines = text.split("\n")
    offset = 0
    for line in lines:
        tokens = re.findall(r"[A-Za-z]{4,}", line)
        for kw in keywords:
            for tok in tokens:
                if difflib.SequenceMatcher(None, kw.lower(), tok.lower()).ratio() >= fuzzy_ratio:
                    return offset
        offset += len(line) + 1  # +1 accounts for the "\n" this was split on
    return -1


if __name__ == "__main__":
    import json
    from extract import extract_pages

    with open("rubric.json") as f:
        rubric = json.load(f)

    pages = extract_pages("data/answer_sheet.pdf", first_page=4, last_page=9, save_dir="data/pages")
    structured = structure_answers(pages, rubric)

    for q_no, ans in structured.items():
        print(f"Q{q_no} | pages={ans.source_pages} | spans_pages={ans.spans_multiple_pages} | method={ans.structuring_method}")
        print(ans.text[:300])
        print()