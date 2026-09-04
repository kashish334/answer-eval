# Approach Note

Fill this in with your own reflections once you've run the pipeline end to end —
the points below are a starting skeleton based on the pipeline as built, not a
finished submission.

1. Sourced a real CBSE Class X Social Science model answer script (Q.P. Code 087)
   from CBSE's official archive, and matched it to its exact question paper
   (32/3/2) by cross-referencing MCQ answer text against 18 candidate papers —
   this gave a genuine rubric-worthy source rather than an invented one.
2. Chose 5 short-answer questions (Q21–Q25, 2–3 marks each) deliberately covering
   a clean answer, two handwriting-correction cases (caret insertion,
   strike-through), and one real page-spanning answer (Q25), rather than
   cherry-picking only the easiest cases.
3. Extraction defaults to Tesseract + OpenCV preprocessing (deskew, denoise,
   adaptive threshold) for a fully offline baseline, and upgrades to Claude's
   vision model when an API key is present — the vision model is explicitly
   prompted to mark corrections and skip diagrams, which plain OCR cannot do.
4. Structuring uses a simple stateful "current question" pointer rather than a
   trained layout model — good enough for this script's clear margin numbering,
   but would misfire on scripts with inconsistent numbering.
5. Scoring combines an LLM-as-grader (primary, meaning-based) with an embedding
   similarity cross-check (sentence-transformers) — disagreement between the two
   is itself used as a confidence signal.
6. Confidence flagging is rule-based and traceable (each flag names the exact
   signal that triggered it) rather than a black-box score, which matters more
   for an oral defence than a marginally more "clever" ML confidence model.
7. Weakest part of the pipeline: the diagram-detection heuristic in extract.py
   is a rough edge-density heuristic, not a trained layout model — it would
   likely misfire on real maps/diagrams beyond this specific script.
8. Also weak: question-number detection assumes numbers are legible and in the
   expected margin; a badly OCR'd number would silently misattribute an answer.
9. Only tested against one student's handwriting style — confidence thresholds
   (e.g. the 0.45 embedding similarity cutoff) are not calibrated against a
   labeled dataset of correct/incorrect matches.
10. With more time: fine-tune or swap in a proper HTR model (TrOCR) trained on
    Indian exam handwriting; add a real layout-detection model (LayoutParser)
    instead of the edge-density heuristic; build a small calibration set of
    known-correct/incorrect answers to tune confidence thresholds against actual
    human-grader disagreement, addressing the 15–20% variance mentioned in the brief.
11. At scale (millions of pages, mixed scripts): the LLM-as-grader step is the
    main cost/latency bottleneck — would batch requests, cache rubric embeddings,
    and use the offline embedding scorer as a cheap first-pass filter, only
    escalating uncertain/borderline cases to the LLM grader.
12. Confidence calibration itself would need to be validated against real
    multi-evaluator disagreement data before being trusted in production, not
    just against this pipeline's own internal signals.
