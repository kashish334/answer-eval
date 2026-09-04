# Approach Note

1. I first sourced a genuine CBSE Class X Social Science answer sheet and its corresponding question paper to create a realistic evaluation dataset.
2. I prepared the required data by creating an answer key/model answers and defining the maximum marks and important points for each selected question.
3. I selected five short-answer questions covering different cases, including normal answers, corrections, and an answer continuing across multiple pages.
4. I then implemented the backend pipeline first, following the order: extraction, question-wise structuring, semantic scoring, confidence evaluation, and final output generation.
5. For text extraction, I used OCR with image preprocessing techniques.
6. Since handwritten text was difficult to extract accurately, I also integrated an LLM vision-based approach to improve extraction and handle corrections and non-text regions.
7. For structuring, I originally planned to segment purely by detecting handwritten question numbers in the margin, but OCR only recovered 1 of 5 margin numbers correctly on this sheet. So I anchored structuring on the page each answer starts on (identified by me as the rubric author) instead, and used printed section headers / rubric key-point phrases to find the split point when two questions share a page. A regex-based question-number pass is still in the code as a path for cleaner scans, but on this sample it isn't what actually drives segmentation — the page-hint anchoring is.
8. For evaluation, I used an LLM-based semantic grader against the prepared answer key rather than relying only on keyword matching.
9. I also used embedding similarity as an additional cross-check to determine how closely the student's answer matched the expected meaning.
10. I combined these signals to generate a confidence level and provide a reason whenever an answer required further attention.
11. After completing and testing the backend pipeline, I developed the frontend to make the complete evaluation process accessible through a simple interface.
12. One of the main difficulties I faced was extracting accurate text from handwritten answer sheets, especially when there were corrections and variations in handwriting.
13. Another challenge was managing the LLM API key and getting the vision-based extraction and grading pipeline working reliably.
14. The pipeline is currently working end-to-end, although the LLM-based processing makes the evaluation somewhat slow because of API response time.
15. With more time I'd fix two specific weak spots: (a) the page range the pipeline looks at is currently hardcoded to this sample sheet's layout, so a different scan would need that generalized — probably via layout/section detection instead of a fixed page window; and (b) structuring is currently anchored on rubric-author-supplied page hints rather than being fully automatic — I'd invest in better handwritten-digit OCR or a small margin-number detector so segmentation doesn't depend on a human already knowing where each answer is. More broadly I'd keep improving OCR/HTR, layout detection, confidence calibration, and processing speed.